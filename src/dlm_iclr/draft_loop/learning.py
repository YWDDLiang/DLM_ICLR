"""Draft-only LoRA distillation + matched-mask denoising preferences.

This is an explicitly labelled surrogate, NOT exact DPO of the C1 execution
trajectory. F and physics never participate in its autograd graph.
"""
from __future__ import annotations
from contextlib import contextmanager
from pathlib import Path
import random
import re
import torch
import torch.nn.functional as F
from .common import digest, read_rows, seed_for, write_json


def geometry_positions(n: int) -> list[int]:
    return list(range(1, 7)) + [8 + 4*i + a for a in range(3) for i in range(n)]


class TypedVocabulary:
    def __init__(self, tokenizer):
        self.tables = {k: {} for k in ("LA", "LB", "LC", "AA", "AB", "AG", "X", "Y", "Z")}
        for text, token in tokenizer.get_vocab().items():
            match = re.fullmatch(r"<(LA|LB|LC|AA|AB|AG|X|Y|Z)_(\d+)>", text)
            if match:
                self.tables[match[1]][int(match[2])] = int(token)
        if any(not v for v in self.tables.values()):
            raise ValueError("Incomplete crystal vocabulary")

    @staticmethod
    def family(pos):
        return ("LA", "LB", "LC", "AA", "AB", "AG")[pos-1] if pos < 7 else "XYZ"[(pos-8) % 4]

    def log_probs(self, vector, pos, temperature):
        family = self.family(pos)
        table = self.tables[family]
        keys = sorted(table)
        ids = [table[k] for k in keys]
        values = vector[ids].float()
        if family in "XYZ" and 100 in table:
            j0, j1 = keys.index(0), keys.index(100)
            merged = torch.logaddexp(values[j0], values[j1])
            values = torch.cat((merged[None], values[[i for i in range(len(ids)) if i not in (j0, j1)]]))
            ids = [table[0]] + [v for i, v in enumerate(ids) if i not in (j0, j1)]
        return (values / temperature).log_softmax(-1), ids

    def canonical(self, body):
        out = list(body)
        for pos in geometry_positions((len(body)-7)//4):
            family = self.family(pos)
            if family in "XYZ" and out[pos] == self.tables[family].get(100):
                out[pos] = self.tables[family][0]
        return out


def construction_patterns(generated, mask_id):
    """Replay observed mask shapes; visible values still come from each target.

    This is teacher forcing on construction-shaped views, not replay of the
    original prefix values or an exact C1 trajectory likelihood.
    """
    trace=generated.get('trace',{})
    events=trace.get('periodic_axis',{}).get('events',[])
    patterns=[]
    for event in events:
        body=event['input_body'];n=(len(body)-7)//4;axis=event['axis']
        masked=[p for p in geometry_positions(n) if body[p]==mask_id]
        previous=[8+4*i+a for a in range(axis) for i in range(n)]
        future=[8+4*i+a for a in range(axis+1,3) for i in range(n)]
        active=list(event['active_positions'])
        if (active and all(p in masked for p in active+future)
                and not any(p in masked for p in list(range(1,7))+previous)):
            patterns.append({'axis':axis,'active_positions':active,'masked_positions':masked})
    # The first construction attempt masks all coordinates during lattice work.
    # Recovery canvases can retain coordinates, so do not infer their cell masks.
    recovery=trace.get('construction_recovery',{})
    if events and not recovery.get('recoveries_used',0) and not trace.get('lattice_gamma_last',False):
        n=(len(events[0]['input_body'])-7)//4
        coordinates=[8+4*i+a for a in range(3) for i in range(n)]
        for event in trace.get('construction_geometry',{}).get('events',[]):
            if event.get('stage')=='lattice' and event.get('active_positions'):
                active=list(event['active_positions'])
                patterns.append({'axis':-1,'active_positions':active,'masked_positions':active+coordinates})
    return patterns


def mask_view(body, seed, mask_id, low, high, *, strategy='random', patterns=()):
    if (len(body)-7) % 4 or len(body) < 11:
        raise ValueError("Bad crystal token length")
    positions = geometry_positions((len(body)-7)//4)
    rng = random.Random(seed)
    if strategy == 'lattice_anchor':
        observed=[p for p in patterns if p['axis']==-1]
        selected=list(rng.choice(observed)['active_positions']) if observed else list(range(1,7))
        noisy=list(body)
        masked=selected+[p for p in positions if p>=8]
        for p in masked:noisy[p]=mask_id
        return noisy,selected,len(masked)/len(positions)
    if strategy == 'construction':
        n=(len(body)-7)//4
        groups=[list(range(1,7))]+[[8+4*i+a for i in range(n)] for a in range(3)]
        phase=rng.randrange(4)
        observed=[p for p in patterns if p['axis']==phase-1]
        if observed:
            pattern=rng.choice(observed)
            selected=list(pattern['active_positions'])
            masked=list(pattern['masked_positions'])
        else:
            active=groups[phase].copy();rng.shuffle(active)
            selected=active[rng.randrange(len(active)):]
            masked=selected+[p for group in groups[phase+1:] for p in group]
        noisy=list(body)
        for p in masked:noisy[p]=mask_id
        return noisy,selected,len(masked)/len(positions)
    if strategy != 'random':raise ValueError('Unknown mask strategy: '+strategy)
    probability = rng.uniform(low, high)
    selected = [p for p in positions if rng.random() < probability]
    if not selected:
        selected = [rng.choice(positions)]
    noisy = list(body)
    for p in selected:
        noisy[p] = mask_id
    return noisy, selected, probability


def forward_score(model, tokenizer, vocab, row, tokens, mask_seed, recipe, mask_id):
    """The same mask_seed produces identical position masks for a same-Plan pair."""
    return forward_scores(model,tokenizer,vocab,[(row,tokens,mask_seed)],recipe,mask_id)[0]


def forward_scores(model, tokenizer, vocab, examples, recipe, mask_id):
    """Right-padded true micro-batch with per-example matched position masks."""
    device = next(model.parameters()).device
    prepared=[]
    for row,tokens,mask_seed in examples:
        target=vocab.canonical(tokens)
        noisy,selected,_=mask_view(target,mask_seed,mask_id,recipe['mask_min'],recipe['mask_max'],
            strategy=recipe.get('mask_strategy','random'),patterns=row.get('construction_masks',()))
        prefix=tokenizer(row.get('body_prompt',row.get('prompt')),add_special_tokens=False)['input_ids']
        if len(prefix)+len(target)>recipe['max_length']:
            raise ValueError('No truncation of crystal or its Plan is permitted')
        prepared.append((prefix+noisy,len(prefix),target,selected))
    width=max(len(x[0]) for x in prepared)
    ids=torch.full((len(prepared),width),tokenizer.pad_token_id,device=device,dtype=torch.long)
    attention=torch.zeros_like(ids)
    for i,(tokens,_,_,_) in enumerate(prepared):
        ids[i,:len(tokens)]=torch.tensor(tokens,device=device);attention[i,:len(tokens)]=1
    result=model(ids,attention_mask=attention)
    scores=[]
    for i,(_,offset,target,selected) in enumerate(prepared):
        vectors,losses=[],[]
        for p in selected:
            lp,legal=vocab.log_probs(result.logits[i,offset+p],p,recipe['temperature'])
            if target[p] not in legal: raise ValueError('Target is not in typed output support')
            losses.append(-lp[legal.index(target[p])]);vectors.append(lp)
        scores.append((-torch.stack(losses).mean(),vectors))
    return scores


def preference_loss(chosen, rejected, ref_chosen, ref_rejected, beta):
    return -F.logsigmoid(beta * ((chosen-ref_chosen) - (rejected-ref_rejected)))


@contextmanager
def reference_parameters(named, reference):
    live = {n: p.detach().clone() for n, p in named}
    try:
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(reference[name].to(parameter.device))
        yield
    finally:
        with torch.no_grad():
            for name, parameter in named:
                parameter.copy_(live[name])


def train_draft(base_model, adapter, teachers, pairs, replay, history, output, settings, *, device="cuda:0"):
    from dlm_iclr.runtime.models import load_model_and_tokenizer
    from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    recipe = settings["training"]
    if not teachers or not replay:
        raise ValueError("Both verified teachers and original TRAIN replay are required")
    model, tokenizer = load_model_and_tokenizer(base_model, adapter, torch.device(device), mean_resizing=False)
    if not hasattr(model, "peft_config"):
        raise ValueError("Expected existing B0 PEFT asset; do not silently retrain an 8B model")
    named = []
    for name, p in model.named_parameters():
        p.requires_grad_("lora_A" in name or "lora_B" in name)
        if p.requires_grad:
            p.data = p.data.float()
            named.append((name, p))
    if not named:
        raise ValueError("No trainable draft LoRA parameters")
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.p = 0
        setter = getattr(module, "set_activation_checkpointing", None)
        if callable(setter) and hasattr(module, "transformer"):
            setter("whole_layer")
    model.enable_input_require_grads()
    model.eval()  # fixed dropout; autograd remains enabled
    reference = {name: p.detach().cpu().clone() for name, p in named}
    optimizer = torch.optim.AdamW([p for _, p in named], lr=recipe["learning_rate"], weight_decay=0)
    vocab = TypedVocabulary(tokenizer)
    paired = {r["source_id"]: r for r in pairs}
    data_key = digest([teachers, pairs, [(r["source_id"], r["body_token_ids"]) for r in replay], history])
    checkpoint = output / "resume.pt"
    start = 0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if saved["recipe"] != recipe or saved["input_adapter"] != str(adapter) or saved.get("data_key") != data_key:
            raise ValueError("Training resume identity changed")
        with torch.no_grad():
            for name, p in named:
                p.copy_(saved["parameters"][name].to(p.device))
        optimizer.load_state_dict(saved["optimizer"])
        reference, start = saved["reference"], saved["step"]
    logs = []
    for step in range(start, recipe["updates"]):
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        anchor_total = 0.0
        micro_size=recipe.get('micro_batch_size',1)
        for offset in range(0,recipe['effective_batch_size'],micro_size):
            examples=[];positive_indexes=[];negative_indexes=[]
            for micro in range(offset,min(offset+micro_size,recipe['effective_batch_size'])):
                seed=seed_for(settings['seed'],'draft_train',step,micro)
                rng=random.Random(seed);choice=rng.random()
                if choice<settings['replay_fraction']:
                    row=rng.choice(replay);pair=None
                elif history and choice<settings['replay_fraction']+settings['history_fraction']:
                    row=rng.choice(history);pair=None
                else:
                    row=rng.choice(teachers);pair=paired.get(row['source_id'])
                positive=row['body_token_ids']
                use_pair=pair is not None and recipe['preference_weight']>0 and step>=recipe['preference_warmup_updates']
                positive_indexes.append(len(examples));examples.append((row,positive,seed))
                if use_pair:
                    negative=pair['rejected_tokens']
                    if len(negative)!=len(positive): raise ValueError('Preference pair changed cardinality')
                    negative_indexes.append(len(examples));examples.append((row,negative,seed))
                else: negative_indexes.append(None)
            # All reference passes precede live passes, so autograd version counters are valid.
            anchor_weight=float(recipe.get('lattice_anchor_kl',0.0))
            if anchor_weight:
                anchor_recipe=dict(recipe,mask_strategy='lattice_anchor')
                anchors=[(examples[pi][0],examples[pi][0].get('source_body_token_ids',examples[pi][1]),examples[pi][2])
                         for pi in positive_indexes]
            with reference_parameters(named, reference), torch.no_grad():
                refs=forward_scores(model,tokenizer,vocab,examples,recipe,MASK_TOKEN_ID)
                if anchor_weight:
                    anchor_refs=forward_scores(model,tokenizer,vocab,anchors,anchor_recipe,MASK_TOKEN_ID)
            live=forward_scores(model,tokenizer,vocab,examples,recipe,MASK_TOKEN_ID)
            if anchor_weight:
                anchor_live=forward_scores(model,tokenizer,vocab,anchors,anchor_recipe,MASK_TOKEN_ID)
            losses=[]
            for example_index,(pi,ni) in enumerate(zip(positive_indexes,negative_indexes,strict=True)):
                pos,vectors=live[pi];ref_pos,ref_vectors=refs[pi]
                kl=torch.stack([(q.exp()*(q-p)).sum() for p,q in zip(vectors,ref_vectors,strict=True)]).mean()
                loss=-pos+recipe['reference_kl']*kl
                if ni is not None:
                    loss=loss+recipe['preference_weight']*preference_loss(pos,live[ni][0],ref_pos,refs[ni][0],recipe['preference_beta'])
                if anchor_weight:
                    anchor_kl=torch.stack([(q.exp()*(q-p)).sum() for p,q in
                        zip(anchor_live[example_index][1],anchor_refs[example_index][1],strict=True)]).mean()
                    loss=loss+anchor_weight*anchor_kl
                    anchor_total+=float(anchor_kl.detach())/recipe['effective_batch_size']
                losses.append(loss)
            loss=torch.stack(losses).sum()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite draft objective")
            (loss / recipe["effective_batch_size"]).backward()
            total += float(loss.detach()) / recipe["effective_batch_size"]
        torch.nn.utils.clip_grad_norm_([p for _, p in named], 1.0, error_if_nonfinite=True)
        optimizer.step()
        report = {"step": step+1, "loss": total, "preference_active": step >= recipe["preference_warmup_updates"]}
        if anchor_weight:report['lattice_context_reference_KL']=anchor_total
        logs.append(report)
        if (step+1) % recipe["save_every"] == 0 or step+1 == recipe["updates"]:
            temporary = output / "resume.tmp.pt"
            torch.save({"parameters": {n:p.detach().cpu() for n,p in named}, "reference":reference,
                        "optimizer":optimizer.state_dict(), "step":step+1, "recipe":recipe,
                        "input_adapter":str(adapter), "data_key":data_key}, temporary)
            temporary.replace(checkpoint)
            write_json(output / "progress.json", report)
            print({"stage":"draft_training", **report}, flush=True)
    # B0 uses modules_to_save for trained IO tables. Preserve them despite freezing their gradients.
    if not any(getattr(c, "modules_to_save", None) for c in model.peft_config.values()):
        raise ValueError("B0 trained vocabulary tables must be explicitly preserved in PEFT modules_to_save")
    target = output / "checkpoint"
    model.save_pretrained(target, save_embedding_layers=False)
    tokenizer.save_pretrained(target)
    write_json(output / "training.json", {"objective":"verified_SFT+reference_KL+matched_mask_preference_surrogate",
               "mask_strategy":recipe.get('mask_strategy','random'),
               "not_exact_C1_trajectory_DPO":True, "updates":recipe["updates"],
               "trainable_parameters":sum(p.numel() for _,p in named), "IO_tables_frozen_and_saved":True,
               "teachers":len(teachers), "pairs":len(pairs), "history":logs})
    del model, optimizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return str(target)


def adapt_c1(base_model, adapter, old_head, rows, output, settings, *, device="cuda:0"):
    from dlm_iclr.runtime.models import load_model_and_tokenizer
    from dlm_iclr.c1.trainer import load_head
    from dlm_iclr.c1.objectives import AxisTrainingSchema, axis_view, forward_views, loss_from_prediction
    from .common import digest
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    model, tokenizer = load_model_and_tokenizer(base_model, adapter, torch.device(device), mean_resizing=False)
    model.eval().requires_grad_(False)
    head = load_head(old_head, device).train().requires_grad_(True)
    schema = AxisTrainingSchema(tokenizer)
    prepared = []
    for row in rows:
        row = dict(row)
        body = list(row["body_token_ids"])
        for i in range(row["plan_state"]["N"]):
            for axis in range(3):
                pos = 8 + 4*i + axis
                body[pos] = schema.axis_tokens[axis][schema.coord_map[axis][body[pos]] % 100]
        row.update(body_token_ids=body, _source_hash=row["source_id"], _target_hash=digest(body))
        prepared.append(row)
    optimizer = torch.optim.Adam(head.parameters(), lr=settings["training"]["c1_learning_rate"])
    for step in range(settings["training"]["c1_updates"]):
        rng = random.Random(seed_for(settings["seed"], "head", step))
        batch = [rng.choice(prepared) for _ in range(settings['training'].get('c1_batch_size',4))]
        views = [axis_view(r, schema, seed=seed_for(settings["seed"], step, i), epoch=step)
                 for i,r in enumerate(batch)]
        for view, source in zip(views, batch, strict=True):
            view["supervision"] = "verified_draft_teacher" if source.get("origin") else "original_TRAIN_replay"
            view["view_hash"] = digest({k:v for k,v in view.items() if k != "view_hash"})
        predictions, _ = forward_views(model, tokenizer, views, max_length=settings["training"]["max_length"])
        optimizer.zero_grad(set_to_none=True)
        losses = [loss_from_prediction(head, schema, v, p, temperature=0.7)[0]
                  for v,p in zip(views,predictions,strict=True)]
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
        optimizer.step()
    temporary = output.with_suffix(".tmp.pt")
    torch.save({"schema":"periodic_axis_head_v1", "head_config":head.config(),
                "state_dict":head.state_dict(), "warm_started_from":str(old_head),
                "draft_adapter":str(adapter), "fresh_hidden_only":True}, temporary)
    temporary.replace(output)
    del model, head, optimizer
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()
    return str(output)
