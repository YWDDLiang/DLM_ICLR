"""Bounded source-balanced fitting of existing verified complete geometries.

Teacher prefixes are supervised views, not recorded student trajectories.
No preference loss, physical backpropagation, or generation occurs here.
"""
from collections import Counter, defaultdict
from pathlib import Path
import random

import torch

from .common import digest, read_json, seed_for, write_json
from .learning import TypedVocabulary, forward_scores


def phase_positions(n):
    if not isinstance(n, int) or n < 1:
        raise ValueError('Positive atom count required')
    return [list(range(1, 7))]+[[8+4*i+a for i in range(n)] for a in range(3)]


def source_phase(rows, draw, seed):
    """Every four passes visit every source in every phase exactly once."""
    if not rows or draw < 0 or len({r['source_id'] for r in rows}) != len(rows):
        raise ValueError('Unique nonempty sources and nonnegative cursor required')
    cycle, offset = divmod(draw, len(rows))
    order = sorted(range(len(rows)), key=lambda i: rows[i]['source_id'])
    random.Random(seed_for(seed, 'teacher_pass', cycle)).shuffle(order)
    index = order[offset]
    phases = [0, 1, 2, 3]
    random.Random(seed_for(seed, 'source_phases', rows[index]['source_id'])).shuffle(phases)
    return index, phases[cycle % 4], cycle//4


def teacher_view(row, phase, pass_index, seed, mask_id, vocabulary):
    target = vocabulary.canonical(row['body_token_ids'])
    if len(target) < 11 or (len(target)-7) % 4 or mask_id in target:
        raise ValueError('Complete crystal teacher required')
    n = (len(target)-7)//4
    if row['plan_state']['N'] != n or phase not in range(4):
        raise ValueError('Teacher layout or phase mismatch')
    groups = phase_positions(n); selected = groups[phase].copy()
    if pass_index % 2:
        rng = random.Random(seed_for(seed, row['source_id'], phase, pass_index, 'partial_teacher'))
        rng.shuffle(selected); selected = sorted(selected[rng.randrange(len(selected)):])
    body = target.copy()
    for p in selected+[p for g in groups[phase+1:] for p in g]: body[p] = mask_id
    return {'source_id': row['source_id'], 'phase': phase, 'pass_index': pass_index,
            'prompt': row.get('body_prompt', row.get('prompt')),
            'input_body': body, 'target': target, 'positions': selected,
            'view_semantics': 'consistent_teacher_prefix'}


def score_views(model, tokenizer, vocabulary, views, recipe, *, detailed=False):
    device = next(model.parameters()).device; prepared = []
    for view in views:
        prefix = tokenizer(view['prompt'], add_special_tokens=False)['input_ids']
        ids = prefix+view['input_body']
        if len(ids) > recipe['max_length']: raise ValueError('No crystal truncation allowed')
        prepared.append((ids, len(prefix)))
    width = max(len(x[0]) for x in prepared)
    ids = torch.full((len(views), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    for i, (tokens, _) in enumerate(prepared):
        ids[i, :len(tokens)] = torch.tensor(tokens, device=device); attention[i, :len(tokens)] = 1
    prediction = model(ids, attention_mask=attention); losses = []; details = []
    for i, (view, (_, offset)) in enumerate(zip(views, prepared, strict=True)):
        terms = []; fields = defaultdict(list)
        for p in view['positions']:
            lp, legal = vocabulary.log_probs(prediction.logits[i, offset+p], p, recipe['temperature'])
            target = view['target'][p]; index = legal.index(target); terms.append(-lp[index])
            if detailed:
                family = vocabulary.family(p); bins = {v:k for k,v in vocabulary.tables[family].items()}
                chosen = legal[int(lp.argmax())]; distance = abs(bins[chosen]-bins[target])
                if family in 'XYZ': distance = min(distance, 100-distance)
                field = 'lattice_lengths' if p < 4 else 'lattice_angles' if p < 7 else 'coordinates'
                fields[field].append((float(-lp[index]), int(chosen == target), int(distance <= 1), distance))
        losses.append(torch.stack(terms).mean())
        if detailed:
            for field, values in fields.items():
                details.append({'source_id': view['source_id'], 'phase': view['phase'], 'field': field,
                    'tokens': len(values), 'NLL': sum(v[0] for v in values)/len(values),
                    'top1': sum(v[1] for v in values)/len(values),
                    'within_one_bin': sum(v[2] for v in values)/len(values),
                    'mean_bin_error': sum(v[3] for v in values)/len(values)})
    return losses, details


def evaluate_fit(model, tokenizer, vocabulary, rows, recipe, seed, mask_id, budget=None):
    views = [teacher_view(r, phase, 0, seed, mask_id, vocabulary) for r in rows for phase in range(4)]
    losses, details = [], []
    with torch.no_grad():
        for start in range(0, len(views), recipe['micro_batch_size']):
            if budget: budget.reserve('diagnostic_DLM_forwards')
            ll, dd = score_views(model, tokenizer, vocabulary,
                views[start:start+recipe['micro_batch_size']], recipe, detailed=True)
            losses.extend(float(v) for v in ll); details.extend(dd)
    groups = defaultdict(list)
    for row in details: groups[row['field']].append(row)
    summary = {k:{metric:sum(x[metric] for x in v)/len(v)
                 for metric in ['NLL','top1','within_one_bin','mean_bin_error']} for k,v in groups.items()}
    return {'macro_NLL': sum(losses)/len(losses), 'fields': summary, 'rows': details,
            'sources': len(rows), 'teacher_conditioned_not_free_generation': True}


def fitting_progress_gate(before, after):
    progress = {}; accepted = True
    for field in ['lattice_lengths','coordinates']:
        a, b = before['fields'][field], after['fields'][field]
        reduction = (a['NLL']-b['NLL'])/max(a['NLL'], 1e-8)
        ok = reduction >= .2 or b['top1'] >= .9 or b['within_one_bin'] >= .95
        progress[field] = {'relative_NLL_reduction': reduction, 'top1': b['top1'],
                          'within_one_bin': b['within_one_bin'], 'progress': ok}
        accepted = accepted and ok
    return {'ready_for_bounded_generation_check': accepted, 'fields': progress,
            'exact_token_reproduction_not_required_for_physical_success': True,
            'scientific_effect_established': False}


def feedback_view(row):
    """Keep the recorded input and supervise only its observed commit positions."""
    view=row['positive_view'];body=list(view['input_body']);target=body.copy()
    if digest(view['prompt'])!=row['input_prompt_key'] or digest(body)!=row['input_prefix_key']:
        raise ValueError('Verified feedback prompt or visible prefix changed')
    for p,t in zip(view['positions'],view['tokens'],strict=True):
        if body[p]!=view['mask_id']:raise ValueError('Feedback target was already visible')
        target[p]=t
    return {'source_id':view['source_id'],'phase':view['axis']+1,'prompt':view['prompt'],
            'input_body':body,'target':target,'positions':list(view['positions'])}


class PrefixFeedbackSampler:
    def __init__(self, rows, seed):
        self.groups=defaultdict(lambda:defaultdict(list));self.seed=seed
        for row in rows:
            if not row['weight']>0:raise ValueError('Positive verified sampling weight required')
            v=feedback_view(row);self.groups[v['source_id']][v['phase']].append(row)
        self.sources=sorted(self.groups)
        if not self.sources:raise ValueError('Verified prefix feedback required')

    def sample(self,index):
        cycle,offset=divmod(index,len(self.sources));order=self.sources.copy()
        random.Random(seed_for(self.seed,'feedback_source_pass',cycle)).shuffle(order)
        source=order[offset];phases=sorted(self.groups[source])
        random.Random(seed_for(self.seed,source,'feedback_phases')).shuffle(phases)
        phase=phases[cycle%len(phases)];pool=self.groups[source][phase]
        row=random.Random(seed_for(self.seed,'feedback_draw',index)).choices(pool,
            weights=[r['weight'] for r in pool],k=1)[0]
        return feedback_view(row)


def evaluate_conditioned_fit(model,tokenizer,vocab,teachers,feedback,replay,recipe,seed,mask_id,budget):
    report=evaluate_fit(model,tokenizer,vocab,teachers,recipe,seed,mask_id,budget)
    sampler=PrefixFeedbackSampler(feedback,seed)
    chosen=[feedback_view(random.Random(seed_for(seed,s,p,'fixed_probe')).choice(sampler.groups[s][p]))
            for s in sampler.sources for p in sorted(sampler.groups[s])]
    losses=[]
    with torch.no_grad():
        for start in range(0,len(chosen),recipe['micro_batch_size']):
            if budget:budget.reserve('diagnostic_DLM_forwards')
            ll,_=score_views(model,tokenizer,vocab,chosen[start:start+recipe['micro_batch_size']],recipe)
            losses.extend(float(v) for v in ll)
    report['prefix_macro_NLL']=sum(losses)/len(losses)
    # Model selection remains on frozen teacher/prefix fitting, not new physics.
    teacher_weight=recipe.get('fit_selection_teacher_weight',.5)
    if not 0<=teacher_weight<=1:raise ValueError('Invalid fixed fitting selection weight')
    report['selection_NLL']=teacher_weight*report['macro_NLL']+(1-teacher_weight)*report['prefix_macro_NLL']
    report['fixed_prefix_views']=len(chosen)
    return report


def train_existing_teachers(base_model, adapter, rows, output, recipe, *, seed, device='cuda:0', budget=None,
                           feedback=None,replay=None):
    from dlm_iclr.runtime.models import load_model_and_tokenizer
    from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    key = digest([rows, recipe, seed, str(adapter)])
    mixture=recipe.get('microbatch_mixture',['feedback','teacher','feedback','replay'])
    if feedback is not None:
        if not mixture or any(k not in ['feedback','teacher','replay'] for k in mixture):
            raise ValueError('Invalid fixed microbatch mixture')
        if 'feedback' not in mixture or 'teacher' not in mixture:raise ValueError('Feedback and teacher views required')
        if 'replay' in mixture and not replay:raise ValueError('TRAIN replay required for conditional correction')
        key=digest([key,feedback,[(r['source_id'],r['body_token_ids']) for r in replay or []]])
    torch.manual_seed(seed)
    model, tokenizer = load_model_and_tokenizer(base_model, adapter, torch.device(device), mean_resizing=False)
    named = []
    for name,p in model.named_parameters():
        p.requires_grad_('lora_A' in name or 'lora_B' in name)
        if p.requires_grad: p.data=p.data.float(); named.append((name,p))
    if not named or not hasattr(model, 'peft_config'):
        raise ValueError('Real draft LoRA parameters required')
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout): module.p = 0
        setter = getattr(module, 'set_activation_checkpointing', None)
        if callable(setter) and hasattr(module, 'transformer'): setter('whole_layer')
    model.enable_input_require_grads(); model.eval()
    optimizer = torch.optim.AdamW([p for _,p in named], lr=recipe['learning_rate'], weight_decay=0)
    vocabulary = TypedVocabulary(tokenizer); start = 0; best = float('inf'); best_step = 0
    sampler=PrefixFeedbackSampler(feedback,seed) if feedback is not None else None
    replay_recipe={**recipe,'mask_min':.15,'mask_max':.85,'mask_strategy':'random'}
    def evaluate():
        if sampler is not None:
            return evaluate_conditioned_fit(model,tokenizer,vocabulary,rows,feedback,replay,recipe,seed,MASK_TOKEN_ID,budget)
        return evaluate_fit(model,tokenizer,vocabulary,rows,recipe,seed,MASK_TOKEN_ID,budget)
    def selection(report):return report.get('selection_NLL',report['macro_NLL'])
    resume = output/'resume.pt'
    saved_exposure=None;total_kinds=Counter()
    if resume.exists():
        saved = torch.load(resume, map_location='cpu', weights_only=False)
        if saved['identity'] != key: raise ValueError('Teacher fitting resume identity changed')
        with torch.no_grad():
            for name,p in named: p.copy_(saved['parameters'][name].to(p.device))
        optimizer.load_state_dict(saved['optimizer']); start = saved['step']
        best, best_step = saved['best'], saved['best_step']
        saved_exposure=saved.get('source_phase_counts')
        total_kinds.update(saved.get('mixture_views_total',{}))
    def save_model(folder):
        if not any(getattr(c,'modules_to_save',None) for c in model.peft_config.values()):
            raise ValueError('Frozen trained IO tables must be preserved')
        model.save_pretrained(folder, save_embedding_layers=False); tokenizer.save_pretrained(folder)
    if not (output/'fit_000.json').exists():
        if start: raise ValueError('Missing original fitting baseline')
        baseline = evaluate()
        write_json(output/'fit_000.json',baseline); best=selection(baseline); save_model(output/'best')
    else:
        baseline = read_json(output/'fit_000.json')
        if not start: best=selection(baseline)
    counts = Counter();kinds=Counter()
    if saved_exposure is not None:
        counts.update({(s,p):c for s,p,c in saved_exposure})
    elif sampler is None:
        for draw in range(start*recipe['effective_batch_size']):
            i,phase,_=source_phase(rows,draw,seed); counts[(rows[i]['source_id'],phase)] += 1
    elif start:
        raise ValueError('Conditional feedback resume lacks exposure counters')
    for step in range(start, recipe['updates']):
        if budget: budget.reserve('teacher_fit_updates')
        optimizer.zero_grad(set_to_none=True); total = 0.
        for offset in range(0,recipe['effective_batch_size'],recipe['micro_batch_size']):
            size=min(recipe['micro_batch_size'],recipe['effective_batch_size']-offset)
            if sampler is None:
                kind='teacher';views=[]
                for j in range(offset,offset+size):
                    i,phase,pass_index=source_phase(rows,step*recipe['effective_batch_size']+j,seed)
                    views.append(teacher_view(rows[i],phase,pass_index,seed,MASK_TOKEN_ID,vocabulary))
            else:
                if recipe['effective_batch_size']%recipe['micro_batch_size']:raise ValueError('Exact microbatch mixture required')
                micro=step*(recipe['effective_batch_size']//recipe['micro_batch_size'])+offset//recipe['micro_batch_size']
                cycle,part=divmod(micro,len(mixture));kind=mixture[part]
                kind_micro=cycle*mixture.count(kind)+mixture[:part].count(kind)
                if kind=='feedback':
                    cursor=kind_micro*recipe['micro_batch_size']
                    views=[sampler.sample(cursor+j) for j in range(size)]
                elif kind=='teacher':
                    views=[]
                    for j in range(size):
                        i,phase,pass_index=source_phase(rows,kind_micro*recipe['micro_batch_size']+j,seed)
                        views.append(teacher_view(rows[i],phase,pass_index,seed,MASK_TOKEN_ID,vocabulary))
                else:
                    rng=random.Random(seed_for(seed,'TRAIN_replay_micro',micro))
                    examples=[(r,r['body_token_ids'],seed_for(seed,micro,j,'TRAIN_mask'))
                              for j in range(size) for r in [rng.choice(replay)]]
            if kind=='replay':
                scored=forward_scores(model,tokenizer,vocabulary,examples,replay_recipe,MASK_TOKEN_ID)
                losses=[-x[0] for x in scored]
            else:
                losses,_=score_views(model,tokenizer,vocabulary,views,recipe)
                for view in views:counts[(view['source_id'],view['phase'])]+=1
            kinds[kind]+=size;total_kinds[kind]+=size
            loss=torch.stack(losses).sum()/recipe['effective_batch_size']
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite teacher fitting loss')
            loss.backward();total+=float(loss.detach())
        grad=torch.nn.utils.clip_grad_norm_([p for _,p in named],1.,error_if_nonfinite=True)
        optimizer.step()
        if (step+1)%recipe['evaluate_every']==0 or step+1==recipe['updates']:
            report=evaluate()
            write_json(output/f'fit_{step+1:03d}.json',report)
            if selection(report) < best:
                best,best_step=selection(report),step+1;save_model(output/'best')
        if (step+1)%recipe['save_every']==0 or step+1==recipe['updates']:
            temporary=output/'resume.tmp.pt'
            torch.save({'identity':key,'step':step+1,'parameters':{n:p.detach().cpu() for n,p in named},
                'optimizer':optimizer.state_dict(),'best':best,'best_step':best_step,
                'source_phase_counts':[(s,p,c) for (s,p),c in counts.items()],
                'mixture_views_total':dict(total_kinds)},temporary)
            temporary.replace(resume)
            progress={'step':step+1,'updates':recipe['updates'],'loss':total,'gradient_norm':float(grad),
                'best_NLL':best,'best_step':best_step,'sources':len(rows),'sampled_views':(step+1)*recipe['effective_batch_size'],
                'source_phase_min':min(counts.values()),'source_phase_max':max(counts.values()),
                'covered_source_phases':len(counts),'mixture_views_this_process':dict(kinds),
                'mixture_views_total':dict(total_kinds),
                'objective':'verified_actual_prefix_correction_plus_teacher_TRAIN_replay' if sampler else 'verified_teacher_geometry_SFT'}
            write_json(output/'progress.json',progress);print(progress,flush=True)
    save_model(output/'last')
    selected=read_json(output/f'fit_{best_step:03d}.json')
    result={'identity':key,'selected_checkpoint':str(output/'best'),'best_step':best_step,
            'before':baseline,'selected':selected,'gate':fitting_progress_gate(baseline,selected),
            'exposure': [{'source_id':s,'phase':p,'views':c} for (s,p),c in sorted(counts.items())],
            'pure_SFT_diagnostic_not_generalization':True,'IO_tables_frozen_and_saved':True}
    if sampler is not None:
        result.update(conditional_feedback=True,mixture_views_this_process=dict(kinds),
            mixture_views_total=dict(total_kinds),
            source_mixture={k:mixture.count(k)/len(mixture) for k in sorted(set(mixture))},
            input_prefix_preserved=True,physical_gain_not_yet_established=True,
            best_selected_on_teacher_and_prefix_NLL=True)
    write_json(output/'REPORT.json',result)
    del model,optimizer,named;return result
