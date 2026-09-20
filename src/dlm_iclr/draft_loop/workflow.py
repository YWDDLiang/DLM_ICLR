"""Bounded synchronous fresh-Plan -> verified teacher -> draft-learning rounds."""
from __future__ import annotations
from collections import Counter
from copy import deepcopy
from pathlib import Path
import random
from .common import (BASE_COMMIT, SCHEMA, Budget, digest, exclusive_run, model_identity,
                     read_json, read_rows, seed_for, shuffled, write_json, write_rows)
from .conditions import ConditionLedger
from .quality import Measurement, raw_utility, publication_gate
from .data import compile_teachers, balanced_history


def refresh_reference(config, plans, settings):
    """Extend the run-local reference before measurements, never inside pairs."""
    if settings["refresh_hull"]:
        from dlm_iclr.runtime.config import run_root
        from dlm_iclr.evaluation.hull import query
        query(plans,run_root(config)/"hull",batch_size=config["evaluation"]["hull_batch_size"])


def should_stop(round_id, consecutive, settings, *, collection=False):
    patience = settings["max_collection_stalls"] if collection else settings["max_stalled_rounds"]
    return round_id+1 >= settings["min_rounds_before_stopping"] and consecutive >= patience


def initial_assets(config):
    from dlm_iclr.runtime.config import asset
    return {"base":asset(config,"dlm"),"draft":asset(config,"b0"),"head":asset(config,"c1"),"verifier":None}


def preflight(config, output, settings):
    import importlib.util
    from dlm_iclr.runtime.config import asset,run_root
    from dlm_iclr.c1.generation import Constructor
    import inspect
    if "joint_controller" not in inspect.signature(Constructor).parameters:
        raise RuntimeError("Missing additive C1 controller patch")
    for name in ("torch","transformers","peft","numpy","pymatgen","chgnet","torch_geometric","ase"):
        if importlib.util.find_spec(name) is None:
            raise ImportError(f"Required baseline dependency is missing: {name}")
    root = run_root(config)
    required = [Path(asset(config,k)) for k in ("planner","planner_base","dlm","b0","c1","diffusion")]
    required += [root/"data/structures/train.jsonl",root/"data/plans/val.jsonl"]
    # GPU compute is offline: require local assets instead of triggering implicit HF downloads.
    missing = [str(p) for p in required if not p.exists()]
    if settings["require_hull"] and not settings["refresh_hull"]:
        if not (root/"hull/official_slim_cache.jsonl").exists():
            missing.append(str(root/"hull/official_slim_cache.jsonl"))
    if missing:
        raise FileNotFoundError("Missing explicit local assets/data: " + "; ".join(missing))
    out = Path(output).resolve()
    for p in required:
        p = p.resolve()
        if out == p or out in p.parents:
            raise ValueError("Loop output cannot contain/overwrite baseline assets")
    return {"schema":SCHEMA,"base_commit":BASE_COMMIT,"local_assets_verified":True,
            "no_old_C2_editor":True,"output":str(out)}


def prepare_development(config,root,settings,ledger):
    from dlm_iclr.runtime.config import run_root
    path=root/"development_plans.jsonl"
    all_val=read_rows(run_root(config)/"data/plans/val.jsonl")
    # Reservation uses composition identities only, never test outcomes.
    ledger.reserve(all_val,"validation")
    test=run_root(config)/"data/plans/test.jsonl"
    if test.exists(): ledger.reserve(read_rows(test),"test_identity_reservation")
    if path.exists(): return read_rows(path)
    selected=shuffled(all_val,settings["development_seed"])[:settings["development_sources"]]
    if len(selected)<settings["development_sources"]:
        raise ValueError("Not enough independent development requests")
    result=[]
    for i,row in enumerate(selected):
        row=deepcopy(row); row["ordinal"]=i; row["original_ordinal"]=i
        row["provenance"]["usage_role"]="development"
        row["body_noise_seed"]=seed_for(settings["development_seed"],row["source_id"],"draft")
        row["refiner_noise_seed"]=seed_for(settings["development_seed"],row["source_id"],"F")
        result.append(row)
    write_rows(path,result)
    return result


def fresh_plans(config,folder,round_id,settings,ledger,budget,device):
    from dlm_iclr.runtime.config import backend_config
    from dlm_iclr.planner.sampling import generate_plans
    from dlm_iclr.data.plans import load_plans
    from .backend import release
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    output=folder/"plans.jsonl"
    if output.exists(): return read_rows(output)
    accepted=[]; attempts=[]; counts=Counter()
    for batch in range(settings["max_planner_batches"]):
        drawfile=folder/f"planner_draws_{batch:03d}.jsonl"
        if not drawfile.exists():
            size=(settings["plans_per_round"]-len(accepted))*settings["planner_oversample"]
            budget.reserve("planner_draws",size)
            generate_plans(backend_config(config).assets,str(drawfile),requests=size,
                           seed=seed_for(settings["seed"],round_id,batch,"fresh_planner"),device=device,
                           usage_role="train",batch_size=config["planner"]["sampling"]["batch_size"],
                           temperature=settings["planner_temperature"],
                           top_p=config["planner"]["sampling"]["top_p"],
                           top_k=config["planner"]["sampling"]["top_k"],
                           max_new_tokens=config["planner"]["sampling"]["max_new_tokens"])
            release()
        draws,_=load_plans(str(drawfile),legal_only=False,seed=settings["seed"])
        for draw in draws:
            if len(accepted)>=settings["plans_per_round"]: break
            row=deepcopy(draw)
            row["source_id"]=f"loop:{round_id}:batch{batch}:"+draw["source_id"]
            row["provenance"].update(usage_role="train",loop_round=round_id,
                                      planner_weights_updated=False)
            ok,reason=ledger.admit(row,round_id,settings["max_same_composition_per_round"],counts)
            attempts.append({"source_id":row["source_id"],"admitted":ok,"reason":reason})
            if ok:
                row["ordinal"]=len(accepted); row["original_ordinal"]=len(accepted)
                accepted.append(row)
        if len(accepted)>=settings["plans_per_round"]: break
    write_rows(folder/"admission.jsonl",attempts)
    write_json(folder/"admission_summary.json",{"draws_considered":len(attempts),"accepted":len(accepted),
                  "reasons":dict(Counter(a["reason"] for a in attempts)),
                  "admission_uses_no_physical_scores":True})
    write_rows(output,accepted)
    return accepted


def _reindex(records):
    return [dict(r,ordinal=i,original_ordinal=i) for i,r in enumerate(records)]


def collect_data(config,assets,plans,folder,settings,budget,device,guide=None):
    from .backend import sample_drafts,collect_teacher_candidates,measure,quantized_record,release
    from transformers import AutoTokenizer
    folder=Path(folder)
    if (folder/"bundles.jsonl").exists(): return read_rows(folder/"bundles.jsonl")
    drafts=sample_drafts(config,assets,plans,folder/"drafts",settings,budget,guide=guide,device=device)
    candidates=collect_teacher_candidates(config,assets,plans,drafts,folder/"teachers",settings,budget,device=device)
    flat=[]; per_source=[]
    for i,group in enumerate(candidates):
        ids=[]
        for j,row in enumerate(group["candidates"]):
            row=dict(row,source_id=plans[i]["source_id"]+f":qteacher{j}")
            ids.append(len(flat)); flat.append(row)
        per_source.append(ids)
    # All noisy/rounded alternatives are screened as actual decoded crystals, not by F time.
    screening=measure(config,_reindex(flat),folder/"screening",settings,budget,device=device,singlepoint=True) if flat else []
    chosen=[]; selected_by_source=[]
    for ids in per_source:
        valid=[j for j in ids if raw_utility(Measurement.from_dict(screening[j]["measurement"]),
                                           require_hull=settings["require_hull"]) is not None]
        valid.sort(key=lambda j:raw_utility(Measurement.from_dict(screening[j]["measurement"]),
                                           require_hull=settings["require_hull"]),reverse=True)
        indexes=[]
        for j in valid[:settings["teacher_shortlist"]]:
            indexes.append(len(chosen)); chosen.append(flat[j])
        selected_by_source.append(indexes)
    raw_records=[d["record"] for d in drafts]
    measurements=measure(config,_reindex(raw_records+chosen),folder/"full_labels",settings,budget,device=device)
    raw_labels=measurements[:len(plans)]; teacher_labels=measurements[len(plans):]
    for row,label in zip(chosen,teacher_labels,strict=True):
        row.update(label)
    # A physics-relaxed candidate is a separately tagged teacher, not a diffusion trajectory point.
    relaxed=[]; relaxed_sources=[]
    if settings["include_relaxed_teacher"]:
        tokenizer=AutoTokenizer.from_pretrained(assets["draft"],trust_remote_code=True)
        for i,indices in enumerate(selected_by_source):
            pool=[(chosen[j],teacher_labels[j]) for j in indices]
            pool += [(raw_records[i],raw_labels[i])]
            pool=[(r,l) for r,l in pool if l.get("final_structure") and l["measurement"]["verified"]]
            if not pool: continue
            r,l=min(pool,key=lambda x:x[1]["measurement"]["terminal_hull"]
                    if x[1]["measurement"]["terminal_hull"] is not None else float("inf"))
            try:
                q=quantized_record(plans[i],l["final_structure"],tokenizer,"physics_terminal_requantized")
                q["source_id"]=plans[i]["source_id"]+":physics_teacher"
                relaxed.append(q); relaxed_sources.append(i)
            except (ValueError,KeyError,TypeError):
                pass
        del tokenizer; release()
    relaxed_labels=measure(config,_reindex(relaxed),folder/"relaxed_exact_labels",settings,budget,device=device) if relaxed else []
    bundles=[]
    for i,(plan,draft) in enumerate(zip(plans,drafts,strict=True)):
        raw=dict(draft["record"],measurement=raw_labels[i]["measurement"])
        teachers=[chosen[j] for j in selected_by_source[i]]
        for j,index in enumerate(relaxed_sources):
            if index==i:
                teachers.append(dict(relaxed[j],**relaxed_labels[j]))
        bundles.append({"plan":plan,"raw":raw,"teachers":teachers})
    write_rows(folder/"bundles.jsonl",bundles)
    return bundles


def development(config,assets,plans,folder,settings,budget,device,guide=None):
    from .backend import sample_drafts,measure
    drafts=sample_drafts(config,assets,plans,Path(folder)/"drafts",settings,budget,guide=guide,device=device)
    measured=measure(config,_reindex([d["record"] for d in drafts]),Path(folder)/"labels",settings,budget,device=device)
    return {p["source_id"]:Measurement.from_dict(m["measurement"]) for p,m in zip(plans,measured,strict=True)}


def train_verifier(config,assets,plans,folder,settings,budget,device):
    from .backend import sample_drafts,measure
    from .verifier import fit_process_verifier
    folder=Path(folder); folder.mkdir(parents=True,exist_ok=True)
    if (folder/"model.json").exists(): return read_json(folder/"model.json")
    plans=shuffled(plans,seed_for(settings["seed"],"verifier_sources"))[:settings["verifier"]["sources"]]
    raw=sample_drafts(config,assets,plans,folder/"base",settings,budget,collect=True,device=device)
    baseline=measure(config,_reindex([r["record"] for r in raw]),folder/"base_labels",settings,budget,device=device)
    fork_plans=[]; forced={}; selected={}
    for i,(plan,result) in enumerate(zip(plans,raw,strict=True)):
        for probe in result["process_probes"]:
            alternatives=[]
            for action in probe["actions"]:
                if action != probe["selected_action"] and action not in alternatives:
                    alternatives.append(action)
            if alternatives:
                # One exact state per request; continuation is the unchanged plain current constructor.
                selected[i]=probe
                for k,action in enumerate(alternatives[:settings["verifier"]["alternatives_per_source"]]):
                    fork=deepcopy(plan); fork["source_id"]=plan["source_id"]+f":branch{k}"
                    fork["ordinal"]=len(fork_plans); fork["original_ordinal"]=len(fork_plans)
                    forced[fork["source_id"]]={"state_key":probe["state_key"],"action":action,"parent_index":i}
                    fork_plans.append(fork)
                break
    rows=[]
    for i,probe in selected.items():
        target=raw_utility(Measurement.from_dict(baseline[i]["measurement"]),require_hull=settings["require_hull"])
        rows.append({"source_id":plans[i]["source_id"],"state_key":probe["state_key"],
                     "features":probe["features"][probe["selected"]],"target":target,
                     "action":probe["selected_action"],"outcome_source":"unrefined_draft"})
    if fork_plans:
        forks=sample_drafts(config,assets,fork_plans,folder/"branches",settings,budget,collect=True,forced=forced,device=device)
        measured=measure(config,_reindex([f["record"] for f in forks]),folder/"branch_labels",settings,budget,device=device)
        for plan,result,label in zip(fork_plans,forks,measured,strict=True):
            spec=forced[plan["source_id"]]; i=spec["parent_index"]; probe=selected[i]
            j=probe["actions"].index(spec["action"])
            target=raw_utility(Measurement.from_dict(label["measurement"]),require_hull=settings["require_hull"])
            rows.append({"source_id":plans[i]["source_id"],"state_key":probe["state_key"],
                         "features":probe["features"][j],"target":target,"action":spec["action"],
                         "outcome_source":"exact_replay_counterfactual_unrefined_draft"})
    write_rows(folder/"training_rows.jsonl",rows)
    if settings['verifier'].get('fit_mode','absolute')=='paired':
        from .advantage import fit_completion_advantage
        return fit_completion_advantage(rows,folder/'model.json',assets['binding'],settings['verifier'],seed=settings['seed'])
    return fit_process_verifier(rows,folder/"model.json",assets["binding"],settings["verifier"],seed=settings["seed"])


def run(config,output,settings,*,device="cuda:0",budget=None,seed_assets=None,gate_evaluator=None):
    from dlm_iclr.runtime.config import run_root
    from .learning import train_draft,adapt_c1
    from .verifier import ProcessVerifier
    root=Path(output).resolve(); root.mkdir(parents=True,exist_ok=True)
    with exclusive_run(root):
        identity={"schema":SCHEMA,"baseline":seed_assets or initial_assets(config),"settings":settings,
                  "dataset_root":str(run_root(config).resolve())}
        if (root/"identity.json").exists() and read_json(root/"identity.json")!=identity:
            raise ValueError("Cannot resume a loop with different settings/assets")
        write_json(root/"identity.json",identity)
        preflight(config,root,settings)
        budget=budget if budget is not None else Budget(root/"budget.json",settings["budget"])
        ledger=ConditionLedger(root/"conditions.sqlite")
        try:
            dev=prepare_development(config,root,settings,ledger)
            active_path=root/"active_assets.json"
            active=deepcopy(seed_assets) if seed_assets is not None else initial_assets(config)
            active["binding"]=model_identity(active["base"],active["draft"],active["head"])
            initial_path=root/"initial_assets.json"
            if initial_path.exists() and read_json(initial_path)!=active:
                raise ValueError("Baseline model bytes changed since loop creation")
            write_json(initial_path,active)
            if not active_path.exists(): write_json(active_path,active)
            replay=read_rows(run_root(config)/"data/structures/train.jsonl")
            for row in replay:
                row["source_split"]="train"
            # Reserve identities before admitting any synthetic source. No validation outcome enters training.
            write_json(root/'progress.json',{'stage':'baseline_development','requests':len(dev)})
            refresh_reference(config,dev,settings)
            baseline=development(config,active,dev,root/"initial_development",settings,budget,device)
            baseline_folder=root/'initial_development'
            stalled=0; collection_stalls=0; history=[]; pending_teachers=[]; pending_pairs=[]
            for r in range(settings["max_rounds"]):
                folder=root/f"round_{r:03d}"; folder.mkdir(parents=True,exist_ok=True)
                decision_path=folder/"decision.json"
                if decision_path.exists():
                    decision=read_json(decision_path)
                    if (folder/"teachers.jsonl").exists(): history.append(read_rows(folder/"teachers.jsonl"))
                    if decision["status"]=="accepted":
                        active=decision["assets"]
                        baseline={k:Measurement.from_dict(v) for k,v in decision["development"].items()}
                        baseline_folder=folder/'development'
                        stalled=0; collection_stalls=0; pending_teachers=[]; pending_pairs=[]
                    elif decision["status"]=="deferred":
                        collection_stalls+=1
                        pending_teachers.extend(read_rows(folder/"teachers.jsonl"))
                        pending_pairs.extend(read_rows(folder/"pairs.jsonl"))
                    else:
                        stalled+=1
                        pending_teachers=[]; pending_pairs=[]
                    write_json(active_path,active)
                    if decision.get("stop"): break
                    continue
                budget.reserve("rounds",0)
                snapshot=folder/"snapshot.json"
                if snapshot.exists() and read_json(snapshot)!=active:
                    raise ValueError("Round sampler snapshot differs from active checkpoint")
                write_json(snapshot,active)
                write_json(root/'progress.json',{'round':r,'stage':'fresh_plans','requests':settings['plans_per_round']})
                plans=fresh_plans(config,folder/"conditions",r,settings,ledger,budget,device)
                if not plans:
                    write_json(decision_path,{"status":"stopped","stop":True,"reason":"no_new_admissible_conditions"}); break
                refresh_reference(config,plans,settings)
                guide=ProcessVerifier(read_json(active["verifier"]),active["binding"]) if active.get("verifier") else None
                write_json(root/'progress.json',{'round':r,'stage':'teacher_collection','requests':len(plans)})
                collection_settings=deepcopy(settings)
                collection_settings['teacher_steps']=active.get('teacher_steps',settings['teacher_steps'])
                bundles=collect_data(config,active,plans,folder/"collection",collection_settings,budget,device,guide)
                teachers,pairs,report=compile_teachers(bundles,settings)
                if settings['training'].get('lattice_anchor_kl',0):
                    from .learning import construction_patterns
                    from dlm_iclr._core.fixed_slot import MASK_TOKEN_ID
                    indexes={p['source_id']:i for i,p in enumerate(plans)}
                    for teacher in teachers:
                        source=read_json(folder/'collection/drafts'/f"{indexes[teacher['source_id']]:06d}.json")
                        teacher['construction_masks']=construction_patterns(source,MASK_TOKEN_ID)
                        if source['record'].get('body_token_ids'):
                            teacher['source_body_token_ids']=source['record']['body_token_ids']
                report['teacher_diffusion_steps']=collection_settings['teacher_steps']
                unknown = sum(raw_utility(Measurement.from_dict(b["raw"]["measurement"]),
                                          require_hull=settings["require_hull"]) is None for b in bundles)
                report["raw_unknown_fraction"] = unknown / max(1,len(bundles))
                write_rows(folder/"teachers.jsonl",teachers); write_rows(folder/"pairs.jsonl",pairs)
                write_json(folder/"teacher_report.json",report)
                pooled_teachers = pending_teachers + teachers
                pooled_pairs = pending_pairs + pairs
                defer_reason = None
                if report["raw_unknown_fraction"]>settings["quality"]["max_unknown_fraction"]:
                    defer_reason = "too_many_unknown_teacher_sources"
                elif len(pooled_teachers)<settings["min_teachers"]:
                    defer_reason = "insufficient_verified_exact_token_teachers"
                if defer_reason:
                    collection_stalls+=1
                    stop=should_stop(r,collection_stalls,settings,collection=True)
                    write_json(decision_path,{"status":"deferred","stop":stop,"reason":defer_reason,
                                              "report":report,"pooled_teachers":len(pooled_teachers),
                                              "retained_assets":active})
                    pending_teachers=pooled_teachers; pending_pairs=pooled_pairs
                    if stop: break
                    continue
                collection_stalls=0
                teachers,pairs=pooled_teachers,pooled_pairs
                write_json(folder/"training_sources.json",{"teachers":[t["source_id"] for t in teachers],
                                                          "pairs":[p["source_id"] for p in pairs]})
                pending_teachers=[]; pending_pairs=[]
                training_settings=deepcopy(settings)
                if len(pairs)<settings["min_pairs"]:
                    training_settings["training"]["preference_weight"]=0.0  # keep verified SFT; never fabricate pairs
                completed=folder/"candidate_assets.json"
                if completed.exists():
                    candidate=read_json(completed)
                else:
                    write_json(root/'progress.json',{'round':r,'stage':'student_training','teachers':len(teachers),'pairs':len(pairs)})
                    draft=train_draft(active["base"],active["draft"],teachers,pairs,replay,
                                      balanced_history(history),folder/"student",training_settings,device=device)
                    # Original replay and verified targets both stabilize the adapted relation head.
                    c1_rows=teachers+shuffled(replay,seed_for(settings["seed"],r,"c1_replay"))[:len(teachers)]
                    head=adapt_c1(active["base"],draft,active["head"],c1_rows,folder/"c1/head.pt",settings,device=device)
                    candidate={"base":active["base"],"draft":draft,"head":head,"verifier":None}
                    candidate["binding"]=model_identity(candidate["base"],draft,head)
                    write_json(completed,candidate)
                write_json(root/'progress.json',{'round':r,'stage':'development_gate','requests':len(dev)})
                after=development(config,candidate,dev,folder/"development",settings,budget,device)
                gate=publication_gate(baseline,after,settings["gate"],require_hull=settings["require_hull"])
                write_json(folder/"draft_gate.json",gate)
                if gate_evaluator is not None:
                    write_json(root/'progress.json',{'round':r,'stage':'refinement_publication_evaluation'})
                    gate=gate_evaluator(config,active,candidate,dev,baseline_folder,folder,settings,budget,gate,device=device)
                    write_json(folder/'publication_gate.json',gate)
                if gate["accepted"]:
                    if settings['verifier'].get('enabled',True):
                        write_json(root/'progress.json',{'round':r,'stage':'verifier','sources':settings['verifier']['sources']})
                        verifier=train_verifier(config,candidate,plans,folder/"verifier",settings,budget,device)
                        if verifier["validation"]["deployable"]:
                            guide=ProcessVerifier(verifier,candidate["binding"])
                            guided=development(config,candidate,dev,folder/"guided_development",settings,budget,device,guide)
                            guide_gate=publication_gate(after,guided,settings["gate"],require_hull=settings["require_hull"])
                            write_json(folder/"guide_gate.json",guide_gate)
                            if guide_gate["accepted"]: candidate["verifier"]=str(folder/"verifier/model.json")
                    else:
                        write_json(folder/'verifier_policy.json',{'online_prefix_guidance':False,
                            'verified_teacher_and_preference_signal_compilation':True,
                            'reason':'learning_signal_role_does_not_require_online_intervention'})
                    # This uses only already-generated development drafts. It
                    # never conditions teacher admission on a test result.
                    if settings.get('probe_shorter_during_training',True):
                        from .step_budget import probe_shorter_refinement
                        write_json(root/'progress.json',{'round':r,'stage':'development_step_budget'})
                        step_decision=probe_shorter_refinement(config,candidate,dev,folder,settings,budget,
                            current_steps=collection_settings['teacher_steps'],round_id=r,device=device)
                    else:
                        step_decision={'selected_steps':collection_settings['teacher_steps'],'tested':False,
                            'reason':'retain_teacher_budget_until_separate_step_validation'}
                    candidate['teacher_steps']=step_decision['selected_steps']
                    if gate_evaluator is not None:candidate['effect_scope']=gate['effect_scope']
                    active=candidate; baseline=after; stalled=0
                    baseline_folder=folder/'development'
                    decision={"status":"accepted","stop":False,"assets":active,"gate":gate,
                              "development":{k:v.to_dict() for k,v in after.items()}}
                    # Persist the decision before changing the global pointer, so recovery is transactional.
                    write_json(decision_path,decision); write_json(active_path,active)
                else:
                    stalled+=1
                    write_json(decision_path,{"status":"rejected","stop":should_stop(r,stalled,settings),
                                               "gate":gate,"retained_assets":active})
                history.append(teachers)
                if should_stop(r,stalled,settings): break
            write_json(root/"COMPLETE.json",{"status":"completed_or_policy_stopped","active_assets":active,
                         "note":"Only accepted generator checkpoints are used for subsequent rounds.",
                         "physics_is_proxy":True,"budget":budget.state})
            write_json(root/'progress.json',{'stage':'loop_complete','accepted_assets':active})
            return active
        except Exception as error:
            write_json(root/"STOPPED.json",{"status":"stopped","exception":type(error).__name__,"reason":str(error)})
            raise
        finally:
            ledger.close()
