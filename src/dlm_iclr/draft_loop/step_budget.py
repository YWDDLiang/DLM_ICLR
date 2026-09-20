"""Reduce F time only on a paired development panel with preserved quality."""
from copy import deepcopy
from pathlib import Path
from .common import read_json,read_rows,write_json
from .quality import Measurement,publication_gate


def paired_step_gate(long_scores,short_scores,quality_gate,*,minimum_common_fraction=.8):
    if [r['source_id'] for r in long_scores] != [r['source_id'] for r in short_scores] or not long_scores:
        raise ValueError('Step comparisons require the same ordered development sources')
    reasons=list(quality_gate['reasons']);metrics={}
    for field in ('strict_sun','meta_sun','strict_stable','meta_stable'):
        known=[(a[field],b[field]) for a,b in zip(long_scores,short_scores) if a[field] is not None and b[field] is not None]
        ub=sum(r[field] is None for r in long_scores);ua=sum(r[field] is None for r in short_scores)
        gains=sum(a is False and b is True for a,b in known)
        losses=sum(a is True and b is False for a,b in known)
        if len(known)/len(long_scores)<minimum_common_fraction: reasons.append(field+'_insufficient_pairs')
        if losses>gains: reasons.append(field+'_regression')
        if ua>ub: reasons.append(field+'_more_unknown')
        metrics[field]={'common_known':len(known),'gains':gains,'losses':losses,
                        'unknown_long':ub,'unknown_short':ua,'requests':len(long_scores)}
    if not any(r['meta_stable'] is True for r in long_scores):
        reasons.append('reference_has_no_confirmed_meta_stable_output')
    return {'accepted':not reasons,'reasons':reasons,'paired_metrics':metrics,
            'quality_gate':quality_gate,'evidence':'fixed_development_panel_not_a_population_guarantee'}


def probe_shorter_refinement(config,assets,plans,round_folder,settings,budget,*,current_steps,round_id,device='cuda:0'):
    from .evaluation import refine_drafts
    from .backend import measure
    from dlm_iclr.evaluation.workflow import evaluate
    folder=Path(round_folder)/'step_budget';folder.mkdir(parents=True,exist_ok=True)
    target=folder/'decision.json'
    if target.exists(): return read_json(target)
    minimum=min(settings['evaluation']['refine_steps'])
    shorter=max(minimum,current_steps//2)
    # Keep enough of the existing total budget for remaining training and the
    # full final comparison, even if a guided model is also published.
    budget.state=read_json(budget.path)
    used=budget.state['counts'].get('teacher_calls',0)
    remaining=(settings['max_rounds']-round_id-1)*settings['plans_per_round']
    final=3*len(settings['evaluation']['refine_steps'])*settings['evaluation']['requests']
    cap=settings['budget'].get('max_teacher_calls',float('inf'))
    if shorter>=current_steps or used+2*len(plans)+remaining+final>cap:
        result={'selected_steps':current_steps,'tested':False,'reason':'minimum_steps_or_reserved_final_budget'}
        write_json(target,result);return result
    development='guided_development' if assets.get('verifier') else 'development'
    draft_folder=Path(round_folder)/development/'drafts'
    drafts=[read_json(draft_folder/f'{i:06d}.json') for i in range(len(plans))]
    measurements={};scores={}
    for steps in (current_steps,shorter):
        recipe=deepcopy(settings);recipe['teacher_steps']=steps
        part=folder/f'F{steps}'
        generated=refine_drafts(config,assets,plans,drafts,part/'generated',recipe,budget,device=device)
        records=[r['record'] for r in generated]
        measured=measure(config,records,part/'measurements',settings,budget,device=device)
        labels=read_rows(part/'measurements/physical/labels.jsonl')
        scores[steps],summary=evaluate(config,records,part/'scoring',labels=labels)
        write_json(part/'SUMMARY.json',summary)
        measurements[steps]={r['source_id']:Measurement.from_dict(m['measurement']) for r,m in zip(records,measured,strict=True)}
    gate_settings=deepcopy(settings['gate']);gate_settings['min_mean_gain']=0.0
    quality=publication_gate(measurements[current_steps],measurements[shorter],gate_settings,
                             require_hull=settings['require_hull'])
    quality['comparison']='same_student_same_draft_shorter_F'
    gate=paired_step_gate(scores[current_steps],scores[shorter],quality,
                          minimum_common_fraction=settings['gate']['min_common_fraction'])
    result={'selected_steps':shorter if gate['accepted'] else current_steps,
            'tested':True,'reference_steps':current_steps,'candidate_steps':shorter,'gate':gate,
            'decoder_calls_per_structure':2*(shorter if gate['accepted'] else current_steps),
            'raw_oracle_and_SUN_thresholds_unchanged':True}
    write_json(target,result);return result
