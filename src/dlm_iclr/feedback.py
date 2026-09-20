"""Train and evaluate the registered, physically verified feedback method.

Supervision preparation uses draft_loop.teacher_registration, prefix_teacher,
state_replay, and backend.measure. This entry consumes their reviewed artifacts;
it never constructs labels from unknown outcomes or queries physics in sampling.
"""
import argparse
from copy import deepcopy
import math
from pathlib import Path

from .draft_loop.common import Budget,digest,exclusive_run,model_identity,read_json,read_rows,write_json


def validate_material(teachers,feedback,review):
    from .draft_loop.teacher_fit import feedback_view
    from .draft_loop.quality import Measurement,raw_utility
    if (not review.get('eligible') or review.get('teachers_digest')!=digest(teachers)
            or review.get('feedback_digest')!=digest(feedback)):
        raise ValueError('Material review does not bind these exact supervision artifacts')
    sources={t['source_id']:t for t in teachers}
    if not teachers or len(sources)!=len(teachers) or not feedback:
        raise ValueError('Unique nonempty teacher sources and verified prefix feedback required')
    for teacher in teachers:
        if teacher.get('source_split')!='train':raise ValueError('Training-only material required')
        if not teacher.get('body_prompt') or not teacher.get('registration',{}).get('physically_reverified'):
            raise ValueError('Original prompt and independently remeasured registered teacher required')
        if teacher['measurement']['record_key']!=teacher['exact_record_key']:
            raise ValueError('Teacher measurement binding differs')
        m=Measurement.from_dict(teacher['measurement'])
        if m.status not in ['singlepoint','verified'] or not m.has_raw(False):
            raise ValueError('Unknown teacher physics cannot certify a geometric target')
    for row in feedback:
        view=feedback_view(row)
        if view['source_id'] not in sources or view['prompt']!=sources[view['source_id']]['body_prompt']:
            raise ValueError('Prefix supervision must keep the teacher Plan/prompt')
        candidate=Measurement.from_dict(row['measurement']);parent=Measurement.from_dict(row['parent_measurement'])
        a,b=raw_utility(parent),raw_utility(candidate)
        if (candidate.status not in ['singlepoint','verified'] or a is None or b is None
                or candidate.composition!=parent.composition or candidate.protocol_key!=parent.protocol_key):
            raise ValueError('Prefix feedback physical comparison is unknown or changes reference')
        if (row['measurement']['record_key']!=row['proposal_record_key'] or not row.get('raw_utility_gain',0)>0
                or not math.isfinite(row['raw_utility_gain']) or abs(row['raw_utility_gain']-(b-a))>1e-8
                or not row.get('weight',0)>0 or not math.isfinite(row['weight'])):
            raise ValueError('Prefix feedback must retain its positive, known physical measurement')
    return {'teachers':len(teachers),'prefix_views':len(feedback),'sources':len(sources),
            'teachers_digest':digest(teachers),'feedback_digest':digest(feedback)}


def locked_write(path,value):
    if path.exists() and read_json(path)!=value:raise ValueError('Frozen run inputs or configuration changed')
    write_json(path,value)


def settings_from_profile(profile,limits):
    from .draft_loop.settings import DEFAULTS
    settings=deepcopy(DEFAULTS)
    settings['runtime'].update(profile['runtime'])
    settings['evaluation']['direct_workers']=profile['direct_workers']
    settings['teacher_steps']=profile['F_steps']
    settings['budget']=limits
    return settings


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['train','evaluate'])
    for key in ['config','assets','recipe','budget-ledger','budget-limits','output']:
        parser.add_argument('--'+key,required=True,type=Path)
    parser.add_argument('--teachers',type=Path);parser.add_argument('--prefix-feedback',type=Path)
    parser.add_argument('--material-review',type=Path);parser.add_argument('--plans',type=Path)
    parser.add_argument('--raw-only',action='store_true');parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--reference-split',choices=['val','test'],default='val')
    parser.add_argument('--panel-scope',choices=['seen_train','development','final_test'],default='seen_train')
    args=parser.parse_args(argv)
    from .runtime.config import load
    from .draft_loop.inference_protocol import bind_inference_protocol
    config=load(args.config);assets=read_json(args.assets);profile=read_json(args.recipe)
    limits=read_json(args.budget_limits);limits=limits.get('limits',limits)
    if profile.get('schema')!='registered_feedback_recipe_v1':raise ValueError('Unsupported feedback recipe')
    if model_identity(assets['base'],assets['draft'],assets['head'])!=assets['binding']:
        raise ValueError('Model bytes do not match the asset manifest')
    out=args.output.resolve()
    with exclusive_run(out):
        budget=Budget(args.budget_ledger.resolve(),limits)
        if args.stage=='train':
            if not all([args.teachers,args.prefix_feedback,args.material_review]):
                parser.error('train needs --teachers, --prefix-feedback and --material-review')
            teachers=read_rows(args.teachers);feedback=read_rows(args.prefix_feedback)
            material=validate_material(teachers,feedback,read_json(args.material_review))
            # Binding a digest is not sufficient if a token body was paired with
            # an unrelated physical record key during external material assembly.
            from transformers import AutoTokenizer
            from .evaluation.physics import record_key
            tokenizer=AutoTokenizer.from_pretrained(assets['draft'],trust_remote_code=True)
            inverse={int(v):k for k,v in tokenizer.get_vocab().items()}
            for teacher in teachers:
                record={'body':''.join(inverse[x] for x in teacher['body_token_ids'])}
                if record_key(record)!=teacher['exact_record_key']:raise ValueError('Teacher token/physics identity differs')
            del tokenizer
            frozen={'assets':assets,'material':material,'recipe':profile,'ledger':str(budget.path)}
            locked_write(out/'identity.json',frozen)
            if (out/'assets.json').exists():return
            from .draft_loop.teacher_fit import train_existing_teachers
            report=train_existing_teachers(assets['base'],assets['draft'],teachers,out/'student',
                profile['training'],seed=profile['seed'],device=args.device,budget=budget,feedback=feedback)
            candidate={**assets,'draft':report['selected_checkpoint'],'verifier':None,
                       'inference_protocol':deepcopy(profile['inference_protocol'])}
            candidate['generation_policy_identity']=digest(candidate['inference_protocol'])
            candidate['binding']=model_identity(candidate['base'],candidate['draft'],candidate['head'])
            write_json(out/'assets.json',candidate)
            write_json(out/'SUMMARY.json',{'status':'trained_not_physically_validated','material':material,
                'selected_step':report['best_step'],'scientific_gain_established':False,
                'no_online_teacher_or_physical_oracle':True})
        else:
            if not args.plans:parser.error('evaluate needs the already fixed --plans JSONL')
            from .data.plans import load_plans
            from .draft_loop.backend import sample_drafts
            from .draft_loop.evaluation import evaluate_endpoint,refine_drafts
            # Both comparison arms use this explicit frozen protocol. A baseline
            # without the new manifest fields must not silently remain at T0.7.
            assets={**assets,'inference_protocol':deepcopy(profile['inference_protocol'])}
            assets['generation_policy_identity']=digest(assets['inference_protocol'])
            config=bind_inference_protocol(config,assets)
            config['evaluation']['reference_test']=config['dataset']['splits'][args.reference_split]
            plans,_=load_plans(args.plans,legal_only=False)
            frozen={'assets':assets,'recipe':profile,'plans':plans,'config':config,'raw_only':args.raw_only,
                    'ledger':str(budget.path),'panel_scope':args.panel_scope,'reference_split':args.reference_split,
                    'no_test_feedback_into_training':True}
            locked_write(out/'identity.json',frozen)
            settings=settings_from_profile(profile,limits)
            drafts=sample_drafts(config,assets,plans,out/'drafts',settings,budget,device=args.device)
            reports={'raw':evaluate_endpoint(config,[x['record'] for x in drafts],out/'raw',settings,budget,device=args.device)}
            if not args.raw_only:
                refined=refine_drafts(config,assets,plans,drafts,out/f"F{profile['F_steps']}/generated",settings,budget,device=args.device)
                reports[f"F{profile['F_steps']}"]=evaluate_endpoint(config,[x['record'] for x in refined],
                    out/f"F{profile['F_steps']}/evaluation",settings,budget,device=args.device)
            write_json(out/'SUMMARY.json',{'status':'complete','requests':len(plans),'reports':reports,
                'panel_scope':args.panel_scope,'reference_split':args.reference_split,
                'root_review_required':True,'no_automatic_publication_or_retuning':True})


if __name__=='__main__':main()
