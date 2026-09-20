"""Frozen-generator comparison on fresh Plans after training/publication ends."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import numpy as np
from .common import read_json,read_rows,write_rows,write_json,digest
from .backend import sample_drafts,measure,release,_numeric_graph,_parallel_sources
from .quality import Measurement,raw_utility


def refine_drafts(config,assets,plans,drafts,folder,settings,budget,*,device='cuda:0'):
    from .inference_protocol import bind_inference_protocol
    config=bind_inference_protocol(config,assets)
    from transformers import AutoTokenizer
    from dlm_iclr.runtime.config import asset
    from dlm_iclr.diffusion.refinement import Refiner
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    cached=[folder/f'{i:06d}.json' for i in range(len(plans))]
    protocol=config['diffusion'].get('reduction_protocol','legacy_scatter')
    if protocol!='legacy_scatter':
        manifest=folder/'execution_identity.json'
        identity={'reduction_protocol':protocol,'steps':settings['teacher_steps'],
            'assets':assets,'ordered_inputs':digest([[p['source_id'],p['refiner_noise_seed'],d['record']]
                                                   for p,d in zip(plans,drafts,strict=True)])}
        if manifest.exists():
            if read_json(manifest)!=identity:raise ValueError('Refiner execution protocol or inputs changed')
        elif any(p.exists() for p in cached):
            raise ValueError('Cannot label unversioned refiner cache as ordered execution')
        else:write_json(manifest,identity)
    if all(p.exists() for p in cached): return [read_json(p) for p in cached]
    if len(settings['runtime']['devices'])>1:
        # Worker resolves this function from its declared module.
        return _parallel_sources('refine_drafts',config,assets,plans,folder,settings,budget,drafts=drafts)
    tokenizer=AutoTokenizer.from_pretrained(assets['draft'],trust_remote_code=True)
    refiner=Refiner(asset(config,'diffusion'),tokenizer,device,steps=settings['teacher_steps'],
                    reuse_fixed_geometry=config['diffusion']['reuse_fixed_geometry'],
                    reduction_protocol=config['diffusion'].get('reduction_protocol','legacy_scatter'))
    pending=[i for i,p in enumerate(cached) if not p.exists()]
    batch_size=settings['runtime']['teacher_batch_size']
    for start in range(0,len(pending),batch_size):
        ids=pending[start:start+batch_size]
        generated=[dict(drafts[i],graph=_numeric_graph(drafts[i]['graph']) if drafts[i].get('graph') is not None else None) for i in ids]
        budget.reserve('teacher_calls',sum(d.get('graph') is not None for d in generated))
        refined=refiner.refine_many([plans[i] for i in ids],generated)
        for i,row in zip(ids,refined,strict=True): write_json(cached[i],row)
        print({'stage':'frozen_evaluation_refine','steps':settings['teacher_steps'],'completed':start+len(ids),'requests':len(pending)},flush=True)
    del refiner,tokenizer;release()
    return [read_json(p) for p in cached]


def evaluate_endpoint(config,records,folder,settings,budget,*,device='cuda:0',
                      cached_measurements=None,cached_labels=None):
    from dlm_iclr.evaluation.workflow import evaluate
    from dlm_iclr.evaluation.direct import evaluate_direct
    from dlm_iclr.runtime.config import run_root,reference_path
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    if (folder/'REPORT.json').exists(): return read_json(folder/'REPORT.json')
    write_rows(folder/'records.jsonl',records)
    def direct():
        return evaluate_direct(records,folder/'direct',metrics='full',
            reference=reference_path(config,'test'),dataset=config['dataset']['name'],
            composition=config['evaluation']['composition'],workers=settings['evaluation']['direct_workers'],
            cache=run_root(config)/'cache/direct')[1]
    if (cached_measurements is None)!=(cached_labels is None):
        raise ValueError('Measurements and physical labels must be reused together')
    if cached_measurements is not None:
        from dlm_iclr.evaluation.physics import record_key
        expected=[record_key(r) for r in records]
        if ([r['measurement']['record_key'] for r in cached_measurements]!=expected
                or [r['record_key'] for r in cached_labels]!=expected):
            raise ValueError('Cumulative physical labels differ from the ordered structures')
    with ThreadPoolExecutor(1) as executor:
        task=executor.submit(direct)
        measured=(cached_measurements if cached_measurements is not None
                  else measure(config,records,folder/'measurements',settings,budget,device=device))
        labels=(cached_labels if cached_labels is not None
                else read_rows(folder/'measurements/physical/labels.jsonl'))
        scores,sun=evaluate(config,records,folder/'sun',labels=labels)
        direct_report=task.result()
    raw=[Measurement.from_dict(row['measurement']) for row in measured]
    report={'requests':len(records),'SUN_MSUN':sun,'Direct':direct_report,'raw':{}}
    for field in ('raw_energy','force_rms','force_max','stress_max','raw_hull','actual_steps'):
        vals=[getattr(m,field) for m in raw if getattr(m,field) is not None]
        report['raw'][field]={'known':len(vals),'mean':float(np.mean(vals)) if vals else None,
                              'median':float(np.median(vals)) if vals else None}
    utilities=[raw_utility(m) for m in raw]
    valid=[u for u in utilities if u is not None]
    report['raw']['utility']={'known':len(valid),'mean':float(np.mean(valid)) if valid else None}
    # This extra table evaluates the measured *unrelaxed* geometry. It is kept
    # separate from the historical SUN convention based on relaxed endpoints.
    from dlm_iclr._core.exact_sun_nu import conjunction
    instantaneous=[]
    for m,s in zip(raw,scores,strict=True):
        if m.status in ('generation_failure','invalid_raw'):
            stable=meta=False
        elif m.has_raw(True):
            mechanical=m.force_max<=config['evaluation']['fmax'] and m.stress_max<=config['evaluation']['stress_tolerance_GPa']
            stable=mechanical and m.raw_hull<=config['evaluation']['stable_threshold']
            meta=mechanical and m.raw_hull<=config['evaluation']['metastable_threshold']
        else: stable=meta=None
        instantaneous.append({'source_id':s['source_id'],'Stable':stable,'MetaStable':meta,
            'SUN':conjunction(stable,s.get('novel'),s.get('unique_representative')),
            'MSUN':conjunction(meta,s.get('novel'),s.get('unique_representative'))})
    report['instantaneous_raw_proxy']={field:{'passed':sum(r[field] is True for r in instantaneous),
            'unknown':sum(r[field] is None for r in instantaneous),'requests':len(records)}
            for field in ('Stable','MetaStable','SUN','MSUN')}
    write_rows(folder/'instantaneous_raw_proxy.jsonl',instantaneous)
    write_json(folder/'REPORT.json',report)
    return report


def run_evaluation(config,root,settings,budget,*,device='cuda:0'):
    """No test outcome feeds publication, training, or selection of the model."""
    from dlm_iclr.runtime.config import backend_config
    from dlm_iclr.planner.sampling import generate_plans
    from dlm_iclr.data.plans import load_plans
    from .workflow import refresh_reference
    from .verifier import ProcessVerifier
    root=Path(root);out=root/'final_evaluation';out.mkdir(exist_ok=True)
    initial=read_json(root/'initial_assets.json');active=read_json(root/'active_assets.json')
    if initial['binding']==active['binding']:
        result={'status':'no_student_published','scientific_gain_established':False,
                'reason':'All candidates failed development publication; original model retained.'}
        write_json(out/'SUMMARY.json',result);return result
    selected=active.get('teacher_steps',settings['teacher_steps'])
    full=max(settings['evaluation']['refine_steps'])
    levels=[selected,full] if selected<full else settings['evaluation']['refine_steps']
    locked={'baseline':initial,'student':active,'evaluation':settings['evaluation'],'actual_refine_steps':levels}
    if (out/'frozen_assets.json').exists() and read_json(out/'frozen_assets.json')!=locked:
        raise ValueError('Final evaluation models/settings changed')
    write_json(out/'frozen_assets.json',locked)
    plan_path=out/'planner.jsonl'
    if not plan_path.exists():
        n=settings['evaluation']['requests'];budget.reserve('planner_draws',n)
        generate_plans(backend_config(config).assets,str(plan_path),requests=n,
            seed=settings['evaluation']['seed'],device=device,usage_role='final_evaluation',
            batch_size=config['planner']['sampling']['batch_size'],temperature=settings['planner_temperature'],
            top_p=config['planner']['sampling']['top_p'],top_k=config['planner']['sampling']['top_k'],
            max_new_tokens=config['planner']['sampling']['max_new_tokens'])
        release()
    plans,_=load_plans(str(plan_path),legal_only=False,seed=settings['evaluation']['seed'])
    refresh_reference(config,plans,settings)
    variants=[('baseline',initial,None),('student',active,None)]
    if active.get('verifier'):
        variants.append(('student_guided',active,ProcessVerifier(read_json(active['verifier']),active['binding'])))
    reports={}
    for name,assets,guide in variants:
        part=out/name
        drafts=sample_drafts(config,assets,plans,part/'drafts',settings,budget,guide=guide,device=device)
        reports[name]={}
        reports[name]['raw']=evaluate_endpoint(config,[d['record'] for d in drafts],part/'raw',settings,budget,device=device)
        for steps in levels:
            recipe=deepcopy(settings);recipe['teacher_steps']=steps
            refined=refine_drafts(config,assets,plans,drafts,part/f'F{steps}/generated',recipe,budget,device=device)
            reports[name][f'F{steps}']=evaluate_endpoint(config,[f['record'] for f in refined],part/f'F{steps}',settings,budget,device=device)
        write_json(out/'SUMMARY.json',{'status':'in_progress','reports':reports})
    write_json(out/'SUMMARY.json',{'status':'complete','reports':reports,'requests':len(plans),
        'no_test_outcome_used_for_training':True,'new_plans_same_for_all_models':True})
    lines=['# MP20 Verified Draft Loop: fixed final panel','',
           '| Model | Output | SUN | MSUN | V | Coverage P | Coverage R |','|---|---|---:|---:|---:|---:|---:|']
    for name,parts in reports.items():
        for endpoint,r in parts.items():
            m=r['SUN_MSUN']['metrics'];d=r['Direct']['metrics'];n=r['requests']
            sun=f"{m['SUN']['passed']}/{n} ({m['SUN']['pending']} unknown)"
            ms=f"{m['MSUN']['passed']}/{n} ({m['MSUN']['pending']} unknown)"
            lines.append(f"| {name} | {endpoint} | {sun} | {ms} | {m['V']['passed']}/{n} | {d['cov_precision']} | {d['cov_recall']} |")
    lines+=['','Raw force/stress/energy and relaxation steps are reported separately in SUMMARY.json.',
            'SUN/MSUN use the retained CHGNet/MP-hull endpoint protocol; they do not by themselves prove raw mechanical stability.']
    (out/'RESULT_ZH.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return reports
