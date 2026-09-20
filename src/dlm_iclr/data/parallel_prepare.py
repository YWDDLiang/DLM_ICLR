"""Ordered, streaming dataset preparation with reusable crystal and token views."""
import json
import multiprocessing as mp
from contextlib import ExitStack
from functools import lru_cache
from .adapters import iter_structures, structure_from_row, align_sites
from .plans import composition_key
from ..runtime.config import asset, path, run_root
from ..runtime.io import fingerprint, write_json, json_default


def initialize(config):
    global _config, _tokenizer, _planner
    from transformers import AutoTokenizer
    from .._core.dynamic_crystal import build_special_tokens
    _config = config
    _tokenizer = AutoTokenizer.from_pretrained(asset(config, 'dlm'), trust_remote_code=True)
    _tokenizer.add_special_tokens({'additional_special_tokens': build_special_tokens()})
    _tokenizer.pad_token = _tokenizer.pad_token or _tokenizer.eos_token
    _planner = AutoTokenizer.from_pretrained(asset(config, 'planner_base'), trust_remote_code=True)


@lru_cache(None)
def space_group_number(symbol):
    from pymatgen.symmetry.groups import SpaceGroup
    return SpaceGroup(symbol).int_number


def encode(item):
    from .._core.dynamic_crystal import structure_to_dynamic_answer, parse_dynamic_answer, arrays_to_dynamic_answer
    from .._core.fixed_slot import metadata_from_csv_row
    from .._core.r5_plan_state import plan_state_from_arrays, build_body_prompt
    from ..planner.prepare import build_records_for_plan
    split, index, identifier, row = item
    dataset = _config['dataset']
    sid = f"{dataset['name']}:{split}:{identifier}:{index}"
    result = {'source_id': sid}
    try:
        crystal = structure_from_row(row)
        result['crystals'] = {'source_id': sid, 'dataset': dataset['name'], 'split': split,
                              'structure': crystal.as_dict()}
        if len(crystal) > dataset['max_atoms']:
            raise ValueError('Structure exceeds configured atom count')
        answer, diagnostic = structure_to_dynamic_answer(crystal)
        if diagnostic.length_clips or diagnostic.angle_clips or diagnostic.coord_clips:
            raise ValueError('Structure exceeds the coordinate vocabulary')
        arrays = parse_dynamic_answer(answer, strict=True)
        if row.get('space_group'):
            row['spacegroup.number'] = space_group_number(row['space_group'])
        if dataset.get('infer_spacegroup_if_missing') and not row.get('spacegroup.number'):
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            row['spacegroup.number'] = SpacegroupAnalyzer(crystal, symprec=0.01).get_space_group_number()
        metadata = metadata_from_csv_row(row)
        plan = plan_state_from_arrays(arrays, metadata=metadata)
        planner = build_records_for_plan(split=split, row_idx=index, plan_state=plan, metadata=metadata,
                 tokenizer=_planner, prompt_style='h1_rich_plan_v1', include_sample_id=False,
                 sample_types=['direct_plan'], weights={'direct_plan': 1.0})[0]
        arrays, order = align_sites(arrays, plan)
        answer, _ = arrays_to_dynamic_answer(arrays['lengths'], arrays['angles'], arrays['species'], arrays['frac_coords'])
        prompt = build_body_prompt(plan).rstrip()+'\n'
        ids = _tokenizer(answer, add_special_tokens=False)['input_ids']
        prefix = _tokenizer(prompt, add_special_tokens=False)['input_ids']
        provenance = {'dataset_origin': dataset['name'], 'original_split': split, 'material_id': identifier,
                      'source_row_index': index, 'site_permutation': order,
                      'usage_role': 'train' if split == 'train' else 'evaluation'}
        if dataset.get('infer_spacegroup_if_missing'):
            provenance['spacegroup_construction'] = 'source_metadata_else_spglib_symprec_0.01'
        result['structures'] = dict(result['crystals'], body_prompt=prompt, body_token_ids=ids,
                                    plan_state=plan, provenance=provenance)
        result['b0'] = {'source_id': sid, 'source_split': split, 'prompt': build_body_prompt(plan),
                        'answer': answer, 'num_atoms': len(crystal), 'sample_weight': 1.0}
        result['planner'] = planner
        planned = {'schema': 'crystal_plan_v1', 'source_id': sid, 'original_ordinal': index,
                   'body_eligible': True, 'body_prompt': prompt, 'plan_state': plan, 'provenance': provenance}
        for field, stage in [('body_noise_seed', 'G'), ('refiner_noise_seed', 'F')]:
            planned[field] = int(fingerprint([_config['b0']['seed'], sid, stage])[:15],16)
        result['plans'] = planned
        result['lengths'] = (len(prefix),len(ids))
    except (ValueError, KeyError, TypeError) as error:
        # Full crystal reference remains available even if token representation fails.
        result = {k:v for k,v in result.items() if k in ('source_id','crystals')}
        result['failures'] = {'source_id': sid, 'reason': str(error)}
    return result


def prepare(config):
    root = run_root(config)/'data'
    root.mkdir(parents=True,exist_ok=True)
    initialize(config)
    _tokenizer.save_pretrained(root/'tokenizer')
    folders = ['crystals','structures','b0','planner','plans','failures']
    stats = {'dataset': config['dataset']['name'], 'splits': {}}
    max_prompt=max_answer=0
    workers=config['dataset'].get('prepare_workers',1)
    with ExitStack() as stack:
        pool = stack.enter_context(mp.get_context('spawn').Pool(workers, initialize, (config,))) if workers>1 else None
        for split, source in config['dataset']['splits'].items():
            processed=prepared=failed=references=0
            compositions=set()
            with ExitStack() as files:
                streams={}
                for folder in folders:
                    d=root/folder; d.mkdir(exist_ok=True)
                    streams[folder]=files.enter_context((d/f'{split}.jsonl.tmp').open('w',encoding='utf-8'))
                rows=((split,i,sid,row) for i,sid,row in iter_structures(path(config,source),config['dataset']['fields']))
                for result in (pool.imap(encode,rows,chunksize=64) if pool else map(encode,rows)):
                    processed+=1
                    if 'plans' in result:
                        result['plans']['ordinal']=prepared
                        prepared+=1
                        compositions.add(composition_key(result['plans']['plan_state']))
                        a,b=result['lengths'];max_prompt=max(max_prompt,a);max_answer=max(max_answer,b)
                    failed+=int('failures' in result);references+=int('crystals' in result)
                    for folder in folders:
                        if folder in result:
                            streams[folder].write(json.dumps(result[folder],ensure_ascii=False,allow_nan=False,default=json_default)+'\n')
                    if processed%1000==0:
                        progress={'split':split,'processed':processed,'prepared':prepared,'failed':failed}
                        write_json(root/'prepare_progress.json',progress)
                        print(progress,flush=True)
            for folder in folders:
                (root/folder/f'{split}.jsonl.tmp').replace(root/folder/f'{split}.jsonl')
            stats['splits'][split]={'processed':processed,'prepared':prepared,'failed':failed,
                                    'reference_structures':references,'compositions':len(compositions)}
    stats['observed_max_length']=max_prompt+max_answer
    stats['max_length']=max(256,max_prompt+max_answer+48)
    write_json(root/'statistics.json',stats)
    return stats
