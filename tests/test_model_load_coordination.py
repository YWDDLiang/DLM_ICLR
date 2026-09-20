from concurrent.futures import ThreadPoolExecutor
import threading
import time

import torch

from dlm_iclr.runtime import models


def test_gpu_loading_is_serialized_but_preserves_arguments_and_outputs(monkeypatch, tmp_path):
    monkeypatch.setenv('DLM_MODEL_LOAD_LOCK_DIR', str(tmp_path))
    monkeypatch.setenv('DLM_MODEL_LOAD_SETTLE_SECONDS', '0')
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '1,2')
    counter = {'active': 0, 'peak': 0}; guard = threading.Lock(); calls = []
    def fake(base, adapter, device, *, mean_resizing):
        with guard:
            counter['active'] += 1; counter['peak'] = max(counter['peak'], counter['active'])
            calls.append((base, adapter, str(device), mean_resizing))
        time.sleep(.03)
        with guard: counter['active'] -= 1
        return base, adapter
    monkeypatch.setattr(models, '_load_model_and_tokenizer', fake)
    with ThreadPoolExecutor(3) as pool:
        results = list(pool.map(lambda _: models.load_model_and_tokenizer('base', 'adapter', torch.device('cuda:0'), mean_resizing=False), range(3)))
    assert counter['peak'] == 1
    assert results == [('base', 'adapter')]*3
    assert calls == [('base', 'adapter', 'cuda:0', False)]*3


def test_cpu_and_default_behavior_do_not_create_coordination_files(monkeypatch, tmp_path):
    monkeypatch.delenv('DLM_MODEL_LOAD_LOCK_DIR', raising=False)
    monkeypatch.setattr(models, '_load_model_and_tokenizer', lambda *a, **k: ('unchanged', k))
    assert models.load_model_and_tokenizer('b', 'a', torch.device('cuda:0'))[0] == 'unchanged'
    monkeypatch.setenv('DLM_MODEL_LOAD_LOCK_DIR', str(tmp_path/'locks'))
    assert models.load_model_and_tokenizer('b', 'a', torch.device('cpu'))[0] == 'unchanged'
    assert not (tmp_path/'locks').exists()
