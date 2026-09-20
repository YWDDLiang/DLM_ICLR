import torch
import pytest
from dlm_iclr.diffusion.refinement import refinement_noise
from dlm_iclr.diffusion.batching import PendingBatches


def test_refiner_noise_independent_of_batch_membership():
    combined = refinement_noise([17, 81], [3, 5], 8, 'cpu')
    reordered = refinement_noise([81, 17], [5, 3], 8, 'cpu')
    single = refinement_noise([81], [5], 8, 'cpu')
    for key in combined:
        start = 1 if key == 'predictor_lattice' else 3
        width = 1 if key == 'predictor_lattice' else 5
        torch.testing.assert_close(combined[key][:, start:], single[key], rtol=0, atol=0)
        torch.testing.assert_close(reordered[key][:, :width], single[key], rtol=0, atol=0)


def test_resume_retains_full_original_batch(tmp_path):
    (tmp_path/'refined').mkdir()
    for i in (0, 2, 4):
        (tmp_path/'refined'/f'{i:06d}.json').write_text('{}')
    groups = PendingBatches(list(range(0, 12, 2)), 2, tmp_path)
    assert list(groups) == [[4, 6], [8, 10]]


def test_sparse_fc_edges_equal_original_order():
    pytest.importorskip('torch_scatter')
    from dlm_iclr._vendor.crysllmgen.models_ddpm.cspnet import CSPNet
    from torch_geometric.utils import dense_to_sparse

    sizes = torch.tensor([1, 4, 7])
    coordinates = torch.rand(sum(sizes), 3)
    edges, difference = CSPNet().gen_edges(sizes, coordinates, None, None)
    expected, _ = dense_to_sparse(torch.block_diag(*(torch.ones(n, n) for n in sizes)))
    torch.testing.assert_close(edges, expected, rtol=0, atol=0)
    torch.testing.assert_close(difference, (coordinates[expected[1]]-coordinates[expected[0]]) % 1)
