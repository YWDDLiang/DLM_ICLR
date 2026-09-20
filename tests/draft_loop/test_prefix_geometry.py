import numpy as np
from dlm_iclr.draft_loop.prefix_geometry import visible_geometry_features


def test_visible_periodic_geometry_uses_lattice_and_preserves_translation():
    coords=np.array([[.99,0.,0.],[.01,0.,0.],[0.,0.,np.nan]])
    known=np.isfinite(coords);lattice=np.eye(3)*10
    a=visible_geometry_features(lattice,coords,known,0)
    b=visible_geometry_features(lattice,(coords+.3)%1,known,0)
    np.testing.assert_allclose(a,b,atol=1e-12)
    assert a[1]==.5 and a[2]==1.
    np.testing.assert_allclose(a[3],np.log1p(.2))
    c=visible_geometry_features(lattice*2,coords,known,0)
    np.testing.assert_allclose(c[3],np.log1p(.4))


def test_unknown_neighbors_do_not_turn_into_imputed_collisions():
    coords=np.array([[.1,.2,.3],[.1,.2,np.nan]])
    out=visible_geometry_features(np.eye(3),coords,np.isfinite(coords),0)
    assert out[1]==0 and np.array_equal(out[2:],np.zeros(6))
