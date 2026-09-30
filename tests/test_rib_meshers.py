"""Quality and conformity checks shared by every structured rib mesher."""
import numpy as np
import pytest
from pyNastran.bdf.bdf import read_bdf

from conftest import VARIANTS, make_wing
from perfilador import save_bdf
from perfilador.mesh import (
    RIB_MESHERS, expected_area, free_edges, mesh_quality, mesh_rib_face,
    rib_patches,
)
from perfilador.mesh.topology import rib_points

try:
    import gmsh  # noqa: F401
    MESHERS = sorted(RIB_MESHERS)
except ImportError:  # gmsh is optional
    MESHERS = sorted(set(RIB_MESHERS) - {"gmsh"})
_WINGS = {}


def wing(variant):
    if variant not in _WINGS:
        _WINGS[variant] = make_wing(variant)
    return _WINGS[variant]


@pytest.mark.parametrize("variant", sorted(VARIANTS))
@pytest.mark.parametrize("method", MESHERS)
def test_rib_face_mesh(method, variant):
    for rib in wing(variant).ribs:
        mesh = mesh_rib_face(rib, method, n_web=4, n_wall=2)
        q = mesh_quality(mesh)
        assert q.n_invalid == 0, (rib.span_position, q)

        # geometry honoured: patches minus cavities (cavity contours are
        # re-discretised, hence the tolerance)
        area = expected_area(rib, rib_patches(rib))
        assert q.area == pytest.approx(area, rel=0.06)

        # conformity: every skin segment is a free edge of the rib face, and
        # every other free edge lies on a cavity (never on a web or cut line)
        n = len(rib_points(rib))
        skin = {frozenset((("p", i), ("p", (i + 1) % n))) for i in range(n)}
        fe = set(free_edges(mesh))
        assert skin <= fe
        for edge in fe - skin:
            assert all(k[0] not in ("p", "w") for k in edge), edge


@pytest.mark.parametrize("method", MESHERS)
def test_cut_cavities_reach_the_bdf(method, tmp_path, fea):
    """With cuts + cavities the rib faces must have holes (the legacy
    exporter ignored inner_cuts and wrote a solid rib)."""
    solid = tmp_path / "solid.bdf"
    hollow = tmp_path / "hollow.bdf"
    save_bdf(make_wing("cuts"), str(solid), fea=fea, rib_mesher=method, n_web=4)
    save_bdf(make_wing("cuts_hollow"), str(hollow), fea=fea,
             rib_mesher=method, n_web=4)

    def rib_area(path):
        m = read_bdf(str(path), debug=None)
        return sum(e.Area() for e in m.elements.values() if e.pid == 1)

    assert rib_area(hollow) < 0.8 * rib_area(solid)


@pytest.mark.parametrize("method", MESHERS)
def test_spar_webs_share_rib_nodes(method, tmp_path, fea):
    path = tmp_path / "w.bdf"
    save_bdf(make_wing("hollow"), str(path), fea=fea, rib_mesher=method,
             n_web=4, n_span_div=2)
    m = read_bdf(str(path), debug=None)
    rib_nodes = {n for e in m.elements.values() if e.pid == 1 for n in e.node_ids}
    spar_nodes = {n for e in m.elements.values() if e.pid == 2 for n in e.node_ids}
    # 4 ribs x 2 spars x 5 nodes through the height, all in rib faces
    shared = rib_nodes & spar_nodes
    assert len(shared) == 4 * 2 * 5
    n_spar = sum(e.pid == 2 for e in m.elements.values())
    assert n_spar == 3 * 2 * 2 * 4  # bays x spars x span_div x n_web
