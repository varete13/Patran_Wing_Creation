"""Regression tests for ID allocation, inner cuts in the BDF and entity counts."""
from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pytest
from pyNastran.bdf.bdf import read_bdf

from perfilador import FEAProperties, NACAAirfoil, Rib, Wing, save_bdf, save_ses
from perfilador.exporters.ses import GROUP_SKINS, GROUP_SPARS, _rib_geometry_commands
from perfilador.id_manager import IDManager

SPAN = 35.0
BASE = dict(
    span=SPAN,
    spar_positions=[0.2, 0.65],
    stringer_positions=np.linspace(0.05, 0.8, 8),
    chord_distribution=([0, 5.18, SPAN], [9, 7, 2]),
    offset_distribution=([0, 5.18, SPAN], [0, 2.42, 13.06]),
    twist_distribution=([0, SPAN], [np.deg2rad(2), -np.deg2rad(4)]),
    elevation_distribution=([0, SPAN], [0.0, SPAN * np.sin(np.deg2rad(6))]),
)


def _wing(**kw) -> Wing:
    params = dict(airfoil=NACAAirfoil("6412"), n_ribs=4, n_skin_points=25,
                  point_spacing="cosine", **BASE)
    params.update(kw)
    return Wing(**params)


def _ses_text(wing: Wing, tmp_path) -> str:
    path = tmp_path / "wing.ses.01"
    save_ses(wing, str(path))
    return path.read_text()


def _grid_ids(text: str) -> list[int]:
    return [int(i) for i in re.findall(r'asm_const_grid_xyz\("(\d+)', text)]


def _surface_ids(text: str) -> list[int]:
    return [int(i) for i in re.findall(
        r'sgm_(?:create_surface_trimmed_v1|const_surface_vertex)\(\s*"(\d+)', text)]


def _duplicates(ids: list[int]) -> list[int]:
    return [k for k, v in Counter(ids).items() if v > 1]


# ---------------------------------------------------------------------------
# Element and GRID ID ranges in the BDF
# ---------------------------------------------------------------------------

def test_bdf_fine_model_has_unique_element_ids(tmp_path):
    """Rib elements used to run into the spar element range (EID 20001)."""
    wing = _wing(n_ribs=20, n_skin_points=100, min_point_spacing=None)
    path = tmp_path / "fine.bdf"
    save_bdf(wing, str(path), FEAProperties(), n_rib_layers=4, n_span_div=4)

    model = read_bdf(str(path), debug=None)  # raises on duplicate IDs
    counts = Counter(e.pid for e in model.elements.values())
    n_bays, n_unique = 19, len(wing.ribs[0].profile[0]) - 1
    assert counts[2] == 2 * 4 * n_bays            # spar webs
    assert counts[3] == n_unique * 4 * n_bays     # skin panels
    assert counts[4] == 16 * 4 * n_bays           # stringer bars


@pytest.mark.parametrize("n_ribs", [100, 101])
def test_many_ribs_keep_id_ranges_apart(tmp_path, n_ribs):
    """With 100 ribs the last rib used to start at GRID 200000 and surface 2000."""
    wing = _wing(n_ribs=n_ribs, n_skin_points=20)
    s = wing.id_scheme
    first, last = wing.ribs[0].ids, wing.ribs[-1].ids

    def disjoint(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return a[1] < b[0] or b[1] < a[0]

    outer_points = (first.point_start, last.point_end)
    inner_points = (first.inner_point_start, last.inner_point_start + s.inner_point_stride - 1)
    rib_surfaces = (first.surface_base, last.surface_base + s.surface_rib_stride - 1)
    spar_surfaces = (s.surface_spar_base + s.surface_spar_stride,
                     s.surface_spar_base + s.surface_spar_stride * n_ribs - 1)
    assert disjoint(outer_points, inner_points)
    assert disjoint(rib_surfaces, spar_surfaces)

    text = _ses_text(wing, tmp_path)
    assert not _duplicates(_grid_ids(text))
    assert not _duplicates(_surface_ids(text))

    path = tmp_path / "many.bdf"
    save_bdf(wing, str(path), FEAProperties())
    read_bdf(str(path), debug=None)


def test_id_manager_does_not_mutate_given_scheme():
    from perfilador.id_manager import IDScheme
    scheme = IDScheme()
    IDManager(n_ribs=500, n_skin_points=20, n_spars=2, scheme=scheme)
    assert scheme == IDScheme()


def test_default_scheme_unchanged_for_typical_wings():
    """Existing Patran sessions depend on the default numbering."""
    wing = _wing(n_ribs=10)
    s = wing.id_scheme
    assert (s.point_base, s.surface_rib_base, s.surface_spar_base,
            s.surface_skin_base, s.inner_point_base) == (100_000, 1_000, 2_000, 30_000, 200_000)


# ---------------------------------------------------------------------------
# Inner cuts in the BDF
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("cut_wall", [None, [0.15, 0.12, 0.10]])
def test_bdf_meshes_cut_subsections(tmp_path, cut_wall):
    wing = _wing(inner_cuts=[(2, 43)], cut_wall_thickness=cut_wall)
    path = tmp_path / "cuts.bdf"
    save_bdf(wing, str(path), FEAProperties(), n_rib_layers=2)
    model = read_bdf(str(path), debug=None)

    n_rib_elems = sum(1 for e in model.elements.values() if e.pid == 1)
    expected = sum(
        len(sub) for rib in wing.ribs for subs in rib.cut_sections.values() for sub in subs
    ) * 2
    assert n_rib_elems == expected

    # The cut edge must be an element edge: both of its end nodes belong to
    # rib elements of two different sub-sections.
    rib0 = wing.ribs[0]
    ga, gb = (rib0.ids.point_start + i for i in (2, 43))
    touching = [e for e in model.elements.values()
                if e.pid == 1 and ga in e.node_ids and gb in e.node_ids]
    assert len(touching) == 2


def test_bdf_without_cuts_differs_from_bdf_with_cuts(tmp_path):
    counts = []
    for kw in ({}, {"inner_cuts": [(2, 43)]}):
        path = tmp_path / "w.bdf"
        save_bdf(_wing(**kw), str(path), FEAProperties())
        counts.append(len(read_bdf(str(path), debug=None).elements))
    assert counts[0] != counts[1]


# ---------------------------------------------------------------------------
# Hollow sections combined with inner cuts
# ---------------------------------------------------------------------------

def test_wall_thickness_with_inner_cuts_is_rejected():
    with pytest.raises(ValueError, match="cut_wall_thickness"):
        _wing(wall_thickness=[0.08, 0.10, 0.06], inner_cuts=[(2, 43)])


def test_ses_inner_grids_unique_when_rib_has_both_cavity_kinds():
    """A Rib built directly can still carry both; its SES grids must not repeat."""
    ids = IDManager(1, 25, 2).allocate_rib()
    rib = Rib(
        span_position=0.0, airfoil=NACAAirfoil("6412"), chord=9.0, offset=0.0,
        twist=0.0, elevation=0.0, spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8), ids=ids,
        n_skin_points=25, point_spacing="cosine",
        wall_thickness=[0.08, 0.10, 0.06], cut_pairs=[(2, 43)],
        cut_wall_thickness=[0.15, 0.12, 0.10],
    )
    cmds, _ = _rib_geometry_commands(rib, ids.surface_base)
    assert not _duplicates(_grid_ids("".join(cmds)))


def test_ses_raises_when_inner_grids_overflow_rib_range():
    ids = IDManager(1, 25, 2).allocate_rib()
    rib = Rib(
        span_position=0.0, airfoil=NACAAirfoil("6412"), chord=9.0, offset=0.0,
        twist=0.0, elevation=0.0, spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8), ids=ids,
        n_skin_points=25, point_spacing="cosine", wall_thickness=[0.08, 0.10, 0.06],
    )
    with pytest.raises(ValueError, match="inner grid IDs"):
        _rib_geometry_commands(rib, ids.surface_base, inner_stride=5)


# ---------------------------------------------------------------------------
# Entity summary and budget
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kw", [
    {},
    {"wall_thickness": [0.08, 0.10, 0.06]},
    {"inner_cuts": [(2, 43)]},
    {"inner_cuts": [(2, 43)], "cut_wall_thickness": [0.15, 0.12, 0.10]},
])
def test_summary_matches_ses_output(tmp_path, kw):
    wing = _wing(**kw)
    text = _ses_text(wing, tmp_path)
    s = wing.summary()

    grids = _grid_ids(text)
    inner_base = wing.id_scheme.inner_point_base
    assert s.grid_points == sum(1 for g in grids if g < inner_base)
    assert s.inner_grid_points == sum(1 for g in grids if g >= inner_base)
    assert s.rib_surfaces == text.count("sgm_create_surface_trimmed_v1(")
    assert s.rib_curves == text.count("asm_const_line_pwl(")
    assert s.total_lines == text.count("asm_const_line_2point(")
    assert s.spar_panel_surfaces == text.count(f'ga_group_entity_add( "{GROUP_SPARS}"')
    assert s.skin_panel_surfaces == text.count(f'ga_group_entity_add( "{GROUP_SKINS}"')


def test_max_skin_points_is_the_largest_value_within_budget():
    budget = 1_190
    n = Wing.max_skin_points(n_ribs=10, n_spars=2, n_stringers=8, max_entities=budget)

    def total(n_sp: int) -> int:
        return _wing(n_ribs=10, n_skin_points=n_sp, min_point_spacing=None).summary().total_entities

    assert total(n) <= budget < total(n + 1)
