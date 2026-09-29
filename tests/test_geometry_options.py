import numpy as np
import pytest

from conftest import BASE, make_wing
from perfilador import Wing, save_bdf


def test_chord_fraction_cut_matches_index_cut():
    by_index = make_wing("cuts")
    by_fraction = Wing(n_skin_points=25, inner_cuts=[(0.017, 0.1)], **BASE)
    assert by_fraction.ribs[0].cut_sections == by_index.ribs[0].cut_sections


def test_chord_fraction_cut_survives_resolution_change():
    # (2, 43) is only valid for 25 points; the chord form works for any count
    with pytest.raises(ValueError):
        Wing(n_skin_points=60, inner_cuts=[(2, 43)], **BASE)
    w = Wing(n_skin_points=60, inner_cuts=[(0.017, 0.1)], **BASE)
    assert len(w.ribs[0].cut_sections["LE"]) == 2


def test_exact_stations_places_spars_on_request():
    # Default: the rear spar (0.65c) snaps to the 0.629c station
    xs = Wing(n_skin_points=25, **BASE)._x_stations
    assert np.min(np.abs(xs - 0.65)) > 1e-2
    w = Wing(n_skin_points=25, exact_stations=True, **BASE)
    for sp in BASE["spar_positions"] + BASE["stringer_positions"]:
        assert np.min(np.abs(w._x_stations - sp)) < 1e-12
    rib = w.ribs[0]
    assert np.allclose(rib.profile[0][np.array(rib.spar_ids) - rib.ids.point_start],
                       rib.spar_vertices[0])


def test_exact_stations_roundtrip_json(tmp_path):
    w = Wing(n_skin_points=25, exact_stations=True,
             inner_cuts=[(0.017, 0.1)], **BASE)
    w.to_json(tmp_path / "w.json")
    w2 = Wing.from_json(tmp_path / "w.json")
    assert np.allclose(w2._x_stations, w._x_stations)
    assert w2.ribs[0].cut_sections == w.ribs[0].cut_sections


def test_legacy_rejects_web_subdivision(tmp_path, fea):
    with pytest.raises(ValueError):
        save_bdf(make_wing("solid"), str(tmp_path / "x.bdf"), fea=fea, n_web=3)


def test_legacy_export_still_works(tmp_path, fea):
    save_bdf(make_wing("hollow"), str(tmp_path / "x.bdf"), fea=fea)
    assert (tmp_path / "x.bdf").stat().st_size > 0
