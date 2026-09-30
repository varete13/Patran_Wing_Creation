"""Option C: Patran meshing commands appended to the SES file."""
import re

from conftest import make_wing
from perfilador import save_ses
from perfilador.exporters.ses import _id_list


def test_id_list_compacts_ranges():
    assert _id_list([7, 1, 2, 3, 9, 10]) == "1:3 7 9:10"


def _ses(tmp_path, variant, **kw):
    path = tmp_path / "w.ses.01"
    save_ses(make_wing(variant), str(path), **kw)
    return path.read_text()


def test_geometry_only_by_default(tmp_path):
    assert "fem_create_mesh" not in _ses(tmp_path, "cuts_hollow")


def test_mesh_commands_cover_every_surface(tmp_path):
    text = _ses(tmp_path, "cuts_hollow", mesh=True, mesh_size=0.25)
    created = {int(i) for i in re.findall(
        r'sgm_(?:create_surface_trimmed_v1|const_surface_vertex)\(\s*"(\d+)"', text)}
    meshed = set()
    for ids in re.findall(r'fem_create_mesh_surf_4\( "\w+", 49152, "Surface ([\d: ]+)"', text):
        for tok in ids.split():
            a, _, b = tok.partition(":")
            meshed.update(range(int(a), int(b or a) + 1))
    assert created and meshed == created
    assert '"Paver"' in text and '"IsoMesh"' in text
    assert '["0.25"]' in text
    assert "fem_create_mesh_curv_1" in text
    assert text.rstrip().splitlines()[-1].startswith("fem_equiv_all_group4")


def test_cut_subsections_are_paver_meshed(tmp_path):
    text = _ses(tmp_path, "cuts_hollow", mesh=True)
    # 4 ribs x 4 cut sub-sections, all trimmed surfaces
    rib_ids = re.findall(r'sgm_create_surface_trimmed_v1\( "(\d+)"', text)
    assert len(rib_ids) == 4 * 4
    paver = re.search(r'"Paver", 49152, "Surface ([\d: ]+)"', text).group(1)
    assert paver.split()[0].startswith(rib_ids[0])
