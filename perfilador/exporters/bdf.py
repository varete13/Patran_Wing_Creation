"""Nastran BDF exporter — full mesh with materials and properties.

Generates a BDF file via pyNastran containing:
- MAT1 cards for ribs, spars, skin and stringers.
- PSHELL cards for the three shell structural groups.
- PBAR card for the stringer bar elements.
- GRID cards for all outer-profile nodes plus inner/intermediate ring nodes.
- Rib-face CQUAD4/CTRIA3 mesh built by the selected *rib_mesher* (see
  :mod:`perfilador.mesh`), which honours hollow sections, ``inner_cuts`` and
  their cavities.  ``"legacy"`` keeps the original concentric-ring mesh.
- CQUAD4 spar webs, subdivided into *n_web* elements through the height and
  sharing those nodes with the rib faces, and skin panels between adjacent
  ribs, with optional spanwise subdivision controlled by *n_span_div*.
- CBAR stringer elements between adjacent ribs, also subdivided by *n_span_div*.

Coordinate convention (consistent with the SES exporter):
    GRID X1 = chordwise direction (LE → TE)
    GRID X2 = airfoil thickness direction
    GRID X3 = spanwise direction (root → tip)
"""
from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

import numpy as np
from pyNastran.bdf.bdf import BDF

if TYPE_CHECKING:
    from ..rib import Rib
    from ..wing import Wing

from ..fea_props import BarProps, FEAProperties, MatProps
from ..mesh import mesh_rib_face
from ..mesh.topology import RibMesh, rib_points, web_key, web_node_xy

# -------------------------------------------------------------------------
# Property / material IDs (fixed, no collision with GRID or element IDs)
# -------------------------------------------------------------------------
_MID_RIB  = 1
_MID_SPAR = 2
_MID_SKIN = 3
_MID_STR  = 4

_PID_RIB  = 1
_PID_SPAR = 2
_PID_SKIN = 3
_PID_STR  = 4

# Element ID counters start well above the GRID ID space
_EID_RIB_START      = 10_001
_EID_SPAR_START     = 20_001
_EID_SKIN_START     = 60_001
_EID_STRINGER_START = 70_001

# Inner / intermediate ring node IDs start above the outer profile range
# (max outer profile ID ≈ 100_000 + 1_000 * n_ribs + 2*n_skin_points ≤ ~200_000)
_GID_INNER_START = 200_001

# Spanwise interpolated nodes (skin / spar / stringer intermediate rows)
_GID_INTERP_START = 300_001

# Virtual inner contour scale for solid rib sections (no cavity).
# The innermost ring sits at 5 % of the outer section size around the centroid,
# leaving a central void of ≈ 0.25 % of the section area — negligible for
# thin-shell structural analysis.
_SOLID_INNER_SCALE = 0.05

# Corner-attraction strength for inner-contour resampling.
# Each segment adjacent to a sharp corner is inflated by up to (1 + _CORNER_BIAS)×
# in the weighted arc-length, drawing extra sample points toward corners and
# reducing element distortion in the concentric rib mesh near cavity corners.
# 0 → pure uniform arc-length (original behaviour); 4 → ~3× more nodes at 90° corners.
_CORNER_BIAS = 4.0


# -------------------------------------------------------------------------
# Geometry helpers
# -------------------------------------------------------------------------

def _resample_contour(coords: np.ndarray, n: int) -> np.ndarray:
    """Resample a closed polygon to *n* unique points by uniform arc-length.

    Parameters
    ----------
    coords : (M, 2) array
        Closed polygon (first point == last point).
    n : int
        Number of unique output points.

    Returns
    -------
    (n, 2) array — not closed.
    """
    diffs = np.diff(coords, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    arc = np.concatenate([[0.0], np.cumsum(seg_len)])
    s = np.linspace(0.0, arc[-1], n, endpoint=False)
    x = np.interp(s, arc, coords[:, 0])
    y = np.interp(s, arc, coords[:, 1])
    return np.column_stack([x, y])


def _curvature_biased_resample(
    coords: np.ndarray,
    n: int,
    bias: float = _CORNER_BIAS,
) -> np.ndarray:
    """Resample a closed polygon to *n* points, concentrating nodes near corners.

    Each polygon segment is inflated in a *weighted* arc-length proportional
    to the average turning angle of its two endpoint vertices.  Sampling
    uniformly in this weighted space draws extra points toward high-curvature
    regions (sharp corners), then the coordinates are recovered by
    interpolating on the real arc-length.

    This significantly reduces element distortion in the concentric rib mesh
    near inner-cavity corners produced by the Shapely ``offset_polygon``.

    Parameters
    ----------
    coords : (M, 2) array
        Closed polygon (first point == last point).
    n : int
        Number of unique output points.
    bias : float
        Corner-attraction strength ≥ 0.  ``0`` → identical to pure arc-length
        resampling.  At a 90° corner with ``bias=4`` the adjacent segments are
        inflated by ~3×, roughly tripling the node count there.

    Returns
    -------
    (n, 2) array — not closed.
    """
    # Ensure the polygon is closed
    if not np.allclose(coords[0], coords[-1], atol=1e-12):
        coords = np.vstack([coords, coords[:1]])

    pts = coords[:-1]           # unique vertices
    m   = len(pts)

    # Turning angle at each vertex: 0 = straight, π = 180° reversal
    v_in  = pts - np.roll(pts,  1, axis=0)
    v_out = np.roll(pts, -1, axis=0) - pts
    l_in  = np.linalg.norm(v_in,  axis=1, keepdims=True).clip(min=1e-12)
    l_out = np.linalg.norm(v_out, axis=1, keepdims=True).clip(min=1e-12)
    cos_t = np.clip((v_in / l_in * v_out / l_out).sum(axis=1), -1.0, 1.0)
    turn  = np.arccos(cos_t)    # [0, π]

    # Geometric arc-length
    diffs   = np.diff(coords, axis=0)
    seg_len = np.hypot(diffs[:, 0], diffs[:, 1])
    arc     = np.concatenate([[0.0], np.cumsum(seg_len)])

    # Inflate each segment by the average turning-angle weight of its endpoints
    # w ∈ [1, 1+bias]:  straight edge → 1×,  90° corner → ~(1 + bias/2)×
    w_vert = 1.0 + bias * turn / np.pi
    w_seg  = 0.5 * (w_vert + np.roll(w_vert, -1))
    arc_w  = np.concatenate([[0.0], np.cumsum(seg_len * w_seg)])

    # Sample uniformly in weighted space → project back to geometric arc
    s_w = np.linspace(0.0, arc_w[-1], n, endpoint=False)
    s   = np.interp(s_w, arc_w, arc)

    x = np.interp(s, arc, coords[:, 0])
    y = np.interp(s, arc, coords[:, 1])
    return np.column_stack([x, y])


def _aligned_resample(inner_poly: np.ndarray, n: int,
                      ref_point: np.ndarray) -> np.ndarray:
    """Resample *inner_poly* to *n* points with corner-biased sampling,
    rotating it so its first node is as close as possible to *ref_point*
    (= outer ring node 0).

    Corner-biased resampling (:func:`_curvature_biased_resample`) is used so
    that sharp corners of the inner cavity contour (produced by Shapely's
    morphological offset) attract more nodes, preventing very thin/slender
    elements in the concentric rib mesh near those corners.
    """
    pts = inner_poly[:-1]                          # drop closing duplicate
    dists = np.linalg.norm(pts - ref_point, axis=1)
    start = int(np.argmin(dists))
    pts_rolled = np.roll(pts, -start, axis=0)
    closed = np.vstack([pts_rolled, pts_rolled[:1]])
    return _curvature_biased_resample(closed, n)


def _section_gids_and_coords(
    rib: Rib,
) -> list[tuple[list[int], np.ndarray]]:
    """Return outer-ring GIDs and coordinates per section.

    Mirrors the index logic of :func:`geometry.extract_sections` but also
    returns the Nastran GIDs already assigned to the outer profile nodes.

    Returns
    -------
    list of (gids, coords) per section (LE → TE order):
        *gids*  — ordered list of existing GIDs for the outer ring.
        *coords* — (N, 2) ``[x, z]`` coordinates matching *gids*.
    """
    profile_x, profile_y = rib.profile
    n_sp = rib.n_skin_points
    n_spars = len(rib.spar_positions)
    ps = rib.ids.point_start

    upper_idx = [rib.spar_ids[i] - ps for i in range(n_spars)]
    lower_idx = list(reversed(
        [rib.spar_ids[n_spars + i] - ps for i in range(n_spars)]
    ))

    upper_bounds = [0] + upper_idx + [n_sp - 1]
    # profile[len(profile_x)-1] is a duplicate of the LE upper node (index 0),
    # added by transform_airfoil_points.  Use index len-2 so the ring wraps
    # from the last unique lower-surface node directly back to node 0 (LE),
    # keeping all GIDs within the created range ps+0 … ps+n_unique-1.
    lower_bounds = [len(profile_x) - 2] + lower_idx + [n_sp]

    result: list[tuple[list[int], np.ndarray]] = []
    for s in range(n_spars + 1):
        u_start = upper_bounds[s]
        u_end   = upper_bounds[s + 1]
        l_from  = lower_bounds[s + 1]
        l_to    = lower_bounds[s]

        gids = (
            [ps + i for i in range(u_start, u_end + 1)]
            + [ps + i for i in range(l_from, l_to + 1)]
        )
        coords = np.column_stack([
            np.concatenate([profile_x[u_start:u_end + 1],
                            profile_x[l_from:l_to + 1]]),
            np.concatenate([profile_y[u_start:u_end + 1],
                            profile_y[l_from:l_to + 1]]),
        ])
        result.append((gids, coords))

    return result


# -------------------------------------------------------------------------
# pyNastran model builders
# -------------------------------------------------------------------------

def _add_rib_outer_grids(rib: Rib, model: BDF) -> None:
    """Add the 2*N-2 outer profile GRID nodes of *rib* to *model*."""
    px, py = rib.profile
    n_unique = len(px) - 1       # skip last point (duplicate LE)
    y = rib.span_position
    ps = rib.ids.point_start
    for j in range(n_unique):
        model.add_grid(ps + j, [float(px[j]), float(py[j]), y])


def _add_rib_face(
    rib: Rib,
    n_layers: int,
    model: BDF,
    gid_counter,
    eid_counter,
) -> None:
    """Add concentric-ring CQUAD4 face mesh for *rib*.

    For each section (LE / Box / TE) the mesh has *n_layers* layers of
    CQUAD4 from the outer profile ring inward to either the inner cavity
    contour (hollow rib) or a virtual inner contour (solid rib).

    Solid sections use a virtual inner contour computed by contracting the
    outer ring by :data:`_SOLID_INNER_SCALE` toward the section centroid.
    This produces proper (non-degenerate) CQUAD4 elements throughout.
    """
    y = rib.span_position
    n_secs = len(rib.spar_positions) + 1
    inner_profs = rib.inner_profiles or ([None] * n_secs)

    for sec_idx, (outer_gids, outer_coords) in enumerate(
        _section_gids_and_coords(rib)
    ):
        n = len(outer_gids)
        inner_prof = inner_profs[sec_idx]

        if inner_prof is not None:
            ix, iy = inner_prof
            inner_coords = _aligned_resample(
                np.column_stack([ix, iy]), n, outer_coords[0]
            )
        else:
            # Solid section: virtual inner contour contracted toward centroid.
            centroid = outer_coords.mean(axis=0)
            inner_coords = centroid + _SOLID_INNER_SCALE * (outer_coords - centroid)

        rings: list[list[int]] = [outer_gids]

        for layer in range(1, n_layers + 1):
            t = layer / n_layers
            rc = (1.0 - t) * outer_coords + t * inner_coords
            rg: list[int] = []
            for j in range(n):
                gid = next(gid_counter)
                model.add_grid(
                    gid, [float(rc[j, 0]), float(rc[j, 1]), y]
                )
                rg.append(gid)
            rings.append(rg)

        for k in range(n_layers):
            r0, r1 = rings[k], rings[k + 1]
            for j in range(n):
                j1 = (j + 1) % n
                model.add_cquad4(
                    next(eid_counter), _PID_RIB,
                    [r0[j], r0[j1], r1[j1], r1[j]],
                )


def _add_rib_mesh(
    rib: Rib,
    mesh: RibMesh,
    model: BDF,
    gid_counter,
    eid_counter,
) -> dict:
    """Write a :class:`RibMesh` into *model* as GRID + CQUAD4/CTRIA3 cards.

    Profile nodes (``("p", i)``) map to the existing outer GRIDs; every other
    node gets a new GID from *gid_counter*.  Returns the ``key -> GID`` map so
    the spar webs can reuse the web-edge nodes.
    """
    y = rib.span_position
    ps = rib.ids.point_start
    gid: dict = {}
    for key, xy in mesh.nodes.items():
        if key[0] == "p":
            gid[key] = ps + key[1]
        else:
            g = next(gid_counter)
            model.add_grid(g, [float(xy[0]), float(xy[1]), y])
            gid[key] = g
    for elem in mesh.elements:
        nids = [gid[k] for k in elem]
        if len(nids) == 4:
            model.add_cquad4(next(eid_counter), _PID_RIB, nids)
        else:
            model.add_ctria3(next(eid_counter), _PID_RIB, nids)
    return gid


def _interp_span_rows(
    r: Rib,
    r1: Rib,
    n_sub: int,
    model: BDF,
    gid_counter,
) -> list[list[int]]:
    """Create *n_sub* - 1 intermediate profile-node rows between ribs *r* / *r1*.

    Nodes are linearly interpolated in 3-D between the two rib profiles.
    Each row has ``n_unique`` GIDs ordered identically to the outer profile
    (same local indices), so spar and stringer nodes can be extracted by
    local index.

    Returns
    -------
    list of length *n_sub* - 1, each entry a list of *n_unique* GIDs.
    An empty list is returned when *n_sub* ≤ 1.
    """
    if n_sub <= 1:
        return []
    n_unique = len(r.profile[0]) - 1
    px_r,  pz_r  = r.profile
    px_r1, pz_r1 = r1.profile
    y_r,   y_r1  = r.span_position, r1.span_position
    rows: list[list[int]] = []
    for k in range(1, n_sub):
        t = k / n_sub
        y_k = (1.0 - t) * y_r + t * y_r1
        row: list[int] = []
        for j in range(n_unique):
            gid = next(gid_counter)
            model.add_grid(gid, [
                float((1.0 - t) * px_r[j] + t * px_r1[j]),
                float((1.0 - t) * pz_r[j] + t * pz_r1[j]),
                y_k,
            ])
            row.append(gid)
        rows.append(row)
    return rows


def _spar_web_column(
    rib: Rib,
    k: int,
    n_web: int,
    rib_gids: dict,
    model: BDF,
    gid_counter,
) -> list[int]:
    """GIDs along spar *k* of *rib*, upper → lower (``n_web + 1`` nodes).

    Web-edge nodes already created by the rib-face mesher are reused; missing
    ones (e.g. with the legacy mesher) are created on the straight web.
    """
    ns = len(rib.spar_positions)
    ps = rib.ids.point_start
    iu = rib.spar_ids[k] - ps
    il = rib.spar_ids[2 * ns - 1 - k] - ps
    col = [ps + iu]
    P = None
    for s in range(1, n_web):
        key = web_key(iu, il, s, n_web)
        if key not in rib_gids:
            if P is None:
                P = rib_points(rib)
            xy = web_node_xy(P, key, n_web)
            g = next(gid_counter)
            model.add_grid(g, [float(xy[0]), float(xy[1]), rib.span_position])
            rib_gids[key] = g
        col.append(rib_gids[key])
    col.append(ps + il)
    return col


def _add_spar_elements(
    r: Rib,
    r1: Rib,
    model: BDF,
    eid_counter,
    interp_rows: list[list[int]] = (),
    n_web: int = 1,
    web_r: dict | None = None,
    web_r1: dict | None = None,
    gid_counter=None,
) -> None:
    """Add CQUAD4 spar web panels between adjacent ribs *r* / *r1*.

    Each spar web is split into ``len(interp_rows) + 1`` panels along the
    span and *n_web* panels through the height.  The height nodes at the ribs
    are shared with the rib-face mesh (*web_r*, *web_r1*: ``key -> GID``);
    those on the intermediate rows are interpolated along the straight web.
    """
    ns = len(r.spar_positions)
    web_r = {} if web_r is None else web_r
    web_r1 = {} if web_r1 is None else web_r1
    ps = r.ids.point_start

    for k in range(ns):
        iu = r.spar_ids[k] - ps
        il = r.spar_ids[2 * ns - 1 - k] - ps
        cols = [_spar_web_column(r, k, n_web, web_r, model, gid_counter)]
        for row in interp_rows:
            gu, gl = row[iu], row[il]
            col = [gu]
            if n_web > 1:
                xu = np.asarray(model.nodes[gu].xyz)
                xl = np.asarray(model.nodes[gl].xyz)
                for s in range(1, n_web):
                    t = s / n_web
                    g = next(gid_counter)
                    model.add_grid(g, list((1.0 - t) * xu + t * xl))
                    col.append(g)
            col.append(gl)
            cols.append(col)
        cols.append(_spar_web_column(r1, k, n_web, web_r1, model, gid_counter))

        for ca, cb in zip(cols, cols[1:]):
            for s in range(n_web):
                model.add_cquad4(
                    next(eid_counter), _PID_SPAR,
                    [ca[s], cb[s], cb[s + 1], ca[s + 1]],
                )


def _add_skin_elements(
    r: Rib,
    r1: Rib,
    model: BDF,
    eid_counter,
    interp_rows: list[list[int]] = (),
) -> None:
    """Add CQUAD4 skin panels — one per consecutive profile segment per bay.

    *interp_rows* subdivide each span bay into ``len(interp_rows) + 1``
    panels in the spanwise direction.
    """
    n_unique = len(r.profile[0]) - 1
    ps0 = r.ids.point_start
    ps1 = r1.ids.point_start
    r0_row = [ps0 + j for j in range(n_unique)]
    r1_row = [ps1 + j for j in range(n_unique)]
    all_rows = [r0_row] + list(interp_rows) + [r1_row]
    for row_a, row_b in zip(all_rows, all_rows[1:]):
        for j in range(n_unique):
            j1 = (j + 1) % n_unique
            model.add_cquad4(
                next(eid_counter), _PID_SKIN,
                [row_a[j], row_a[j1], row_b[j1], row_b[j]],
            )


def _add_stringer_elements(
    r: Rib,
    r1: Rib,
    model: BDF,
    eid_counter,
    interp_rows: list[list[int]] = (),
) -> None:
    """Add CBAR stringer elements between adjacent ribs.

    *interp_rows* subdivide each span bay into ``len(interp_rows) + 1``
    bar segments per stringer.
    """
    def _str_row(rib: Rib, row: list[int]) -> list[int]:
        p = rib.ids.point_start
        return [row[sid - p] for sid in rib.stringer_ids]

    r0_str = list(r.stringer_ids)
    r1_str = list(r1.stringer_ids)
    all_rows = (
        [r0_str]
        + [_str_row(r, row) for row in interp_rows]
        + [r1_str]
    )
    for row_a, row_b in zip(all_rows, all_rows[1:]):
        for ga, gb in zip(row_a, row_b):
            model.add_cbar(
                next(eid_counter), _PID_STR, [ga, gb],
                x=[1.0, 0.0, 0.0], g0=None,
            )


# -------------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------------

def save_bdf(
    wing: Wing,
    filename: str | None = None,
    fea: FEAProperties | None = None,
    n_rib_layers: int = 2,
    n_span_div: int = 1,
    rib_mesher: str = "native",
    n_web: int | None = None,
) -> None:
    """Write a Nastran BDF file with a full structural mesh via pyNastran.

    The file contains MAT1, PSHELL, PBAR, GRID, CQUAD4 and CBAR cards
    ready for a linear static analysis.

    Rib faces use a concentric CQUAD4 pattern with *n_rib_layers* radial
    rings per section.  Both hollow sections (with inner cavity) and solid
    sections (no cavity) produce proper, non-degenerate CQUAD4 elements.
    Solid sections use a virtual inner contour contracted 95 % toward the
    section centroid (leaving a negligible ~0.25 % central void).

    Skin panels, spar webs and stringer bars can be subdivided in the
    spanwise direction by *n_span_div* — each rib-to-rib bay is split into
    *n_span_div* strips with linearly interpolated intermediate nodes.

    Parameters
    ----------
    wing : Wing
        The wing geometry to export.
    filename : str or None
        Output filename.  Defaults to
        ``Wing_<span>m_<n_ribs>_sections.bdf``.
    fea : FEAProperties or None
        Material and section properties.  If *None*, ``wing.fea_properties``
        is used.  At least one of the two must be provided.
    n_rib_layers : int
        Number of concentric CQUAD4 rings per rib section (radial density).
        Default is ``2``.
    n_span_div : int
        Number of spanwise divisions per rib bay for skin, spar and stringer
        elements.  ``1`` (default) gives a single element per bay.
        Values > 1 insert ``n_span_div - 1`` intermediate profile-node rows.
    rib_mesher : str
        Rib-face mesher: ``"native"`` (default, structured, pure Python),
        ``"gmsh"`` (needs the ``gmsh`` package) or ``"legacy"`` (the original
        concentric-ring mesh, which ignores ``inner_cuts``).  For the
        structured meshers *n_rib_layers* is the number of element layers
        across each cavity wall.
    n_web : int or None
        Elements through the height of each spar web and along each cut line.
        Defaults to 4, or 1 with the legacy mesher (the only value it
        supports).

    Raises
    ------
    ValueError
        If no :class:`FEAProperties` are available.

    Examples
    --------
    Define material and section properties, then export:

    >>> from perfilador import FEAProperties, MatProps, BarProps, save_bdf

    >>> fea = FEAProperties(
    ...     rib=MatProps(
    ...         E=70e9,       # Young's modulus [Pa]
    ...         nu=0.33,      # Poisson's ratio
    ...         rho=2700.0,   # density [kg/m³]
    ...         t=0.003,      # shell thickness [m]
    ...     ),
    ...     spar=MatProps(E=70e9, nu=0.33, rho=2700.0, t=0.004),
    ...     skin=MatProps(E=70e9, nu=0.33, rho=2700.0, t=0.002),
    ...     stringer=MatProps(E=70e9, nu=0.33, rho=2700.0),
    ...     stringer_bar=BarProps(
    ...         A=1.5e-4,   # cross-section area [m²]
    ...         I1=2e-8,    # bending inertia axis-1 [m⁴]
    ...         I2=2e-8,    # bending inertia axis-2 [m⁴]
    ...         J=1e-8,     # torsional constant [m⁴]
    ...     ),
    ... )
    >>> save_bdf(wing, "wing.bdf", fea=fea)
    >>> save_bdf(wing, "wing_fine.bdf", fea=fea, n_rib_layers=3, n_span_div=2)
    """
    if fea is None:
        fea = wing.fea_properties
    if fea is None:
        raise ValueError(
            "FEAProperties are required for BDF export.  "
            "Pass fea= to save_bdf() or set wing.fea_properties."
        )

    if n_web is None:
        n_web = 1 if rib_mesher == "legacy" else 4
    if n_web < 1:
        raise ValueError("n_web must be >= 1")
    if rib_mesher == "legacy" and n_web != 1:
        raise ValueError(
            "The legacy rib mesher has no nodes along the spar webs; "
            "use n_web=1 or another rib_mesher."
        )

    if filename is None:
        filename = f"Wing_{wing.span:.0f}m_{wing.n_ribs:.0f}_sections.bdf"

    model = BDF(debug=False)

    # ------------------------------------------------------------------
    # Materials
    # ------------------------------------------------------------------
    model.add_mat1(_MID_RIB,  fea.rib.E,       None, fea.rib.nu,       fea.rib.rho)
    model.add_mat1(_MID_SPAR, fea.spar.E,      None, fea.spar.nu,      fea.spar.rho)
    model.add_mat1(_MID_SKIN, fea.skin.E,      None, fea.skin.nu,      fea.skin.rho)
    model.add_mat1(_MID_STR,  fea.stringer.E,  None, fea.stringer.nu,  fea.stringer.rho)

    # ------------------------------------------------------------------
    # 2-D properties (PSHELL) — isotropic: MID1 = MID2 = MID3
    # ------------------------------------------------------------------
    model.add_pshell(_PID_RIB,  _MID_RIB,  fea.rib.t,  mid2=_MID_RIB,  mid3=_MID_RIB)
    model.add_pshell(_PID_SPAR, _MID_SPAR, fea.spar.t, mid2=_MID_SPAR, mid3=_MID_SPAR)
    model.add_pshell(_PID_SKIN, _MID_SKIN, fea.skin.t, mid2=_MID_SKIN, mid3=_MID_SKIN)

    # ------------------------------------------------------------------
    # 1-D property (PBAR)
    # ------------------------------------------------------------------
    b = fea.stringer_bar
    model.add_pbar(_PID_STR, _MID_STR, A=b.A, i1=b.I1, i2=b.I2, j=b.J)

    # ------------------------------------------------------------------
    # Element ID counters (independent ranges per element group)
    # ------------------------------------------------------------------
    eid_rib      = itertools.count(_EID_RIB_START)
    eid_spar     = itertools.count(_EID_SPAR_START)
    eid_skin     = itertools.count(_EID_SKIN_START)
    eid_stringer = itertools.count(_EID_STRINGER_START)
    gid_inner    = itertools.count(_GID_INNER_START)
    gid_interp   = itertools.count(_GID_INTERP_START)

    # ------------------------------------------------------------------
    # Grid points + rib face elements
    # ------------------------------------------------------------------
    web_gids: list[dict] = []
    for rib in wing.ribs:
        _add_rib_outer_grids(rib, model)
        if rib_mesher == "legacy":
            _add_rib_face(rib, n_rib_layers, model, gid_inner, eid_rib)
            web_gids.append({})
        else:
            mesh = mesh_rib_face(rib, rib_mesher, n_web=n_web, n_wall=n_rib_layers)
            web_gids.append(
                _add_rib_mesh(rib, mesh, model, gid_inner, eid_rib)
            )

    # ------------------------------------------------------------------
    # Spanwise elements (spar webs, skin panels, stringers)
    # ------------------------------------------------------------------
    for i, (r, r1) in enumerate(zip(wing.ribs, wing.ribs[1:])):
        rows = _interp_span_rows(r, r1, n_span_div, model, gid_interp)
        _add_spar_elements(
            r, r1, model, eid_spar, rows,
            n_web=n_web, web_r=web_gids[i], web_r1=web_gids[i + 1],
            gid_counter=gid_interp,
        )
        _add_skin_elements(r, r1, model, eid_skin, rows)
        _add_stringer_elements(r, r1, model, eid_stringer, rows)

    # ------------------------------------------------------------------
    # ID-range overflow check (each group has a fixed, contiguous range)
    # ------------------------------------------------------------------
    for name, counter, limit in (
        ("rib element", eid_rib, _EID_SPAR_START),
        ("spar element", eid_spar, _EID_SKIN_START),
        ("skin element", eid_skin, _EID_STRINGER_START),
        ("rib-face GRID", gid_inner, _GID_INTERP_START),
    ):
        if next(counter) > limit:
            raise ValueError(
                f"Too many {name}s for the fixed ID range ending at {limit - 1}; "
                "reduce the mesh density."
            )

    # ------------------------------------------------------------------
    # Validate and write
    # ------------------------------------------------------------------
    model.cross_reference()
    model.write_bdf(filename)
