"""Nastran BDF exporter — full mesh with materials and properties.

Generates a BDF file via pyNastran containing:
- MAT1 cards for ribs, spars, skin and stringers.
- PSHELL cards for the three shell structural groups.
- PBAR card for the stringer bar elements.
- GRID cards for all outer-profile nodes plus inner/intermediate ring nodes.
- CQUAD4 concentric mesh per rib section (LE / Box / TE), or per inner-cut
  sub-section when the wing defines ``inner_cuts``; both hollow and solid
  regions use proper non-degenerate rings (solid regions get a virtual inner
  contour contracted 95% toward the region centroid).
- CQUAD4 spar webs and skin panels between adjacent ribs, with optional
  spanwise subdivision controlled by *n_span_div*.
- CBAR stringer elements between adjacent ribs, also subdivided by *n_span_div*.

Coordinate convention (consistent with the SES exporter):
    GRID X1 = chordwise direction (LE → TE)
    GRID X2 = airfoil thickness direction
    GRID X3 = spanwise direction (root → tip)
"""
from __future__ import annotations

import itertools
import logging
from typing import TYPE_CHECKING

import numpy as np
from pyNastran.bdf.bdf import BDF

if TYPE_CHECKING:
    from ..rib import Rib
    from ..wing import Wing

from ..fea_props import BarProps, FEAProperties, MatProps

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

# Preferred first element ID of each element group.  A group is moved up
# (see _range_starts) when the previous group would otherwise run into it.
_EID_RIB_START      = 10_001
_EID_SPAR_START     = 20_001
_EID_SKIN_START     = 60_001
_EID_STRINGER_START = 70_001

# Preferred first GRID ID of the rib ring nodes and of the spanwise
# interpolated rows.  Both are moved above the outer profile GRIDs (and
# above each other) when the model is large enough to need it.
_GID_INNER_START = 200_001
_GID_INTERP_START = 300_001

# Granularity used when a range has to be moved: the new start is the next
# multiple of this value plus one, which keeps IDs readable.
_ID_BLOCK = 10_000

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


def _rib_regions(
    rib: Rib,
) -> list[tuple[list[int], np.ndarray, np.ndarray | None]]:
    """Return the closed regions that make up the rib face.

    Without inner cuts there is one region per structural section
    (LE → TE), with the hollow-rib cavity as its inner contour.  With inner
    cuts there is one region per cut sub-section, in the same order as
    ``rib.cut_sections``, with the matching ``rib.cut_inner_profiles``
    entry as its inner contour.

    Returns
    -------
    list of (gids, coords, inner) per region:
        *gids*   — ordered GIDs of the outer ring.
        *coords* — (N, 2) coordinates matching *gids*.
        *inner*  — closed (M, 2) inner cavity contour, or ``None`` if solid.
    """
    if rib.cut_sections is None:
        n_secs = len(rib.spar_positions) + 1
        inner_profs = rib.inner_profiles or [None] * n_secs
        return [
            (gids, coords,
             None if prof is None else np.column_stack(prof))
            for (gids, coords), prof in zip(_section_gids_and_coords(rib), inner_profs)
        ]

    px, py = rib.profile
    ps = rib.ids.point_start
    subsections = [sub for subs in rib.cut_sections.values() for sub in subs]
    inner_profs = rib.cut_inner_profiles or [None] * len(subsections)

    regions: list[tuple[list[int], np.ndarray, np.ndarray | None]] = []
    for sub, prof in zip(subsections, inner_profs):
        if len(sub) < 3:
            logging.debug(
                "Cut sub-section with %d points at y=%.2fm has no area; "
                "no rib elements generated for it.", len(sub), rib.span_position,
            )
            continue
        idx = np.asarray(sub)
        regions.append((
            [ps + i for i in sub],
            np.column_stack([px[idx], py[idx]]),
            None if prof is None else np.column_stack(prof),
        ))
    return regions


def _range_starts(
    preferred: list[int], counts: list[int], floor: int = 0
) -> list[int]:
    """First ID of each consecutive ID range, free of overlaps.

    Each range keeps its *preferred* start when that start lies above the
    end of the previous range (or above *floor* for the first one).
    Otherwise it moves to the next multiple of ``_ID_BLOCK`` plus one.
    """
    starts: list[int] = []
    next_free = floor + 1
    for start, count in zip(preferred, counts):
        if start < next_free:
            start = -(-(next_free - 1) // _ID_BLOCK) * _ID_BLOCK + 1
        starts.append(start)
        next_free = start + count
    return starts


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

    For each region returned by :func:`_rib_regions` (a structural section,
    or an inner-cut sub-section) the mesh has *n_layers* layers of CQUAD4
    from the outer profile ring inward to either the inner cavity contour
    (hollow region) or a virtual inner contour (solid region).

    Solid regions use a virtual inner contour computed by contracting the
    outer ring by :data:`_SOLID_INNER_SCALE` toward the region centroid.
    This produces proper (non-degenerate) CQUAD4 elements throughout.
    """
    y = rib.span_position

    for outer_gids, outer_coords, inner_poly in _rib_regions(rib):
        n = len(outer_gids)

        if inner_poly is not None:
            inner_coords = _aligned_resample(inner_poly, n, outer_coords[0])
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


def _add_spar_elements(
    r: Rib,
    r1: Rib,
    model: BDF,
    eid_counter,
    interp_rows: list[list[int]] = (),
) -> None:
    """Add CQUAD4 spar web panels between adjacent ribs *r* / *r1*.

    *interp_rows* are intermediate profile-node rows produced by
    :func:`_interp_span_rows`.  Each spar creates one CQUAD4 per consecutive
    row pair (``len(interp_rows) + 1`` panels total).
    """
    ns = len(r.spar_positions)

    def _spar_row(rib: Rib, row: list[int]) -> list[int]:
        p = rib.ids.point_start
        return [row[rib.spar_ids[k] - p] for k in range(2 * ns)]

    r0_spar = list(r.spar_ids)
    r1_spar = list(r1.spar_ids)
    all_rows = (
        [r0_spar]
        + [_spar_row(r, row) for row in interp_rows]
        + [r1_spar]
    )
    for row_a, row_b in zip(all_rows, all_rows[1:]):
        for k in range(ns):
            model.add_cquad4(
                next(eid_counter), _PID_SPAR,
                [row_a[k], row_b[k],
                 row_b[2 * ns - 1 - k], row_a[2 * ns - 1 - k]],
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
    # ID counters (independent, non-overlapping ranges per group)
    # ------------------------------------------------------------------
    n_bays = len(wing.ribs) - 1
    n_div = max(n_span_div, 1)
    n_unique = len(wing.ribs[0].profile[0]) - 1
    n_ring_nodes = sum(
        len(gids) for rib in wing.ribs for gids, _, _ in _rib_regions(rib)
    ) * n_rib_layers

    eid_starts = _range_starts(
        [_EID_RIB_START, _EID_SPAR_START, _EID_SKIN_START, _EID_STRINGER_START],
        [
            n_ring_nodes,                                            # rib CQUAD4
            len(wing.spar_positions) * n_div * n_bays,               # spar CQUAD4
            n_unique * n_div * n_bays,                               # skin CQUAD4
            len(wing.ribs[0].stringer_ids) * n_div * n_bays,         # CBAR
        ],
    )
    gid_starts = _range_starts(
        [_GID_INNER_START, _GID_INTERP_START],
        [n_ring_nodes, n_unique * (n_div - 1) * n_bays],
        floor=max(rib.ids.point_end for rib in wing.ribs),
    )
    eid_rib, eid_spar, eid_skin, eid_stringer = map(itertools.count, eid_starts)
    gid_inner, gid_interp = map(itertools.count, gid_starts)

    # ------------------------------------------------------------------
    # Grid points + rib face elements
    # ------------------------------------------------------------------
    for rib in wing.ribs:
        _add_rib_outer_grids(rib, model)
        _add_rib_face(rib, n_rib_layers, model, gid_inner, eid_rib)

    # ------------------------------------------------------------------
    # Spanwise elements (spar webs, skin panels, stringers)
    # ------------------------------------------------------------------
    for r, r1 in zip(wing.ribs, wing.ribs[1:]):
        rows = _interp_span_rows(r, r1, n_span_div, model, gid_interp)
        _add_spar_elements(r, r1, model, eid_spar, rows)
        _add_skin_elements(r, r1, model, eid_skin, rows)
        _add_stringer_elements(r, r1, model, eid_stringer, rows)

    # ------------------------------------------------------------------
    # Validate and write
    # ------------------------------------------------------------------
    model.cross_reference()
    model.write_bdf(filename)
