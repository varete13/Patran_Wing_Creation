from __future__ import annotations

import logging

import numpy as np
from shapely.geometry import Polygon

from .airfoil import NACAAirfoil
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .rib import Rib


def rotate_2d(
    x: np.ndarray, y: np.ndarray, angle: float
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate 2-D coordinates by *angle* radians (positive = counter-clockwise).

    Works element-wise on arrays without ``@np.vectorize``.
    """
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    return x * cos_a - y * sin_a, x * sin_a + y * cos_a


def transform_airfoil_points(
    airfoil: NACAAirfoil,
    x_stations: np.ndarray,
    twist: float,
    chord: float,
    offset: float,
    elevation: float,
    skip_lower_te: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate *airfoil* at *x_stations*, then rotate / scale / translate.

    Returns concatenated upper + reversed-lower coordinate arrays
    ``(all_x, all_y)`` that trace the full profile contour.

    Parameters
    ----------
    skip_lower_te : bool
        When ``True`` the lower-surface trailing-edge point (which duplicates
        the upper-surface TE in thin airfoils at small chords) is omitted.
        This yields ``2*n - 1`` profile points with a single shared TE,
        avoiding Patran's duplicate-point warning.  Use ``True`` for the
        outer profile; keep ``False`` for spar/stringer vertex evaluation
        where the last station is not necessarily the TE.
    """
    xu, yu, xl, yl = airfoil.evaluate(x_stations)

    xu, yu = rotate_2d(xu, yu, twist)
    xl, yl = rotate_2d(xl, yl, twist)

    xu = xu * chord + offset
    xl = xl * chord + offset
    yu = yu * chord + elevation
    yl = yl * chord + elevation

    if skip_lower_te:
        # Omit xl[-1]/yl[-1] (TE lower): for thin airfoils at small chords
        # it falls within Patran's 0.005 m resolution of the upper TE.
        all_x = np.concatenate([xu, xl[-2::-1]])
        all_y = np.concatenate([yu, yl[-2::-1]])
    else:
        all_x = np.concatenate([xu, xl[::-1]])
        all_y = np.concatenate([yu, yl[::-1]])
    return all_x, all_y


def find_nearest_point(
    profile_x: np.ndarray,
    profile_y: np.ndarray,
    target_x: float,
    target_y: float,
) -> tuple[int, float, float]:
    """Index and coords of the profile point closest to *(target_x, target_y)*.

    Fully vectorised (single ``np.argmin`` call).
    """
    dist_sq = (profile_x - target_x) ** 2 + (profile_y - target_y) ** 2
    idx = int(np.argmin(dist_sq))
    return idx, float(profile_x[idx]), float(profile_y[idx])


def offset_polygon(polygon: np.ndarray, distance: float) -> np.ndarray | None:
    """Compute the inward offset of a closed polygon using Shapely.

    Parameters
    ----------
    polygon : (N, 2) array
        Closed polygon vertices (first == last point).
    distance : float
        Offset distance (positive = inward).

    Returns
    -------
    offset : (M, 2) array or None
        Offset polygon (closed), or ``None`` if the section is too thin
        for the requested offset (caller should treat it as solid).
    """
    if distance <= 0:
        return polygon.copy()

    # Shapely expects the closing vertex to NOT be duplicated
    coords = polygon[:-1].tolist()
    poly = Polygon(coords)

    # Morphological opening: expand → large contract → expand.
    # Net offset = -distance, but the intermediate large contraction removes
    # spikes.  If the intermediate step collapses (section too thin for 4·d),
    # the result ends up at or larger than the original — return None so the
    # caller keeps the section solid.
    buffered = poly.buffer(distance).buffer(-3 * distance).buffer(distance)

    if buffered.is_empty or buffered.area >= poly.area * 0.9:
        return None

    # If the buffer produces a MultiPolygon, keep the largest piece
    if buffered.geom_type == "MultiPolygon":
        buffered = max(buffered.geoms, key=lambda g: g.area)

    # Simplify to remove points closer than Patran's default geometric
    # resolution (0.005) — avoids "duplicate point" warnings.
    buffered = buffered.simplify(5e-3, preserve_topology=True)

    coords_out = np.array(buffered.exterior.coords)
    return coords_out


def extract_sections(rib: Rib) -> list[np.ndarray]:
    """Split a rib profile into closed section polygons at spar positions.

    Each section is a closed polygon (N x 2 array, first == last point)
    bounded by the airfoil skin on top/bottom and spar webs on the sides.

    The profile layout is ``upper[0..n_sp-1] + lower_reversed[0..n_sp-1]``:

    * Indices ``0`` to ``n_sp - 1``: upper surface, LE → TE.
    * Indices ``n_sp`` to ``2*n_sp - 2``: lower surface reversed, TE → LE.

    ``spar_ids`` has ``2 * n_spars`` entries:

    * ``spar_ids[0..n_spars-1]``: upper surface spars, ordered spar1→sparN
      (increasing profile index, LE → TE direction).
    * ``spar_ids[n_spars..2*n_spars-1]``: lower surface spars, ordered
      sparN→spar1 (increasing profile index, TE → LE direction).

    Returns
    -------
    sections : list of (N_i, 2) arrays
        One closed polygon per section, ordered from LE to TE.
        For *S* spars there are *S + 1* sections.
    """
    profile_x, profile_y = rib.profile
    n_sp = rib.n_skin_points
    n_spars = len(rib.spar_positions)

    # Convert spar IDs to profile-array indices
    id_start = rib.ids.point_start
    spar_ids = rib.spar_ids
    # Upper spar indices: ordered spar1..sparN (LE→TE)
    upper_idx = [spar_ids[i] - id_start for i in range(n_spars)]
    # Lower spar indices as stored: ordered sparN..spar1 (TE→LE in profile)
    lower_idx_raw = [spar_ids[n_spars + i] - id_start for i in range(n_spars)]
    # Reverse to get spar1..sparN order (but in profile-array index space,
    # spar1_lower > sparN_lower because lower surface runs TE→LE)
    lower_idx = list(reversed(lower_idx_raw))
    # Now lower_idx[k] is the lower-surface profile index for spar k+1,
    # and lower_idx[0] > lower_idx[1] > ... (closer to LE = higher index).

    sections: list[np.ndarray] = []

    # Upper boundary indices for each section: [0, spar1_u, spar2_u, ..., n_sp-1]
    upper_bounds = [0] + upper_idx + [n_sp - 1]
    # Lower boundary indices: [last_profile_idx, spar1_l, spar2_l, ..., n_sp]
    # lower surface goes TE→LE, so index n_sp is TE end, and last index is LE end.
    # spar closer to LE has HIGHER index.  We want bounds in section order
    # (LE→TE), so for lower: [spar1_l (high idx), spar2_l, ..., n_sp (TE start)]
    lower_bounds = lower_idx + [n_sp]
    # Add the LE end of the lower surface as the start for the LE section
    last_lower_idx = len(profile_x) - 1
    lower_bounds = [last_lower_idx] + lower_bounds

    for sec_idx in range(n_spars + 1):
        # Upper surface: left→right (increasing profile index)
        u_start = upper_bounds[sec_idx]
        u_end = upper_bounds[sec_idx + 1]
        seg_upper_x = profile_x[u_start: u_end + 1]
        seg_upper_y = profile_y[u_start: u_end + 1]

        # Lower surface: right→left (increasing profile index = toward LE)
        # For section k, lower goes from lower_bounds[k+1] to lower_bounds[k]
        l_from = lower_bounds[sec_idx + 1]
        l_to = lower_bounds[sec_idx]
        seg_lower_x = profile_x[l_from: l_to + 1]
        seg_lower_y = profile_y[l_from: l_to + 1]

        # Build closed polygon:
        # upper (LE→TE direction) + lower (TE→LE direction) → CCW winding
        sec_x = np.concatenate([seg_upper_x, seg_lower_x])
        sec_y = np.concatenate([seg_upper_y, seg_lower_y])
        # Close
        sec_x = np.append(sec_x, sec_x[0])
        sec_y = np.append(sec_y, sec_y[0])

        sections.append(np.column_stack([sec_x, sec_y]))

    return sections


def _min_dimension(polygon: np.ndarray) -> float:
    """Return the minimum dimension of a closed polygon's bounding rectangle.

    Uses Shapely's ``minimum_rotated_rectangle`` to find the tightest
    enclosing rectangle and returns the length of its shorter side.
    """
    poly = Polygon(polygon[:-1])
    mrr = poly.minimum_rotated_rectangle
    coords = np.array(mrr.exterior.coords)
    side_a = np.linalg.norm(coords[1] - coords[0])
    side_b = np.linalg.norm(coords[2] - coords[1])
    return min(side_a, side_b)


def hollow_rib_sections(
    rib: Rib, wall_thickness: list[float]
) -> list[tuple[np.ndarray, np.ndarray] | None]:
    """Compute inner cavity contours for each section of a rib.

    Parameters
    ----------
    rib : Rib
        The rib whose sections to hollow out.
    wall_thickness : list[float]
        Inward offset distance (m) per section, ordered LE to TE.
        Must have ``len(spar_positions) + 1`` elements.

    Returns
    -------
    inner_profiles : list of (x_array, y_array) tuples or None
        One inner contour per section, ordered LE to TE.
        Each contour is a closed polygon.  ``None`` entries indicate
        sections that remain solid because the section is too thin
        for the requested wall thickness.
    """
    sections = extract_sections(rib)
    inner_profiles: list[tuple[np.ndarray, np.ndarray] | None] = []

    for section, thickness in zip(sections, wall_thickness):
        min_dim = _min_dimension(section)
        if min_dim < thickness:
            logging.debug(
                "Section minimum dimension (%.4fm) is smaller than wall "
                "thickness (%.4fm) at y=%.2fm. Section remains solid.",
                min_dim, thickness, rib.span_position,
            )
            inner_profiles.append(None)
        else:
            inner = offset_polygon(section, thickness)
            if inner is None:
                logging.debug(
                    "Morphological offset collapsed section (too thin for "
                    "%.4fm wall) at y=%.2fm. Section remains solid.",
                    thickness, rib.span_position,
                )
                inner_profiles.append(None)
            else:
                inner_profiles.append((inner[:, 0], inner[:, 1]))

    return inner_profiles
