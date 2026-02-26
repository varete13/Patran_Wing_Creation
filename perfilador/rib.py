from __future__ import annotations

import numpy as np
from typing import Literal

from .airfoil import NACAAirfoil
from .geometry import find_nearest_point, rotate_2d, transform_airfoil_points
from .geometry import hollow_rib_sections, offset_polygon
from .id_manager import RibIDs


class Rib:
    """A wing cross-section at a given spanwise station.

    All transform parameters (chord, offset, twist, elevation) are received
    as plain values—no reaching back into a parent Wing class.
    Geometry is computed once at construction and cached.

    Parameters
    ----------
    span_position : float
        Distance from root along the span (m).
    airfoil : NACAAirfoil
        Airfoil geometry generator.
    chord, offset, twist, elevation : float
        Local chord length, LE offset, twist (rad) and dihedral elevation.
    spar_positions : list[float]
        Chordwise spar locations (0–1 fraction of chord).
    stringer_positions : array_like
        Chordwise stringer locations (0–1 fraction of chord).
    ids : RibIDs
        Pre-allocated ID ranges from :class:`IDManager`.
    n_skin_points : int
        Number of discretisation points per surface side (default 100).
        Ignored when *x_stations* is provided.
    point_spacing : str
        ``"uniform"`` for evenly-spaced points, ``"cosine"`` for a half-cosine
        distribution that clusters points near the leading edge (x = 0) and
        trailing edge (x = 1).  Default ``"uniform"``.
        Ignored when *x_stations* is provided.
    x_stations : np.ndarray or None
        Explicit chordwise stations (0–1).  When provided, *n_skin_points*
        and *point_spacing* are ignored and ``n_skin_points`` is set to
        ``len(x_stations)``.
    """

    def __init__(
        self,
        span_position: float,
        airfoil: NACAAirfoil,
        chord: float,
        offset: float,
        twist: float,
        elevation: float,
        spar_positions: list[float],
        stringer_positions: np.ndarray,
        ids: RibIDs,
        n_skin_points: int = 100,
        point_spacing: Literal["uniform", "cosine"] = "uniform",
        x_stations: np.ndarray | None = None,
        wall_thickness: list[float] | None = None,
        cut_pairs: list[tuple[int, int]] | None = None,
        cut_wall_thickness: list[float] | None = None,
    ) -> None:
        self.span_position = span_position
        self.airfoil = airfoil
        self.chord = chord
        self.offset = offset
        self.twist = twist
        self.elevation = elevation
        self.spar_positions = list(spar_positions)
        self.stringer_positions = np.asarray(stringer_positions)
        self.ids = ids
        self.point_spacing = point_spacing

        # Build chordwise stations
        if x_stations is not None:
            xs = np.asarray(x_stations)
        elif point_spacing == "cosine":
            xs = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_skin_points)))
        elif point_spacing == "uniform":
            xs = np.linspace(0, 1, n_skin_points)
        else:
            raise ValueError(f"Unknown point_spacing {point_spacing!r}")

        self.n_skin_points = len(xs)

        # Computed and cached geometry
        self._profile: tuple[np.ndarray, np.ndarray] = transform_airfoil_points(
            airfoil, xs, twist, chord, offset, elevation, skip_lower_te=True
        )
        self._spar_vertices: tuple[np.ndarray, np.ndarray] = transform_airfoil_points(
            airfoil, np.asarray(spar_positions), twist, chord, offset, elevation
        )
        self._stringer_vertices: tuple[np.ndarray, np.ndarray] = transform_airfoil_points(
            airfoil, self.stringer_positions, twist, chord, offset, elevation
        )

        # Snap structural vertices to the nearest profile grid points
        self._spar_ids = self._snap_to_profile(self._spar_vertices)
        self._stringer_ids = self._snap_to_profile(self._stringer_vertices)

        # Hollow rib inner cavities (one wall thickness per section)
        self.wall_thickness = wall_thickness
        if wall_thickness is not None:
            n_sections = len(spar_positions) + 1
            if len(wall_thickness) != n_sections:
                raise ValueError(
                    f"wall_thickness must have {n_sections} values "
                    f"(one per section), got {len(wall_thickness)}"
                )
            self._inner_profiles = hollow_rib_sections(self, wall_thickness)
        else:
            self._inner_profiles = None

        # Inner cuts: split sections into sub-sections
        self._cut_sections: dict[str, list[list[int]]] | None = (
            self._apply_cuts(cut_pairs) if cut_pairs else None
        )

        # Inner contours for each cut sub-section (offset_polygon applied)
        if cut_wall_thickness is not None:
            n_sections = len(spar_positions) + 1
            if len(cut_wall_thickness) != n_sections:
                raise ValueError(
                    f"cut_wall_thickness must have {n_sections} values "
                    f"(one per section), got {len(cut_wall_thickness)}"
                )
        self._cut_inner_profiles: list[tuple[np.ndarray, np.ndarray] | None] | None = (
            self._compute_cut_inner_profiles(cut_wall_thickness)
            if self._cut_sections is not None and cut_wall_thickness is not None
            else None
        )

    # ------------------------------------------------------------------
    # Cached geometry properties
    # ------------------------------------------------------------------

    @property
    def profile(self) -> tuple[np.ndarray, np.ndarray]:
        """Full airfoil contour (upper + reversed-lower)."""
        return self._profile

    @property
    def spar_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        """Spar endpoint coordinates (upper + reversed-lower)."""
        return self._spar_vertices

    @property
    def stringer_vertices(self) -> tuple[np.ndarray, np.ndarray]:
        """Stringer endpoint coordinates (upper + reversed-lower)."""
        return self._stringer_vertices

    @property
    def spar_ids(self) -> list[int]:
        """Nastran point IDs snapped to the nearest profile grid point."""
        return self._spar_ids

    @property
    def stringer_ids(self) -> list[int]:
        return self._stringer_ids

    @property
    def inner_profiles(self) -> list[tuple[np.ndarray, np.ndarray] | None] | None:
        """Inner cavity contours per section, or ``None`` for a solid rib.

        Individual entries may be ``None`` if a section is too thin
        for its wall thickness."""
        return self._inner_profiles

    # ------------------------------------------------------------------
    # Cross-section properties
    # ------------------------------------------------------------------

    @property
    def spar_box_area(self) -> float:
        """Trapezoidal spar-box cross-section area."""
        n = len(self.spar_positions)
        areas = 0.0
        for i in range(n - 1):
            w = np.diff(self.spar_positions)[i]
            h1 = self._spar_vertices[1][i] - self._spar_vertices[1][i + n]
            h2 = self._spar_vertices[1][i + 1] - self._spar_vertices[1][i + 1 + n]
            areas += (h1 + h2) * w / 2.0
        return float(areas)

    @property
    def profile_area(self) -> float:
        """Full airfoil cross-section area at this station."""
        return self.airfoil.area * self.chord ** 2

    @property
    def profile_cg(self) -> np.ndarray:
        """CG position in global coordinates ``[x, y]``."""
        local_cg = self.airfoil.cg * self.chord
        rx, ry = rotate_2d(local_cg[0], local_cg[1], self.twist)
        return np.array([float(rx) + self.offset, float(ry) + self.elevation])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def cut_sections(self) -> dict[str, list[list[int]]] | None:
        """Sub-sections resulting from inner cuts, or ``None`` if none defined.

        Returns a dict keyed by section name (``"LE"``, ``"Box"``, ``"TE"``,
        etc.) where each value is a list of sub-sections.  Each sub-section is
        a list of **local profile indices** (0-based from ``ids.point_start``).
        """
        return self._cut_sections

    @property
    def cut_inner_profiles(
        self,
    ) -> list[tuple[np.ndarray, np.ndarray] | None] | None:
        """Inner contour for each cut sub-section, or ``None`` if not requested.

        One entry per sub-section, in the same order as iterating
        ``cut_sections`` values.  An entry is ``None`` when ``offset_polygon``
        could not produce a valid inner contour (section too thin).
        Each non-``None`` entry is a closed ``(x, y)`` array pair (last point
        equals first).
        """
        return self._cut_inner_profiles

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snap_to_profile(
        self, vertices: tuple[np.ndarray, np.ndarray]
    ) -> list[int]:
        """Map each vertex to the nearest profile grid-point ID."""
        px, py = self._profile
        ids: list[int] = []
        for vx, vy in zip(vertices[0], vertices[1]):
            idx, _, _ = find_nearest_point(px, py, float(vx), float(vy))
            ids.append(idx + self.ids.point_start)
        return ids

    def _compute_cut_inner_profiles(
        self, wall_thicknesses: list[float]
    ) -> list[tuple[np.ndarray, np.ndarray] | None]:
        """Apply ``offset_polygon`` to every cut sub-section.

        *wall_thicknesses* has one value per standard section (LE, Box…, TE),
        in the same order as ``cut_sections``.

        Returns a flat list with one entry per sub-section (iterating
        ``cut_sections`` values in order).  Each entry is either a closed
        ``(x, y)`` pair or ``None`` when the section is too thin.
        """
        assert self._cut_sections is not None
        px, py = self.profile
        if np.allclose(px[0], px[-1]) and np.allclose(py[0], py[-1]):
            px, py = px[:-1], py[:-1]
        pts = np.column_stack([px, py])

        result: list[tuple[np.ndarray, np.ndarray] | None] = []
        for sec_idx, subsections in enumerate(self._cut_sections.values()):
            wt = wall_thicknesses[sec_idx]
            for sub_local in subsections:
                # Shapely requires at least 4 unique points for a valid ring
                if len(sub_local) < 4:
                    result.append(None)
                    continue
                coords_arr = pts[sub_local]
                coords_closed = np.vstack([coords_arr, coords_arr[0:1]])
                inner = offset_polygon(coords_closed, wt)
                if inner is not None:
                    result.append((inner[:, 0], inner[:, 1]))
                else:
                    result.append(None)
        return result

    def _apply_cuts(
        self, pairs_local: list[tuple[int, int]]
    ) -> dict[str, list[list[int]]]:
        """Split rib sections into sub-sections at the given cut pairs.

        Parameters
        ----------
        pairs_local : list of (idx_a, idx_b)
            Pairs as **local profile indices** (0-based from
            ``ids.point_start``).  Both indices of a pair must belong to the
            same section; cross-section pairs are silently ignored.

        Returns
        -------
        dict
            ``{section_name: [[local_idx, ...], ...]}`` — one list of
            sub-sections per section.  Sections without cuts contain a single
            sub-section equal to the original section indices.
        """
        px, py = self.profile
        # Drop the closing duplicate point if present
        if np.allclose(px[0], px[-1]) and np.allclose(py[0], py[-1]):
            px, py = px[:-1], py[:-1]
        n = len(px)

        n_spars = len(self.spar_positions)
        # Upper spar local indices: spar_1 → spar_N (LE→TE direction)
        upper_idx = [sid - self.ids.point_start for sid in self.spar_ids[:n_spars]]
        # Lower spar local indices: spar_N → spar_1 (TE→LE in profile array)
        lower_idx = [sid - self.ids.point_start for sid in self.spar_ids[n_spars:]]

        # Build closed index lists per section
        section_indices: list[list[int]] = []
        # LE: profile start → upper_s1, lower_s1 → profile end
        section_indices.append(
            list(range(0, upper_idx[0] + 1)) + list(range(lower_idx[-1], n))
        )
        # Box sections (one per consecutive spar pair)
        for j in range(n_spars - 1):
            section_indices.append(
                list(range(upper_idx[j], upper_idx[j + 1] + 1))
                + list(range(lower_idx[n_spars - 2 - j], lower_idx[n_spars - 1 - j] + 1))
            )
        # TE: upper_sN → lower_sN (consecutive in profile array)
        section_indices.append(list(range(upper_idx[-1], lower_idx[0] + 1)))

        # Section names
        if n_spars == 1:
            sec_names = ["LE", "TE"]
        elif n_spars == 2:
            sec_names = ["LE", "Box", "TE"]
        else:
            sec_names = ["LE"] + [f"Box{k + 1}" for k in range(n_spars - 1)] + ["TE"]

        # Map each local index to its section
        assigned: dict[int, int] = {}
        for s, pts in enumerate(section_indices):
            for k in pts:
                if k not in assigned:
                    assigned[k] = s

        # Start with one sub-section per section (the full original indices)
        pools: dict[int, list[list[int]]] = {
            s: [list(idx)] for s, idx in enumerate(section_indices)
        }

        n_profile = len(px)

        # Apply each cut pair
        for ia, ib in pairs_local:
            if ia == ib:
                continue  # degenerate pair — no cut needed
            sec_a = assigned.get(ia)
            sec_b = assigned.get(ib)
            if sec_a is None or sec_b is None:
                bad = ia if sec_a is None else ib
                raise ValueError(
                    f"inner_cuts: local index {bad} is out of range "
                    f"[0, {n_profile - 1}] for this rib (y={self.span_position:.3f}m)."
                )
            if sec_a != sec_b:
                raise ValueError(
                    f"inner_cuts: indices ({ia}, {ib}) belong to different sections "
                    f"('{sec_names[sec_a]}' and '{sec_names[sec_b]}') "
                    f"at rib y={self.span_position:.3f}m. "
                    f"Both indices of a cut pair must lie in the same section."
                )
            pool = pools[sec_a]
            for i, sub in enumerate(pool):
                if ia in sub and ib in sub:
                    pos_a, pos_b = sub.index(ia), sub.index(ib)
                    if pos_a > pos_b:
                        pos_a, pos_b = pos_b, pos_a
                    pool[i] = sub[pos_a: pos_b + 1]
                    pool.insert(i + 1, sub[pos_b:] + sub[: pos_a + 1])
                    break

        return {
            sec_names[s]: sorted(pool, key=lambda sub: min(sub))
            for s, pool in pools.items()
        }
