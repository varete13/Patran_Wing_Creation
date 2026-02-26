from __future__ import annotations

import numpy as np

from .airfoil import NACAAirfoil
from .geometry import find_nearest_point, rotate_2d, transform_airfoil_points
from .geometry import hollow_rib_sections
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
        point_spacing: str = "uniform",
        x_stations: np.ndarray | None = None,
        wall_thickness: list[float] | None = None,
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
