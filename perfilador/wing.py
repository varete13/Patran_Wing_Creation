from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import numpy as np
from scipy.interpolate import interp1d

from .airfoil import NACAAirfoil
from .id_manager import IDManager, IDScheme
from .rib import Rib


@dataclasses.dataclass
class EntitySummary:
    """Count of every Nastran/Patran entity generated for a wing."""

    n_ribs: int
    n_spars: int
    n_stringers: int
    n_skin_points: int

    # Per-rib entities (totals across all ribs)
    grid_points: int
    rib_curves: int
    rib_surfaces: int

    # Spanwise entities (between consecutive ribs)
    spar_lines: int
    stringer_lines: int
    spar_panel_surfaces: int
    skin_panel_surfaces: int

    @property
    def total_lines(self) -> int:
        return self.spar_lines + self.stringer_lines

    @property
    def total_surfaces(self) -> int:
        return self.rib_surfaces + self.spar_panel_surfaces + self.skin_panel_surfaces

    @property
    def total_entities(self) -> int:
        return self.grid_points + self.total_lines + self.total_surfaces

    def __str__(self) -> str:
        pairs = self.n_ribs - 1
        w = 8  # column width
        sep = "-" * 50
        lines = [
            f"{'  Entity summary  ':=^50}",
            f"  Ribs: {self.n_ribs}  |  Spars: {self.n_spars}  |"
            f"  Stringers: {self.n_stringers}  |  Skin pts: {self.n_skin_points}",
            sep,
            f"  {'Grid points':<28} {self.grid_points:>{w},}",
            f"  {'Rib curves (PWL, aux.)':<28} {self.rib_curves:>{w},}",
            f"  {'Rib trimmed surfaces':<28} {self.rib_surfaces:>{w},}",
            sep,
            f"  Spanwise segments ({pairs} bay{'s' if pairs != 1 else ''}):",
            f"  {'Spar lines':<28} {self.spar_lines:>{w},}",
            f"  {'Stringer lines':<28} {self.stringer_lines:>{w},}",
            f"  {'Spar panel surfaces':<28} {self.spar_panel_surfaces:>{w},}",
            f"  {'Skin panel surfaces':<28} {self.skin_panel_surfaces:>{w},}",
            sep,
            f"  {'Total points':<28} {self.grid_points:>{w},}",
            f"  {'Total lines':<28} {self.total_lines:>{w},}",
            f"  {'Total surfaces':<28} {self.total_surfaces:>{w},}",
            sep,
            f"  {'TOTAL entities':<28} {self.total_entities:>{w},}",
        ]
        return "\n".join(lines)


class Wing:
    """Parametric wing defined by spanwise distributions.

    Parameters
    ----------
    airfoil : NACAAirfoil or None
        Single airfoil used at every station.  Mutually exclusive with
        *airfoil_distribution*.
    airfoil_distribution : dict[float, str] or None
        ``{y_station: naca_code, ...}`` — NACA parameters are linearly
        interpolated between stations so the profile transitions smoothly
        along the span.
    wall_thickness : list[float] or dict[float, list[float]] or None
        Wall thickness per section (LE → TE).  A ``list`` applies the same
        thicknesses to every rib; a ``dict`` maps ``y_station`` keys to
        per-section lists and the values are interpolated along the span.
        ``None`` → solid ribs (default).
    span : float
        Semi-span in metres.
    n_ribs : int
        Number of evenly-spaced rib stations from root to tip.
    spar_positions, stringer_positions : list / array
        Chordwise structural positions as fractions of chord (0–1).
    chord_distribution, offset_distribution, twist_distribution,
    elevation_distribution : tuple[list, list]
        ``(y_stations, values)`` pairs fed to ``scipy.interpolate.interp1d``.
    n_skin_points : int
        Discretisation points per airfoil surface side.
    point_spacing : str
        ``"uniform"`` or ``"cosine"``.  Cosine spacing clusters points near
        the leading and trailing edges where curvature is highest.
    min_point_spacing : float or None
        Minimum physical distance (m) between consecutive profile points.
        Points that would be closer than this at the smallest chord station
        are removed.  Set to ``None`` to disable.  Default ``0.005``.
    """

    def __init__(
        self,
        span: float,
        n_ribs: int,
        spar_positions: list[float],
        stringer_positions,
        chord_distribution: tuple[list[float], list[float]],
        offset_distribution: tuple[list[float], list[float]],
        twist_distribution: tuple[list[float], list[float]],
        elevation_distribution: tuple[list[float], list[float]],
        airfoil: NACAAirfoil | None = None,
        airfoil_distribution: dict[float, str] | None = None,
        n_skin_points: int = 100,
        point_spacing: Literal["uniform", "cosine"] = "uniform",
        min_point_spacing: float | None = 0.005,
        wall_thickness: list[float] | dict[float, list[float]] | None = None,
        inner_cuts: list[tuple[int, int]] | dict[int, list[tuple[int, int]]] | None = None,
        cut_wall_thickness: list[float] | dict[float, list[float]] | None = None,
        wall_thickness_kind: Literal["linear", "quadratic", "cubic", "previous", "next"] = "linear",
        cut_wall_thickness_kind: Literal["linear", "quadratic", "cubic", "previous", "next"] = "linear",
    ) -> None:
        if airfoil is None and airfoil_distribution is None:
            raise ValueError("Provide either 'airfoil' or 'airfoil_distribution'")
        if airfoil is not None and airfoil_distribution is not None:
            raise ValueError(
                "'airfoil' and 'airfoil_distribution' are mutually exclusive"
            )

        self.span = span
        self.n_ribs = n_ribs
        self.spar_positions = spar_positions
        self.stringer_positions = np.asarray(stringer_positions)
        self.point_spacing = point_spacing
        self.min_point_spacing = min_point_spacing

        # Wall thickness: list → constant, dict → interpolated along span
        n_sections = len(spar_positions) + 1
        if isinstance(wall_thickness, dict):
            y_stations = sorted(wall_thickness.keys())
            thicknesses = [wall_thickness[y] for y in y_stations]
            if any(len(t) != n_sections for t in thicknesses):
                raise ValueError(
                    f"Each entry in wall_thickness must have "
                    f"{n_sections} values (one per section)"
                )
            self._wt_interps = [
                interp1d(y_stations, [t[i] for t in thicknesses], kind=wall_thickness_kind)
                for i in range(n_sections)
            ]
            self.wall_thickness = None
        else:
            self._wt_interps = None
            self.wall_thickness = wall_thickness

        # Cut wall thickness: list → constant, dict → interpolated along span
        if isinstance(cut_wall_thickness, dict):
            y_stations = sorted(cut_wall_thickness.keys())
            thicknesses = [cut_wall_thickness[y] for y in y_stations]
            if any(len(t) != n_sections for t in thicknesses):
                raise ValueError(
                    f"Each entry in cut_wall_thickness must have "
                    f"{n_sections} values (one per section)"
                )
            self._cwt_interps = [
                interp1d(y_stations, [t[i] for t in thicknesses], kind=cut_wall_thickness_kind)
                for i in range(n_sections)
            ]
            self.cut_wall_thickness = None
        else:
            self._cwt_interps = None
            self.cut_wall_thickness = cut_wall_thickness

        # Airfoil distribution along the span
        if airfoil is not None:
            self._airfoil_const = airfoil
            self._airfoil_interp = None
        else:
            self._airfoil_const = None
            y_stations = sorted(airfoil_distribution.keys())
            codes = [airfoil_distribution[y] for y in y_stations]
            profiles = [NACAAirfoil(c) for c in codes]
            self._airfoil_interp = {
                "m": interp1d(y_stations, [p.max_camber for p in profiles]),
                "p": interp1d(y_stations, [p.camber_position for p in profiles]),
                "t": interp1d(y_stations, [p.thickness for p in profiles]),
            }

        # Spanwise interpolation functions (instance-owned, not class-level)
        self.chord_at = interp1d(*chord_distribution)
        self.offset_at = interp1d(*offset_distribution)
        self.twist_at = interp1d(*twist_distribution)
        self.elevation_at = interp1d(*elevation_distribution)

        # Build chordwise stations and enforce Patran min-spacing
        x_stations = self._build_x_stations(n_skin_points, point_spacing)
        if min_point_spacing is not None:
            rib_ys = np.linspace(0, span, n_ribs)
            min_chord = float(np.min([self.chord_at(y) for y in rib_ys]))
            x_stations = self._filter_min_spacing(
                x_stations, min_chord, min_point_spacing
            )
        self._x_stations = x_stations
        self.n_skin_points = len(x_stations)

        # ID allocation (uses effective n_skin_points after filtering)
        has_inner = (
            wall_thickness is not None
            or cut_wall_thickness is not None
        )
        # Normalise inner_cuts to a per-rib dict of local index pairs
        if isinstance(inner_cuts, list):
            _ic: dict[int, list[tuple[int, int]]] | None = {
                i: list(inner_cuts) for i in range(n_ribs)
            }
        elif inner_cuts is not None:
            _ic = inner_cuts
        else:
            _ic = None

        n_cuts_per_rib = (
            len(inner_cuts) if isinstance(inner_cuts, list)
            else max(len(v) for v in inner_cuts.values()) if inner_cuts
            else 0
        )
        self._id_manager = IDManager(
            n_ribs, self.n_skin_points, len(spar_positions),
            has_inner=has_inner,
            n_cuts_per_rib=n_cuts_per_rib,
        )

        # Build ribs
        self.ribs: list[Rib] = []
        for i, y in enumerate(np.linspace(0, span, n_ribs)):
            rib_ids = self._id_manager.allocate_rib()

            cut_pairs_local: list[tuple[int, int]] | None = (
                list(_ic[i]) if _ic and i in _ic else None
            )

            self.ribs.append(
                Rib(
                    span_position=float(y),
                    airfoil=self.airfoil_at(float(y)),
                    chord=float(self.chord_at(y)),
                    offset=float(self.offset_at(y)),
                    twist=float(self.twist_at(y)),
                    elevation=float(self.elevation_at(y)),
                    spar_positions=spar_positions,
                    stringer_positions=self.stringer_positions,
                    ids=rib_ids,
                    x_stations=x_stations,
                    wall_thickness=self.wall_thickness_at(float(y)),
                    cut_pairs=cut_pairs_local,
                    cut_wall_thickness=self.cut_wall_thickness_at(float(y)),
                )
            )

        logging.debug("Wing entity summary:\n%s", self.summary())

    @staticmethod
    def _build_x_stations(n_points: int, spacing: str) -> np.ndarray:
        if spacing == "cosine":
            return 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_points)))
        if spacing == "uniform":
            return np.linspace(0, 1, n_points)
        raise ValueError(f"Unknown point_spacing {spacing!r}")

    @staticmethod
    def _filter_min_spacing(
        x_stations: np.ndarray, min_chord: float, min_spacing: float = 0.005
    ) -> np.ndarray:
        """Remove stations whose physical spacing would be < *min_spacing*."""
        min_dx = min_spacing / min_chord  # normalized threshold
        kept = [x_stations[0]]
        for x in x_stations[1:]:
            if x - kept[-1] >= min_dx:
                kept.append(x)
        # Always keep trailing edge
        if kept[-1] < x_stations[-1]:
            if x_stations[-1] - kept[-1] < min_dx:
                kept[-1] = x_stations[-1]
            else:
                kept.append(x_stations[-1])
        return np.array(kept)

    def wall_thickness_at(self, y: float) -> list[float] | None:
        """Return the wall thickness per section at spanwise station *y*.

        Returns ``None`` for solid ribs, a constant list if a ``list`` was
        provided for *wall_thickness*, or an interpolated list if a ``dict``
        was provided.
        """
        if self._wt_interps is not None:
            return [float(interp(y)) for interp in self._wt_interps]
        return self.wall_thickness  # may be None (solid) or a constant list

    def cut_wall_thickness_at(self, y: float) -> list[float] | None:
        """Return the cut wall thickness per section at spanwise station *y*.

        Returns ``None`` for solid cut sections, a constant list if a ``list``
        was provided for *cut_wall_thickness*, or an interpolated list if a
        ``dict`` was provided.
        """
        if self._cwt_interps is not None:
            return [float(interp(y)) for interp in self._cwt_interps]
        return self.cut_wall_thickness  # may be None or a constant list

    def airfoil_at(self, y: float) -> NACAAirfoil:
        """Return the airfoil at spanwise station *y*.

        If a single airfoil was provided, it is returned for every station.
        If an airfoil distribution was provided, the NACA parameters are
        interpolated and a new ``NACAAirfoil`` is created.
        """
        if self._airfoil_const is not None:
            return self._airfoil_const
        interp = self._airfoil_interp
        return NACAAirfoil.from_params(
            max_camber=float(interp["m"](y)),
            camber_position=float(interp["p"](y)),
            thickness=float(interp["t"](y)),
        )

    @property
    def id_scheme(self) -> IDScheme:
        return self._id_manager.scheme

    @staticmethod
    def max_skin_points(
        n_ribs: int,
        n_spars: int,
        n_stringers: int,
        max_entities: int = 10_000,
        min_skin_points: int = 10,
    ) -> int:
        """Compute the largest ``n_skin_points`` that stays within a budget.

        The total entity count is linear in *n_skin_points*::

            total = (2*n_sp - 1)*(n_ribs + n_ribs-1)  [grids + skin panels]
                  + fixed overhead                      [curves, rib surfs,
                                                         spar/stringer lines,
                                                         spar panels]

        This method solves for ``n_sp`` analytically.

        Parameters
        ----------
        n_ribs, n_spars, n_stringers : int
            Structural layout (these are fixed design choices).
        max_entities : int
            Upper bound on total entities (default 10 000).
        min_skin_points : int
            Returned value is clamped to at least this (default 10).

        Returns
        -------
        int
            Recommended ``n_skin_points``.
        """
        R = n_ribs
        P = R - 1  # consecutive-rib pairs

        # Fixed overhead (independent of n_skin_points)
        fixed = (
            (n_spars + 1) * R               # rib trimmed surfaces (curves are aux.)
            + 2 * n_spars * P                # spar lines
            + 2 * n_stringers * P            # stringer lines
            + n_spars * P                    # spar panel surfaces
        )

        # n_sp-dependent coefficient: total_variable = (2*n_sp - 1) * coeff
        coeff = R + P  # grids contribute R, skin panels contribute P

        budget = max_entities - fixed
        if budget <= 0:
            return min_skin_points

        # (2*n_sp - 1) <= budget / coeff  =>  n_sp <= (budget/coeff + 1) / 2
        n_sp = int((budget / coeff + 1) / 2)
        return max(n_sp, min_skin_points)

    @staticmethod
    def section_map_plot(
        airfoil: NACAAirfoil,
        spar_positions: list[float],
        stringer_positions,
        n_skin_points: int = 100,
        point_spacing: Literal["uniform", "cosine"] = "cosine",
        ax=None,
    ):
        """Plot the airfoil profile with section colour-coding and local index labels.

        Builds a flat reference rib (chord=1, no twist/sweep/dihedral) and
        shows which local profile indices belong to each standard section
        (LE, Box, TE).  Use this to choose ``inner_cuts`` index pairs without
        having to build a full :class:`Wing`.

        Parameters
        ----------
        airfoil : NACAAirfoil
        spar_positions : list[float]
            Chordwise spar locations (0–1).
        stringer_positions : array-like
            Chordwise stringer locations (0–1).
        n_skin_points : int
            Points per surface side (default 100).
        point_spacing : "uniform" or "cosine"
        ax : matplotlib Axes or None
            If *None* a new figure is created and returned.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # Build a flat reference rib (chord=1, no twist/offset/elevation)
        _id_mgr = IDManager(1, n_skin_points, len(spar_positions))
        _rib_ids = _id_mgr.allocate_rib()
        rib = Rib(
            span_position=0.0,
            airfoil=airfoil,
            chord=1.0,
            offset=0.0,
            twist=0.0,
            elevation=0.0,
            spar_positions=spar_positions,
            stringer_positions=np.asarray(stringer_positions),
            ids=_rib_ids,
            n_skin_points=n_skin_points,
            point_spacing=point_spacing,
        )

        px, py = rib.profile
        if np.allclose(px[0], px[-1]) and np.allclose(py[0], py[-1]):
            px, py = px[:-1], py[:-1]
        n = len(px)

        # Section boundary indices (mirrors _apply_cuts logic)
        n_spars = len(spar_positions)
        ps = rib.ids.point_start
        upper_idx = [sid - ps for sid in rib.spar_ids[:n_spars]]
        lower_idx = [sid - ps for sid in rib.spar_ids[n_spars:]]

        le_idx = list(range(0, upper_idx[0] + 1)) + list(range(lower_idx[-1], n))
        box_idx = [
            list(range(upper_idx[j], upper_idx[j + 1] + 1))
            + list(range(lower_idx[n_spars - 2 - j], lower_idx[n_spars - 1 - j] + 1))
            for j in range(n_spars - 1)
        ]
        te_idx = list(range(upper_idx[-1], lower_idx[0] + 1))

        if n_spars == 1:
            sec_names = ["LE", "TE"]
            sec_indices = [le_idx, te_idx]
        elif n_spars == 2:
            sec_names = ["LE", "Box", "TE"]
            sec_indices = [le_idx, box_idx[0], te_idx]
        else:
            sec_names = ["LE"] + [f"Box{k + 1}" for k in range(n_spars - 1)] + ["TE"]
            sec_indices = [le_idx] + box_idx + [te_idx]

        sec_colors = ["#2196F3", "#4CAF50", "#FF5722", "#9C27B0", "#FF9800"]

        if ax is None:
            fig, ax = plt.subplots(figsize=(14, 5))
        else:
            fig = ax.figure

        # Thin profile outline
        ax.plot(
            np.append(px, px[0]), np.append(py, py[0]),
            color="lightgray", lw=0.8, zorder=1,
        )

        # Scatter and annotate each section
        for sec_name, idx_list, color in zip(sec_names, sec_indices, sec_colors):
            xs = px[np.array(idx_list)]
            ys = py[np.array(idx_list)]
            ax.scatter(xs, ys, color=color, s=25, zorder=3)
            for i in idx_list:
                ax.annotate(
                    str(i), (px[i], py[i]),
                    fontsize=6, color=color,
                    ha="center", va="bottom",
                    xytext=(0, 4), textcoords="offset points",
                )

        # Spar vertical reference lines
        for sp_x in spar_positions:
            ax.axvline(sp_x, color="dimgray", ls="--", lw=0.8, alpha=0.6)

        # Stringer vertical reference lines
        str_arr = np.asarray(stringer_positions)
        if str_arr.size:
            for st_x in str_arr:
                ax.axvline(st_x, color="darkorange", ls=":", lw=0.8, alpha=0.6)

        # Legend
        legend_handles = [
            Line2D(
                [0], [0], marker="o", color="w",
                markerfacecolor=c, markersize=7, label=name,
            )
            for name, c in zip(sec_names, sec_colors)
        ]
        legend_handles.append(
            Line2D([0], [0], color="dimgray", ls="--", lw=0.8, label="Spar")
        )
        if str_arr.size:
            legend_handles.append(
                Line2D([0], [0], color="darkorange", ls=":", lw=0.8, label="Stringer")
            )
        ax.legend(handles=legend_handles, loc="upper right", fontsize=8)

        designation = getattr(airfoil, "designation", "")
        ax.set_aspect("equal")
        ax.set_xlabel("x/c")
        ax.set_ylabel("y/c")
        ax.set_title(
            f"Section map — NACA {designation} | {n_spars} spar(s) | "
            f"{n_skin_points} pts ({point_spacing})"
        )
        ax.grid(True, alpha=0.3)
        return fig

    def summary(self) -> EntitySummary:
        """Count all Nastran/Patran entities that would be generated."""
        n = self.n_ribs
        n_sp = self.n_skin_points
        n_spars = len(self.spar_positions)
        n_stringers = len(self.stringer_positions)
        pairs = n - 1  # consecutive-rib pairs

        # Per rib: 2*n_sp - 1 grid points, n_spars+1 curves & trimmed surfaces
        grids_per_rib = 2 * n_sp - 1
        curves_per_rib = n_spars + 1
        surfaces_per_rib = n_spars + 1

        # Between consecutive ribs
        structural_pts = (n_spars + n_stringers) * 2  # spar+stringer ids per rib
        spar_lines_per_pair = n_spars * 2
        stringer_lines_per_pair = n_stringers * 2
        spar_panels_per_pair = n_spars
        skin_panels_per_pair = 2 * n_sp - 1

        return EntitySummary(
            n_ribs=n,
            n_spars=n_spars,
            n_stringers=n_stringers,
            n_skin_points=n_sp,
            grid_points=grids_per_rib * n,
            rib_curves=curves_per_rib * n,
            rib_surfaces=surfaces_per_rib * n,
            spar_lines=spar_lines_per_pair * pairs,
            stringer_lines=stringer_lines_per_pair * pairs,
            spar_panel_surfaces=spar_panels_per_pair * pairs,
            skin_panel_surfaces=skin_panels_per_pair * pairs,
        )
