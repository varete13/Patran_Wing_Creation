"""Matplotlib visualisation helpers for wings and ribs."""
from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D

    from .rib import Rib
    from .wing import Wing


def _default_colors() -> list[str]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


# ------------------------------------------------------------------
# 2-D rib plots
# ------------------------------------------------------------------


def plot_rib(
    rib: Rib, ax: plt.Axes | None = None, show_legend: bool = False
) -> plt.Axes:
    """Draw a 2-D rib cross-section (profile + spar lines)."""
    colors = _default_colors()
    if ax is None:
        _, ax = plt.subplots()

    ax.plot(*rib.profile, color=colors[0], label="Profile")
    ax.fill(*rib.profile, color=colors[0], alpha=0.1)

    spar_ids = rib.spar_ids
    sv = rib.spar_vertices
    n_spars = len(rib.spar_positions)
    for i in range(1, n_spars + 1):
        xs = [sv[0][i - 1], sv[0][-i]]
        ys = [sv[1][i - 1], sv[1][-i]]
        ax.plot(xs, ys, color=colors[3], alpha=0.3, linestyle="-.", linewidth=4)

    # Inner cavity contours (hollow ribs)
    if rib.inner_profiles is not None:
        for profile in rib.inner_profiles:
            if profile is None:
                continue
            inner_x, inner_y = profile
            ax.plot(inner_x, inner_y, color=colors[2], linestyle="--", linewidth=1.5)
            ax.fill(inner_x, inner_y, color="white", alpha=0.8)

    ax.set_xlabel("X [m]")
    ax.set_ylabel("Z [m]")
    return ax


def plot_rib_annotations(rib: Rib, wing: Wing) -> None:
    """Plot a rib with node-ID labels for spars and stringers."""
    colors = _default_colors()
    ax = plot_rib(rib)
    fig = plt.gcf()
    fig.set_size_inches(10, 5)
    ax.axis("off")
    ax.set_title(f"Rib at y = {rib.span_position:.2f} m")

    ax.scatter(rib.profile[0][0], rib.profile[1][0], color=colors[0], marker="x")

    cg = rib.profile_cg
    chord = rib.chord
    twist = rib.twist

    # Surface-id range for annotation
    surf_ids = [
        rib.ids.surface_base + n
        for n in range(1, len(rib.spar_positions) * 2)
    ]

    ax.text(
        cg[0] - 0.25 * chord,
        cg[1] - 0.01 * chord,
        f"Points : [{rib.ids.point_start}, {rib.ids.point_end}]\n"
        f" Surfaces : {surf_ids}",
        color=colors[0],
        fontsize=16,
        rotation=-3 * np.rad2deg(twist),
    )

    sv = rib.spar_vertices
    for i in range(len(rib.spar_positions) * 2):
        ax.text(
            -0.04 * chord + sv[0][i],
            0.012 * chord + sv[1][i],
            f"{rib.spar_ids[i]:_.0f}",
            color=colors[3],
            fontsize=12,
            rotation=-3 * np.rad2deg(twist),
        )
        ax.scatter(sv[0][i], sv[1][i], color=colors[3], marker="o", s=80)

    stv = rib.stringer_vertices
    for i in range(len(rib.stringer_positions) * 2):
        ax.text(
            stv[0][i],
            -0.033 * chord + stv[1][i],
            f"{rib.stringer_ids[i]:_.0f}",
            color=colors[1],
            fontsize=10,
            rotation=90,
        )
        ax.scatter(stv[0][i], stv[1][i], color=colors[1], marker="x", s=80)


# ------------------------------------------------------------------
# 3-D plots
# ------------------------------------------------------------------


def plot_rib_3d(rib: Rib, ax: Axes3D | None = None) -> Axes3D:
    """Plot a single rib as a filled 3-D polygon."""
    colors = _default_colors()
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    verts = [[x, rib.span_position, z] for x, z in zip(*rib.profile)]
    poly = Poly3DCollection([verts], alpha=0.5, facecolor=colors[0], edgecolor="blue")
    ax.add_collection3d(poly)
    return ax


def plot_side_view(wing: Wing, ax: plt.Axes | None = None) -> plt.Axes:
    """Overlay all ribs in 2-D."""
    if ax is None:
        _, ax = plt.subplots()
    ax.set_aspect("equal")
    for rib in wing.ribs:
        plot_rib(rib, ax)
    return ax


def plot_3d_view(wing: Wing, ax: Axes3D | None = None) -> Axes3D:
    """Full 3-D wing with rib polygons, spar panels and stringer lines."""
    colors = _default_colors()
    if ax is None:
        _, ax = plt.subplots(subplot_kw={"projection": "3d"})

    for rib in wing.ribs:
        plot_rib_3d(rib, ax)

    # Spar panels
    n_spars = len(wing.spar_positions)
    for i in range(1, n_spars + 1):
        verts = []
        for rib in wing.ribs:
            verts.append([rib.spar_vertices[0][i - 1], rib.span_position, rib.spar_vertices[1][i - 1]])
        for rib in reversed(wing.ribs):
            verts.append([rib.spar_vertices[0][-i], rib.span_position, rib.spar_vertices[1][-i]])
        poly = Poly3DCollection([verts], alpha=0.5, facecolor=colors[3], edgecolor="red")
        ax.add_collection3d(poly)

    # Stringer lines (no global variables)
    n_stringer_pts = len(wing.stringer_positions) * 2
    for i in range(n_stringer_pts):
        xs = [rib.stringer_vertices[0][i] for rib in wing.ribs]
        ys = [rib.span_position for rib in wing.ribs]
        zs = [rib.stringer_vertices[1][i] for rib in wing.ribs]
        ax.plot(xs, ys, zs, color=colors[1])

    return ax
