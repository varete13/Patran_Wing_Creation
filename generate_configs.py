"""Generate JSON config files for all example wing configurations.

Run this script whenever you change parameters.  It writes one JSON file per
wing variant into the ``configs/`` directory so that ``main.py`` can load them
without any hardcoded Wing() calls.

Usage::

    python generate_configs.py
"""
import numpy as np
from pathlib import Path

from perfilador import NACAAirfoil, Wing

CONFIGS = Path("configs")
CONFIGS.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Shared geometry parameters
# ---------------------------------------------------------------------------
span = 35.0

SPARS      = [0.2, 0.65]
STRINGERS  = np.linspace(0.05, 0.8, 8).tolist()

CHORD_DIST = ([0, 5.18, span], [9, 7, 2])
OFFSET_DIST = ([0, 5.18, span], [0, 2.42, 13.06])
TWIST_DIST  = ([0, span], [np.deg2rad(2), -np.deg2rad(4)])
ELEV_DIST   = ([0, span], [0.0, span * np.sin(6 * np.pi / 180)])

BASE = dict(
    span=span,
    n_ribs=10,
    spar_positions=SPARS,
    stringer_positions=STRINGERS,
    chord_distribution=CHORD_DIST,
    offset_distribution=OFFSET_DIST,
    twist_distribution=TWIST_DIST,
    elevation_distribution=ELEV_DIST,
)

# ---------------------------------------------------------------------------
# Entity-budget-optimal skin resolution
# ---------------------------------------------------------------------------
n_sp = Wing.max_skin_points(
    n_ribs=10,
    n_spars=len(SPARS),
    n_stringers=len(STRINGERS),
    max_entities=1_190,
)
print(f"Entity-budget n_skin_points: {n_sp}\n")

# ---------------------------------------------------------------------------
# Wing variants
# ---------------------------------------------------------------------------
configs: dict[str, Wing] = {

    # Constant NACA 6412 profile, coarse skin (demo quality)
    "wing_const": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=20,
        **BASE,
    ),

    # Profile interpolated 6415 → 4412 → 2409 along the span
    "wing_var": Wing(
        airfoil_distribution={0: "6415", span / 2: "4412", span: "2409"},
        n_skin_points=20,
        **BASE,
    ),

    # Budget-optimal skin, uniform spacing
    "wing_optimized": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=n_sp,
        **BASE,
    ),

    # Budget-optimal skin, cosine spacing
    "wing_cosine": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=n_sp,
        point_spacing="cosine",
        **BASE,
    ),

    # Hollow ribs — constant wall thickness per section
    "wing_hollow": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=n_sp,
        point_spacing="cosine",
        wall_thickness=[0.08, 0.10, 0.06],      # LE, spar box, TE
        **BASE,
    ),

    # Hollow ribs — wall thickness tapers root → tip
    "wing_hollow_var": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=n_sp,
        point_spacing="cosine",
        wall_thickness={
            0:    [0.10, 0.12, 0.08],           # root:  LE, spar box, TE
            span: [0.05, 0.05, 0.05],           # tip
        },
        **BASE,
    ),

    # Variable airfoil + variable wall thickness
    "wing_combined": Wing(
        airfoil_distribution={
            0:        "6415",
            span / 2: "4412",
            span:     "2409",
        },
        n_skin_points=n_sp,
        point_spacing="cosine",
        wall_thickness={
            0:        [0.20, 0.15, 0.10],       # root
            span / 2: [0.10, 0.08, 0.05],       # mid
            span:     [0.05, 0.05, 0.05],       # tip
        },
        **BASE,
    ),

    # Inner cuts — same cut on every rib (global mode)
    "wing_cuts": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=n_sp,
        point_spacing="cosine",
        inner_cuts=[(2, 47)],                   # LE section, upper+lower (n_sp = 27)
        **BASE,
    ),

    # Inner cuts + hollow sub-sections (constant wall)
    "wing_cuts_hollow": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=n_sp,
        point_spacing="cosine",
        inner_cuts=[(2, 47)],
        cut_wall_thickness=[0.15, 0.12, 0.10],  # LE, spar box, TE
        **BASE,
    ),

    # Inner cuts + hollow sub-sections (variable wall)
    "wing_cuts_hollow_var": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=n_sp,
        point_spacing="cosine",
        inner_cuts=[(2, 47)],
        cut_wall_thickness={
            0:    [0.18, 0.15, 0.12],           # root
            span: [0.08, 0.06, 0.05],           # tip
        },
        **BASE,
    ),

    # Megalodo — same as wing_cuts_hollow_var with a fixed 50-point skin
    "megalodo": Wing(
        airfoil=NACAAirfoil("6412"),
        n_skin_points=50,
        point_spacing="cosine",
        inner_cuts=[(3, 85)],                   # LE cut remapped to 50 skin points
        cut_wall_thickness={
            0:    [0.18, 0.15, 0.12],           # root
            span: [0.08, 0.06, 0.05],           # tip
        },
        **BASE,
    ),
}

# ---------------------------------------------------------------------------
# Save to configs/
# ---------------------------------------------------------------------------
for name, wing in configs.items():
    path = CONFIGS / f"{name}.json"
    wing.to_json(path)
    print(f"  saved  {path}")

print(f"\n{len(configs)} configs written to {CONFIGS}/")
