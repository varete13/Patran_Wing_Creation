"""Wing Profiler for Nastran/Patran FEA.

Loads wing configurations from JSON files in configs/ and produces plots and
SES exports.  Run generate_configs.py first to create or update the JSON files.

Usage::

    python generate_configs.py   # create / refresh configs/
    python main.py               # plot and export
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from perfilador import (
    NACAAirfoil,
    Wing,
    plot_3d_view,
    plot_side_view,
    save_ses,
    save_bdf,
    FEAProperties,
    plot_rib,
)

CONFIGS = Path("configs")


def load(name: str) -> Wing:
    """Load a wing configuration from configs/<name>.json."""
    return Wing.from_json(CONFIGS / f"{name}.json")


def main() -> None:
    # ------------------------------------------------------------------
    # Load all wing variants from JSON
    # ------------------------------------------------------------------
    wing_const           = load("wing_const")
    wing_var             = load("wing_var")
    wing_optimized       = load("wing_optimized")
    wing_cosine          = load("wing_cosine")
    wing_hollow          = load("wing_hollow")
    wing_hollow_var      = load("wing_hollow_var")
    wing_combined        = load("wing_combined")
    wing_cuts            = load("wing_cuts")
    wing_cuts_hollow     = load("wing_cuts_hollow")
    wing_cuts_hollow_var = load("wing_cuts_hollow_var")

    # ------------------------------------------------------------------
    # Side views — constant vs variable profile
    # ------------------------------------------------------------------
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    plot_side_view(wing_const, ax1)
    ax1.set_title("Constant profile (NACA 6412)")
    plot_side_view(wing_var, ax2)
    ax2.set_title("Variable profile (6415 → 4412 → 2409)")

    # ------------------------------------------------------------------
    # 3-D views
    # ------------------------------------------------------------------
    _, (ax3, ax4) = plt.subplots(
        1, 2, figsize=(16, 6), subplot_kw={"projection": "3d"}
    )
    plot_3d_view(wing_const, ax3)
    ax3.set_title("Constant profile")
    plot_3d_view(wing_var, ax4)
    ax4.set_title("Variable profile")

    # ------------------------------------------------------------------
    # Uniform vs cosine spacing on the root rib
    # ------------------------------------------------------------------
    _, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    ax5.set_aspect("equal")
    ax6.set_aspect("equal")
    plot_rib(wing_const.ribs[0], ax5)
    ax5.set_title("Uniform spacing")
    plot_rib(wing_cosine.ribs[0], ax6)
    ax6.set_title("Cosine spacing")

    # ------------------------------------------------------------------
    # Hollow ribs
    # ------------------------------------------------------------------
    _, (ax7, ax8) = plt.subplots(1, 2, figsize=(14, 5))
    ax7.set_aspect("equal")
    ax8.set_aspect("equal")
    plot_rib(wing_hollow.ribs[0], ax7)
    ax7.set_title("Hollow rib — root (chord=9 m)")
    plot_rib(wing_hollow.ribs[4], ax8)
    ax8.set_title("Hollow rib — mid-span")

    # ------------------------------------------------------------------
    # Variable wall thickness
    # ------------------------------------------------------------------
    _, (ax9, ax10) = plt.subplots(1, 2, figsize=(14, 5))
    ax9.set_aspect("equal")
    ax10.set_aspect("equal")
    plot_rib(wing_hollow_var.ribs[0], ax9)
    ax9.set_title("Variable wall — root (t=0.10/0.12/0.08 m)")
    plot_rib(wing_hollow_var.ribs[-1], ax10)
    ax10.set_title("Variable wall — tip (t=0.05/0.05/0.05 m)")

    # ------------------------------------------------------------------
    # Combined: variable airfoil + variable wall thickness
    # ------------------------------------------------------------------
    _, (ax11, ax12) = plt.subplots(1, 2, figsize=(14, 5))
    ax11.set_aspect("equal")
    ax12.set_aspect("equal")
    plot_rib(wing_combined.ribs[0], ax11)
    ax11.set_title("Combined — root (6415, t=0.20/0.15/0.10 m)")
    plot_rib(wing_combined.ribs[-1], ax12)
    ax12.set_title("Combined — tip (2409, t=0.05/0.05/0.05 m)")

    # ------------------------------------------------------------------
    # Inner cuts
    # ------------------------------------------------------------------
    print("\nInner cuts — root rib section breakdown:")
    for sec_name, subsections in wing_cuts.ribs[0].cut_sections.items():
        for k, sub in enumerate(subsections, 1):
            nastran_ids = [idx + wing_cuts.ribs[0].ids.point_start for idx in sub]
            print(f"  {sec_name}-{k}: {len(sub)} pts — IDs {nastran_ids[0]} -- {nastran_ids[-1]}")

    _, (ax13, ax14) = plt.subplots(1, 2, figsize=(14, 5))
    ax13.set_aspect("equal")
    ax14.set_aspect("equal")
    plot_rib(wing_optimized.ribs[0], ax13)
    ax13.set_title("Sin inner_cuts — root")
    plot_rib(wing_cuts.ribs[0], ax14)
    ax14.set_title("Con inner_cuts — LE cortado en 2")

    # ------------------------------------------------------------------
    # Inner cuts + hollow sub-sections
    # ------------------------------------------------------------------
    _, (ax15, ax16) = plt.subplots(1, 2, figsize=(14, 5))
    ax15.set_aspect("equal")
    ax16.set_aspect("equal")
    plot_rib(wing_cuts_hollow.ribs[0], ax15)
    ax15.set_title("Inner cuts + hollow (const. wall) — root")
    plot_rib(wing_cuts_hollow_var.ribs[0], ax16)
    ax16.set_title("Inner cuts + hollow (var. wall) — root")

    # ------------------------------------------------------------------
    # Section map — identify local indices for inner_cuts
    # ------------------------------------------------------------------
    Wing.section_map_plot(
        airfoil=NACAAirfoil("6412"),
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        n_skin_points=wing_optimized.n_skin_points,
        point_spacing="cosine",
    )

    # plt.show()

    # ------------------------------------------------------------------
    # Export SES
    # ------------------------------------------------------------------
    save_ses(wing_const,           "wing_constant_profile.ses.01")
    save_ses(wing_var,             "wing_variable_profile.ses.01")
    save_ses(wing_optimized,       "wing_optimized_profile.ses.01")
    save_ses(wing_cosine,          "wing_cosine_spacing.ses.01")
    save_ses(wing_hollow,          "wing_hollow_ribs.ses.01")
    save_ses(wing_hollow_var,      "wing_hollow_variable_wall.ses.01")
    save_ses(wing_combined,        "wing_combined.ses.01")
    save_ses(wing_cuts,            "wing_inner_cuts.ses.01")
    save_ses(wing_cuts_hollow,     "wing_inner_cuts_hollow.ses.01")
    save_ses(wing_cuts_hollow_var, "wing_inner_cuts_hollow_var.ses.01")

    save_bdf(wing_combined,        "wing_combined.bdf",FEAProperties(),n_span_div=4,n_rib_layers=4)



if __name__ == "__main__":
    main()
