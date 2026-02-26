"""Wing Profiler for Nastran/Patran FEA.

Generates NACA 4-digit airfoil wing geometry and exports to BDF/SES formats.
"""
import numpy as np
import matplotlib.pyplot as plt

from perfilador import (
    NACAAirfoil,
    Wing,
    plot_3d_view,
    plot_side_view,
    save_ses,
    plot_rib
)


def main() -> None:
    span = 35.0

    # --- Constant airfoil along the span ---
    wing_const = Wing(
        airfoil=NACAAirfoil("6412"),
        span=span,
        n_ribs=10,
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        chord_distribution=([0, 5.18, span], [9, 7, 2]),
        offset_distribution=([0, 5.18, span], [0, 2.42, 13.06]),
        twist_distribution=([0, span], [np.deg2rad(2), -np.deg2rad(4)]),
        elevation_distribution=([0, span], [0, span * np.sin(6 * np.pi / 180)]),
        n_skin_points=20,
    )

    # --- Variable airfoil along the span ---
    # Thick, high-camber root (6415) transitioning to thin,
    # low-camber tip (2409) through an intermediate station (4412).
    wing_var = Wing(
        airfoil_distribution={0: "6415", span / 2: "4412", span: "2409"},
        span=span,
        n_ribs=10,
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        chord_distribution=([0, 5.18, span], [9, 7, 2]),
        offset_distribution=([0, 5.18, span], [0, 2.42, 13.06]),
        twist_distribution=([0, span], [np.deg2rad(2), -np.deg2rad(4)]),
        elevation_distribution=([0, span], [0, span * np.sin(6 * np.pi / 180)]),
        n_skin_points=20,
    )


    # Compare side views

    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.set_aspect("equal")
    ax2.set_aspect("equal")
    plot_side_view(wing_const, ax1)
    ax1.set_title("Constant profile (NACA 6412)")
    plot_side_view(wing_var, ax2)
    ax2.set_title("Variable profile (6415 -> 4412 -> 2409)")

    # 3D views
    _, (ax3, ax4) = plt.subplots(
        1, 2, figsize=(16, 6), subplot_kw={"projection": "3d"}
    )
    plot_3d_view(wing_const, ax3)
    ax3.set_title("Constant profile")
    plot_3d_view(wing_var, ax4)
    ax4.set_title("Variable profile")


    # --- Optimized skin resolution for a given entity budget ---
    n_sp = Wing.max_skin_points(
        n_ribs=10,
        n_spars=len([0.2, 0.65]),
        n_stringers=len(np.linspace(0.05, 0.8, 8)),
        max_entities=1_190,
    )
    print(f"\nOptimal n_skin_points for 10k budget: {n_sp}")

    wing_optimized = Wing(
        airfoil=NACAAirfoil("6412"),
        span=span,
        n_ribs=10,
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        chord_distribution=([0, 5.18, span], [9, 7, 2]),
        offset_distribution=([0, 5.18, span], [0, 2.42, 13.06]),
        twist_distribution=([0, span], [np.deg2rad(2), -np.deg2rad(4)]),
        elevation_distribution=([0, span], [0, span * np.sin(6 * np.pi / 180)]),
        n_skin_points=n_sp,
    )
    print("\nOptimized skin resolution summary:")
    print(wing_optimized.summary())
    # --- Cosine point spacing (clusters points near LE and TE) ---
    wing_cosine = Wing(
        airfoil=NACAAirfoil("6412"),
        span=span,
        n_ribs=10,
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        chord_distribution=([0, 5.18, span], [9, 7, 2]),
        offset_distribution=([0, 5.18, span], [0, 2.42, 13.06]),
        twist_distribution=([0, span], [np.deg2rad(2), -np.deg2rad(4)]),
        elevation_distribution=([0, span], [0, span * np.sin(6 * np.pi / 180)]),
        n_skin_points=n_sp,
        point_spacing="cosine",
    )
    print("\nCosine spacing summary:")
    print(wing_cosine.summary())

    # Compare uniform vs cosine spacing on the root rib

    _, (ax5, ax6) = plt.subplots(1, 2, figsize=(14, 5))
    ax5.set_aspect("equal")
    ax6.set_aspect("equal")
    plot_rib(wing_const.ribs[0], ax5)
    ax5.set_title("Uniform spacing")
    plot_rib(wing_cosine.ribs[0], ax6)
    ax6.set_title("Cosine spacing")

    # --- Hollow ribs (interior cavities via wall offset) ---
    # Each section (LE, spar box, TE) gets an independent wall thickness.
    wing_hollow = Wing(
        airfoil=NACAAirfoil("6412"),
        span=span,
        n_ribs=10,
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        chord_distribution=([0, 5.18, span], [9, 7, 2]),
        offset_distribution=([0, 5.18, span], [0, 2.42, 13.06]),
        twist_distribution=([0, span], [np.deg2rad(2), -np.deg2rad(4)]),
        elevation_distribution=([0, span], [0, span * np.sin(6 * np.pi / 180)]),
        point_spacing="cosine",
        n_skin_points=n_sp,
        wall_thickness=[0.08, 0.10, 0.06],  # LE, spar box, TE
    )

    _, (ax7, ax8) = plt.subplots(1, 2, figsize=(14, 5))
    ax7.set_aspect("equal")
    ax8.set_aspect("equal")
    plot_rib(wing_hollow.ribs[0], ax7)
    ax7.set_title("Hollow rib — root (chord=9m)")
    plot_rib(wing_hollow.ribs[4], ax8)
    ax8.set_title("Hollow rib — mid-span")

    # --- Variable wall thickness along the span ---
    # Wall thickness tapers from root to tip in each section:
    #   LE section:   0.10 m (root) → 0.04 m (tip)
    #   Spar box:     0.12 m (root) → 0.05 m (tip)
    #   TE section:   0.08 m (root) → 0.03 m (tip)
    wing_hollow_var = Wing(
        airfoil=NACAAirfoil("6412"),
        span=span,
        n_ribs=10,
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        chord_distribution=([0, 5.18, span], [9, 7, 2]),
        offset_distribution=([0, 5.18, span], [0, 2.42, 13.06]),
        twist_distribution=([0, span], [np.deg2rad(2), -np.deg2rad(4)]),
        elevation_distribution=([0, span], [0, span * np.sin(6 * np.pi / 180)]),
        point_spacing="cosine",
        n_skin_points=n_sp,
        wall_thickness_distribution={
            0:    [0.10, 0.12, 0.08],   # root:  LE, spar box, TE
            span: [0.05, 0.05, 0.05],   # tip:   LE, spar box, TE
        },
    )

    _, (ax9, ax10) = plt.subplots(1, 2, figsize=(14, 5))
    ax9.set_aspect("equal")
    ax10.set_aspect("equal")
    plot_rib(wing_hollow_var.ribs[0], ax9)
    ax9.set_title("Variable wall — root (t=0.10/0.12/0.08 m)")
    plot_rib(wing_hollow_var.ribs[-1], ax10)
    ax10.set_title("Variable wall — tip (t=0.04/0.05/0.03 m)")

    # --- Combined: variable airfoil + variable wall thickness ---
    # Root: thick cambered section (6415), heavy walls.
    # Mid:  intermediate section  (4412), medium walls.
    # Tip:  thin low-camber section (2409), thin walls.
    wing_combined = Wing(
        airfoil_distribution={
            0:        "6415",
            span / 2: "4412",
            span:     "2409",
        },
        span=span,
        n_ribs=10,
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        chord_distribution=([0, 5.18, span], [9, 7, 2]),
        offset_distribution=([0, 5.18, span], [0, 2.42, 13.06]),
        twist_distribution=([0, span], [np.deg2rad(2), -np.deg2rad(4)]),
        elevation_distribution=([0, span], [0, span * np.sin(6 * np.pi / 180)]),
        point_spacing="cosine",
        n_skin_points=n_sp,
        wall_thickness_distribution={
            0:        [0.20, 0.15, 0.1],   # root:  LE, spar box, TE
            span / 2: [0.10, 0.08, 0.05],   # mid
            span:     [0.05, 0.05, 0.05],   # tip
        },
    )

    _, (ax11, ax12) = plt.subplots(1, 2, figsize=(14, 5))
    ax11.set_aspect("equal")
    ax12.set_aspect("equal")
    plot_rib(wing_combined.ribs[0], ax11)
    ax11.set_title("Combined — root (6415, t=0.10/0.12/0.08 m)")
    plot_rib(wing_combined.ribs[-1], ax12)
    ax12.set_title("Combined — tip (2409, t=0.04/0.05/0.03 m)")

    # plt.show()
    # Export
    save_ses(wing_const, "wing_constant_profile.ses.01")
    save_ses(wing_var, "wing_variable_profile.ses.01")
    save_ses(wing_optimized, "wing_optimized_profile.ses.01")
    save_ses(wing_cosine, "wing_cosine_spacing.ses.01")
    save_ses(wing_hollow, "wing_hollow_ribs.ses.01")
    save_ses(wing_hollow_var, "wing_hollow_variable_wall.ses.01")
    save_ses(wing_combined, "wing_combined.ses.01")


if __name__ == "__main__":
    main()

