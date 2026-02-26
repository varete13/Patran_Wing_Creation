"""Patran SES session-file exporter.

Generates session commands for grids, curves, trimmed surfaces, spar panels,
stringer segments and skin quad panels.
"""
from __future__ import annotations

try:
    from itertools import pairwise
except ImportError:  # Python < 3.10
    from itertools import tee

    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..id_manager import IDScheme
    from ..rib import Rib
    from ..wing import Wing

# Group names written into the SES file (kept in original Spanish for
# backward-compatibility with existing Patran sessions).
GROUP_RIBS = "Secciones_Ala"
GROUP_SPARS = "Tramos_Largueros"
GROUP_STRINGERS = "Tramos_Largerillos"
GROUP_SKINS = "Secciones_Piel"


def _rib_geometry_commands(
    rib: Rib,
    surface_counter: int,
) -> tuple[list[str], int]:
    """SES commands for one rib: grid points, PWL curves and trimmed surfaces.

    Mirrors the original ``Costilla.ses_Geom_v2``.

    Returns the command lines and the updated surface counter.
    """
    cmds: list[str] = []
    cmds.append(
        f"#Curves/Surface for rib {rib.ids.rib_index}, "
        f"at y={rib.span_position:.2f}m \n"
    )

    # --- Grid points (skip the last one to avoid duplicate closing point) ---
    cmds.append("STRING asm_create_grid_xyz_created_ids[VIRTUAL]\n")
    profile_x, profile_y = rib.profile
    n_pts = len(profile_x)
    for n_id in range(n_pts - 1):
        px, py = profile_x[n_id], profile_y[n_id]
        grid_id = n_id + rib.ids.point_start
        cmds.append(
            f'asm_const_grid_xyz("{grid_id:<7.0f}",'
            f'"[{px:<7.4f} {py:<7.4f} {rib.span_position:<7.4f}]", '
            f'"Coord 0", asm_create_grid_xyz_created_ids) \n'
        )

    # --- Inner contour grid points (if hollow rib) ---
    # Each section's inner contour gets consecutive IDs starting from
    # inner_point_start.  Track the ID ranges per section for PWL curves.
    inner_section_ids: list[tuple[int, int] | None] = []  # (first_id, last_id) or None
    if rib.inner_profiles is not None:
        inner_id = rib.ids.inner_point_start
        for sec_idx, profile in enumerate(rib.inner_profiles):
            if profile is None:
                inner_section_ids.append(None)
                continue
            ix, iy = profile
            n_inner = len(ix) - 1  # skip closing duplicate
            first_inner_id = inner_id
            for k in range(n_inner):
                cmds.append(
                    f'asm_const_grid_xyz("{inner_id:<7.0f}",'
                    f'"[{ix[k]:<7.4f} {iy[k]:<7.4f} '
                    f'{rib.span_position:<7.4f}]", '
                    f'"Coord 0", asm_create_grid_xyz_created_ids) \n'
                )
                inner_id += 1
            inner_section_ids.append((first_inner_id, inner_id - 1))

    # --- PWL curves and trimmed surfaces per spar section ---
    cmds.append("STRING asm_create_line_pwl_created_ids[VIRTUAL]\n")
    spar_ids = rib.spar_ids
    n_spar_halves = len(spar_ids) // 2  # e.g. 2 spars → 4 ids → 2

    id_start = rib.ids.point_start
    id_end = rib.ids.point_end

    for section in range(n_spar_halves + 1):
        surface_counter += 1

        # --- Outer boundary PWL curve ---
        if section == 0:
            # Leading-edge section: from start around to first/last spar
            cmds.append(
                f'asm_const_line_pwl( "1", '
                f'"Point {id_start:.0f}:{spar_ids[0]:.0f} '
                f"{spar_ids[-1]:.0f}:{id_end:.0f} "
                f'{id_start:.0f}", '
                f"asm_create_line_pwl_created_ids )\n"
            )
            n_outer_curves = spar_ids[0] - id_start + id_end - spar_ids[-1] + 2

        elif section == n_spar_halves:
            # Last (trailing-edge) section between the two inner spar points
            cmds.append(
                f'asm_const_line_pwl( "1", '
                f'"Point {spar_ids[section - 1]:.0f}:{spar_ids[section]:.0f} '
                f'{spar_ids[section - 1]:.0f} ", '
                f"asm_create_line_pwl_created_ids )\n"
            )
            n_outer_curves = spar_ids[section] - spar_ids[section - 1] + 1

        else:
            i1 = section - 1
            i2 = section
            i3 = len(spar_ids) - 1 - i2
            i4 = len(spar_ids) - 1 - i1
            cmds.append(
                f'asm_const_line_pwl( "1", '
                f'"Point {spar_ids[i1]:.0f}:{spar_ids[i2]:.0f} '
                f"{spar_ids[i3]:.0f}:{spar_ids[i4]:.0f} "
                f'{spar_ids[i1]:.0f}", '
                f"asm_create_line_pwl_created_ids )\n"
            )
            n_outer_curves = (
                spar_ids[i2] - spar_ids[i1]
                + spar_ids[i4] - spar_ids[i3]
                + 2
            )

        # --- Inner boundary PWL curve (hollow rib) ---
        inner_curves_field = ""
        if inner_section_ids and inner_section_ids[section] is not None:
            first_id, last_id = inner_section_ids[section]
            n_inner_curves = last_id - first_id + 1
            # Start inner curve IDs right after the outer curves
            inner_curve_start = n_outer_curves + 1
            cmds.append(
                f'asm_const_line_pwl( "{inner_curve_start}", '
                f'"Point {first_id:.0f}:{last_id:.0f} '
                f'{first_id:.0f}", '
                f"asm_create_line_pwl_created_ids )\n"
            )
            inner_end = n_outer_curves + n_inner_curves
            inner_curves_field = f"Curve {inner_curve_start}:{inner_end}"

        cmds.append("STRING sgm_surface_trimmed__created_id[VIRTUAL]\n")
        cmds.append(
            f'sgm_create_surface_trimmed_v1( "{surface_counter}", '
            f'"Curve 1:{n_outer_curves}", '
            f'"{inner_curves_field}", "", '
            f"TRUE, TRUE, TRUE, TRUE, sgm_surface_trimmed__created_id )\n"
        )
        cmds.append(
            f'ga_group_entity_add( "{GROUP_RIBS}", '
            f'"Surface {surface_counter}" )\n'
        )

    return cmds, surface_counter


def save_ses(wing: Wing, filename: str | None = None) -> None:
    """Write a complete Patran SES session file for *wing*."""
    if filename is None:
        filename = (
            f"Wing_{wing.span:.0f}m_{wing.n_ribs:.0f}_s_"
            f"{wing.n_skin_points}_pts_skin.ses.01"
        )

    s = wing.id_scheme

    with open(filename, "w") as f:
        # Header and group creation
        f.write(
            f"#Points for {wing.n_ribs} sections of the "
            f"{wing.span:.2f}m wing \n"
        )
        f.write(f'ga_group_create( "{GROUP_RIBS}" )\n')
        f.write(f'ga_group_create( "{GROUP_SPARS}" )\n')
        f.write(f'ga_group_create( "{GROUP_STRINGERS}" )\n')
        f.write(f'ga_group_create( "{GROUP_SKINS}" )\n')

        # --- Per-rib geometry (grids, curves, trimmed surfaces) ---
        surface_counter = 0
        for rib in wing.ribs:
            # Initialise the surface counter for this rib exactly as the
            # original: base + stride * rib_index
            surface_counter = rib.ids.surface_base
            cmds, surface_counter = _rib_geometry_commands(rib, surface_counter)
            for line in cmds:
                f.write(line)

        # --- Spar / stringer segments between consecutive ribs ---
        f.write("STRING asm_line_2point_created_ids[VIRTUAL]\n")
        for prev_rib, next_rib in pairwise(wing.ribs):
            ids_prev = sorted(prev_rib.stringer_ids)
            ids_next = sorted(next_rib.stringer_ids)

            for prev_pt, next_pt in zip(ids_prev, ids_next):
                id_line = prev_pt - s.point_base + s.line_base
                f.write(
                    f'asm_const_line_2point( "{id_line:.0f}", '
                    f'"Point {prev_pt}", "Point {next_pt}",'
                    f'0,"",50.,1, asm_line_2point_created_ids )\n'
                )
                f.write(
                    f'ga_group_entity_add( "{GROUP_STRINGERS}", '
                    f'"Line {id_line}" )\n'
                )


            # Spar panel surfaces between consecutive ribs
            f.write("STRING sgm_create_surface__created_ids[VIRTUAL]\n")
            for n in range(1, len(wing.spar_positions) + 1):
                id_surf = (
                    s.surface_spar_base
                    + prev_rib.ids.rib_index * s.surface_spar_stride
                    + n - 1
                )
                p1 = prev_rib.spar_ids[n - 1]
                p2 = next_rib.spar_ids[n - 1]
                p3 = next_rib.spar_ids[-n]
                p4 = prev_rib.spar_ids[-n]

                f.write(
                    f'sgm_const_surface_vertex("{id_surf:.0f}",'
                    f'"Point {p1:.0f}","Point {p2:.0f}",'
                    f'"Point {p3:.0f}","Point {p4:.0f}",'
                    f"sgm_create_surface__created_ids)\n"
                )
                f.write(
                    f'ga_group_entity_add( "{GROUP_SPARS}",'
                    f'"Surface {id_surf:.0f}")\n'
                )

        # --- Skin quad panels ---
        f.write("STRING sgm_create_surface_created_ids[VIRTUAL]\n")
        for prev_rib, next_rib in pairwise(wing.ribs):
            n_sp = prev_rib.n_skin_points
            pts_1 = [prev_rib.ids.point_start + i for i in range(2 * n_sp - 2)]
            pts_1.append(pts_1[0])
            pts_2 = [next_rib.ids.point_start + i for i in range(2 * n_sp - 2)]
            pts_2.append(pts_2[0])

            for n, ((p1, p2), (p4, p3)) in enumerate(
                pairwise(zip(pts_1, pts_2))
            ):
                surf_id = (
                    s.surface_skin_base
                    + prev_rib.ids.rib_index * s.surface_skin_stride
                    + n
                )
                f.write(
                    f'sgm_const_surface_vertex("{surf_id:.0f}",'
                    f'"Point {p1:.0f}","Point {p2:.0f}",'
                    f'"Point {p3:.0f}","Point {p4:.0f}",'
                    f"sgm_create_surface_created_ids)\n"
                )
                f.write(
                    f'ga_group_entity_add( "{GROUP_SKINS}",'
                    f'"Surface {surf_id:.0f}")\n'
                )
