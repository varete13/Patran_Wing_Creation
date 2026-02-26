from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..rib import Rib
    from ..wing import Wing


def rib_to_bdf_lines(rib: Rib) -> list[str]:
    """Generate Nastran GRID card strings for one rib's profile points."""
    lines: list[str] = []
    for n_id, (px, py) in enumerate(zip(*rib.profile)):
        grid_id = n_id + rib.ids.point_start
        lines.append(
            f"GRID     {grid_id:<7.0f} {0:<7.0f} "
            f"{px:<7.4f} {py:<7.4f} {rib.span_position:<7.4f} \n"
        )
    return lines


def save_bdf(wing: Wing, filename: str | None = None) -> None:
    """Write all rib grid points to a Nastran BDF file."""
    if filename is None:
        filename = (
            f"Wing_{wing.span:.0f}m_{wing.n_ribs:.0f}_sections.bdf"
        )
    with open(filename, "w") as f:
        f.write(
            "$---1---$---2---$---3---$---4---$---5---"
            "$---6---$---7---$---8---$---9---$---10--$ \n"
        )
        for rib in wing.ribs:
            for line in rib_to_bdf_lines(rib):
                f.write(line)
