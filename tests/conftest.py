import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from perfilador import BarProps, FEAProperties, MatProps, NACAAirfoil, Wing  # noqa: E402

SPAN = 35.0
BASE = dict(
    span=SPAN,
    n_ribs=4,
    spar_positions=[0.2, 0.65],
    stringer_positions=np.linspace(0.05, 0.8, 8).tolist(),
    chord_distribution=([0, 5.18, SPAN], [9, 7, 2]),
    offset_distribution=([0, 5.18, SPAN], [0, 2.42, 13.06]),
    twist_distribution=([0, SPAN], [np.deg2rad(2), -np.deg2rad(4)]),
    elevation_distribution=([0, SPAN], [0.0, SPAN * np.sin(6 * np.pi / 180)]),
    airfoil=NACAAirfoil("6412"),
    point_spacing="cosine",
)

# Rib-face variants exercised by every mesher
VARIANTS = {
    "solid": dict(n_skin_points=25),
    "hollow": dict(n_skin_points=25, wall_thickness=[0.08, 0.10, 0.06]),
    "cuts": dict(n_skin_points=25, inner_cuts=[(2, 43)]),
    "cuts_hollow": dict(
        n_skin_points=25, inner_cuts=[(2, 43)],
        cut_wall_thickness=[0.15, 0.12, 0.10],
    ),
    "cuts_hollow_fine": dict(
        n_skin_points=60, inner_cuts=[(0.017, 0.1), (0.4, 0.4)],
        cut_wall_thickness=[0.15, 0.12, 0.10], exact_stations=True,
    ),
}


def make_wing(variant: str) -> Wing:
    return Wing(**VARIANTS[variant], **BASE)


@pytest.fixture
def fea() -> FEAProperties:
    m = MatProps(E=70e9, nu=0.33, rho=2700.0, t=0.003)
    return FEAProperties(
        rib=m, spar=m, skin=m, stringer=m,
        stringer_bar=BarProps(A=1.5e-4, I1=2e-8, I2=2e-8, J=1e-8),
    )
