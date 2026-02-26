from __future__ import annotations

import numpy as np
from scipy.integrate import quad


class NACAAirfoil:
    """NACA 4-digit airfoil geometry and cross-section properties.

    Parameters
    ----------
    designation : str
        4-digit NACA code, e.g. ``"6412"``, ``"2412"``, ``"0012"``.
    """

    def __init__(self, designation: str = "6412") -> None:
        if len(designation) != 4 or not designation.isdigit():
            raise ValueError(f"Expected 4-digit NACA code, got '{designation}'")
        self.designation = designation
        self.max_camber = int(designation[0]) / 100.0
        self.camber_position = int(designation[1]) / 10.0
        self.thickness = int(designation[2:4]) / 100.0
        self._area: float | None = None
        self._cg: np.ndarray | None = None

    @classmethod
    def from_params(
        cls, max_camber: float, camber_position: float, thickness: float
    ) -> NACAAirfoil:
        """Create an airfoil from raw aerodynamic parameters.

        Useful for interpolated profiles where parameters don't map to
        exact integer NACA digits.

        Parameters
        ----------
        max_camber : float
            Maximum camber as fraction of chord (e.g. 0.06).
        camber_position : float
            Chordwise position of maximum camber (e.g. 0.4).
        thickness : float
            Maximum thickness as fraction of chord (e.g. 0.12).
        """
        instance = cls.__new__(cls)
        instance.designation = (
            f"{max_camber*100:.1f}/{camber_position*10:.1f}/{thickness*100:.1f}"
        )
        instance.max_camber = max_camber
        instance.camber_position = camber_position
        instance.thickness = thickness
        instance._area = None
        instance._cg = None
        return instance

    def evaluate(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Compute upper and lower surface coordinates.

        Parameters
        ----------
        x : array_like
            Chordwise stations in [0, 1].

        Returns
        -------
        xu, yu, xl, yl
            Upper (xu, yu) and lower (xl, yl) surface coordinates.
        """
        x = np.asarray(x, dtype=float)
        x = np.clip(x, 0.0, 1.0)

        m = self.max_camber
        p = self.camber_position
        t = self.thickness

        # Thickness distribution
        yt = 5.0 * t * (
            0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2
            + 0.2843 * x**3 - 0.1015 * x**4
        )

        # Camber line and its slope
        if p == 0 or m == 0:
            yc = np.zeros_like(x)
            dyc_dx = np.zeros_like(x)
        else:
            yc = np.where(
                x < p,
                m / p**2 * (2.0 * p * x - x**2),
                m / (1.0 - p) ** 2 * ((1.0 - 2.0 * p) + 2.0 * p * x - x**2),
            )
            dyc_dx = np.where(
                x < p,
                2.0 * m / p**2 * (p - x),
                2.0 * m / (1.0 - p) ** 2 * (p - x),
            )

        theta = np.arctan(dyc_dx)

        xu = x - yt * np.sin(theta)
        yu = yc + yt * np.cos(theta)
        xl = x + yt * np.sin(theta)
        yl = yc - yt * np.cos(theta)

        return xu, yu, xl, yl

    @property
    def area(self) -> float:
        """Cross-section area of the unit-chord airfoil (cached)."""
        if self._area is None:
            self._area = quad(
                lambda x: self.evaluate(x)[1] - self.evaluate(x)[3], 0, 1
            )[0]
        return self._area

    @property
    def cg(self) -> np.ndarray:
        """Centre of gravity ``[x, y]`` of the unit-chord airfoil (cached)."""
        if self._cg is None:
            a = self.area
            cx = quad(
                lambda x: (self.evaluate(x)[1] - self.evaluate(x)[3]) * x, 0, 1
            )[0] / a
            cy = quad(
                lambda x: (self.evaluate(x)[1] ** 2 - self.evaluate(x)[3] ** 2) / 2,
                0, 1,
            )[0] / a
            self._cg = np.array([cx, cy])
        return self._cg
