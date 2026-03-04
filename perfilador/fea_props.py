"""FEA material and section properties for Nastran BDF export."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MatProps:
    """Isotropic material for a MAT1 card.

    Parameters
    ----------
    E : float
        Young's modulus [Pa].
    nu : float
        Poisson's ratio.
    rho : float
        Mass density [kg/m³].
    t : float
        Shell thickness [m] — used only for PSHELL cards.
    """

    E:   float = 70e9
    nu:  float = 0.33
    rho: float = 2700.0
    t:   float = 0.003


@dataclass
class BarProps:
    """Beam cross-section for a PBAR card.

    Parameters
    ----------
    A : float
        Cross-section area [m²].
    I1 : float
        Bending inertia about axis 1 [m⁴].
    I2 : float
        Bending inertia about axis 2 [m⁴].
    J : float
        Torsional constant [m⁴].
    """

    A:  float = 1e-4
    I1: float = 1e-8
    I2: float = 1e-8
    J:  float = 1e-8


@dataclass
class FEAProperties:
    """Complete material and section properties for Nastran BDF export.

    Three MAT1/PSHELL entries (ribs, spars, skin) and one MAT1/PBAR entry
    (stringers).  All default to aluminium-like values.

    Parameters
    ----------
    rib : MatProps
        Material and shell thickness for rib face elements (CTRIA3).
    spar : MatProps
        Material and shell thickness for spar web elements (CQUAD4).
    skin : MatProps
        Material and shell thickness for skin panel elements (CQUAD4).
    stringer : MatProps
        Material for stringer beam elements (CBAR).
    stringer_bar : BarProps
        Cross-section geometry for stringer PBAR.
    """

    rib:          MatProps = field(default_factory=MatProps)
    spar:         MatProps = field(default_factory=MatProps)
    skin:         MatProps = field(default_factory=MatProps)
    stringer:     MatProps = field(default_factory=MatProps)
    stringer_bar: BarProps = field(default_factory=BarProps)

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict."""
        return {
            "rib":          asdict(self.rib),
            "spar":         asdict(self.spar),
            "skin":         asdict(self.skin),
            "stringer":     asdict(self.stringer),
            "stringer_bar": asdict(self.stringer_bar),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "FEAProperties":
        """Reconstruct from a dict produced by :meth:`to_dict`."""
        return cls(
            rib=MatProps(**d["rib"]),
            spar=MatProps(**d["spar"]),
            skin=MatProps(**d["skin"]),
            stringer=MatProps(**d["stringer"]),
            stringer_bar=BarProps(**d["stringer_bar"]),
        )
