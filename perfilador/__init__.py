"""Wing profiler for Nastran/Patran FEA."""

from .airfoil import NACAAirfoil
from .exporters import save_bdf, save_ses
from .fea_props import BarProps, FEAProperties, MatProps
from .geometry import extract_sections, hollow_rib_sections, offset_polygon
from .plotting import plot_3d_view, plot_rib, plot_rib_annotations, plot_side_view
from .rib import Rib
from .wing import EntitySummary, Wing

__all__ = [
    "NACAAirfoil",
    "Wing",
    "EntitySummary",
    "Rib",
    "FEAProperties",
    "MatProps",
    "BarProps",
    "save_bdf",
    "save_ses",
    "plot_side_view",
    "plot_3d_view",
    "plot_rib",
    "plot_rib_annotations",
    "extract_sections",
    "hollow_rib_sections",
    "offset_polygon",
]
