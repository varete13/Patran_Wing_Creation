"""Rib-face meshers for the BDF exporter.

Every mesher returns a :class:`RibMesh` built on the shared topology in
:mod:`perfilador.mesh.topology`, so rib faces stay node-compatible with the
skin panels (profile nodes) and with the spar webs (web-edge nodes).
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from .quality import MeshQuality, expected_area, free_edges, mesh_quality
from .topology import RibMesh, RibPatch, rib_patches, web_edges

if TYPE_CHECKING:
    from ..rib import Rib

#: name → callable(rib, n_web=..., n_wall=...) -> RibMesh
RIB_MESHERS: dict[str, Callable[..., RibMesh]] = {}


def mesh_rib_face(rib: Rib, method: str, n_web: int = 4, n_wall: int = 2) -> RibMesh:
    """Mesh the face of *rib* with the mesher registered as *method*."""
    try:
        mesher = RIB_MESHERS[method]
    except KeyError:
        raise ValueError(
            f"Unknown rib mesher {method!r}; available: "
            f"{sorted(RIB_MESHERS) + ['legacy']}"
        ) from None
    return mesher(rib, n_web=n_web, n_wall=n_wall)


# Register the meshers (each module adds itself to RIB_MESHERS)
from . import native  # noqa: E402,F401


__all__ = [
    "RIB_MESHERS",
    "RibMesh",
    "RibPatch",
    "MeshQuality",
    "mesh_rib_face",
    "mesh_quality",
    "expected_area",
    "free_edges",
    "rib_patches",
    "web_edges",
]
