"""Element-quality metrics for 2-D rib-face meshes."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from .topology import RibMesh, RibPatch, rib_points, signed_area


@dataclass
class MeshQuality:
    n_quads: int
    n_trias: int
    n_invalid: int        # inverted or non-convex elements
    min_angle: float      # degrees, over all element corners
    max_angle: float
    max_aspect: float     # longest / shortest edge
    area: float           # total meshed area

    def __str__(self) -> str:
        return (
            f"quads={self.n_quads} trias={self.n_trias} invalid={self.n_invalid} "
            f"angle=[{self.min_angle:.1f}°, {self.max_angle:.1f}°] "
            f"maxAR={self.max_aspect:.1f} area={self.area:.4f}"
        )


def element_metrics(pts: np.ndarray) -> tuple[bool, float, float, float, float]:
    """``(valid, min_angle, max_angle, aspect, signed_area)`` of one polygon."""
    k = len(pts)
    cross, angles, edges = [], [], []
    for i in range(k):
        a = pts[i - 1] - pts[i]
        b = pts[(i + 1) % k] - pts[i]
        cross.append(b[0] * a[1] - b[1] * a[0])
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        edges.append(nb)
        if na < 1e-14 or nb < 1e-14:
            angles.append(0.0)
        else:
            angles.append(np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1))))
    cross = np.array(cross)
    valid = bool(np.all(cross > 0) or np.all(cross < 0))
    aspect = max(edges) / max(min(edges), 1e-14)
    return valid, min(angles), max(angles), aspect, signed_area(pts)


def mesh_quality(mesh: RibMesh) -> MeshQuality:
    nq = nt = bad = 0
    amin, amax, ar, area = 180.0, 0.0, 1.0, 0.0
    for e in mesh.elements:
        pts = np.array([mesh.nodes[k] for k in e])
        valid, a0, a1, asp, sa = element_metrics(pts)
        nq += len(e) == 4
        nt += len(e) == 3
        bad += (not valid) or sa <= 0
        amin, amax, ar = min(amin, a0), max(amax, a1), max(ar, asp)
        area += sa
    return MeshQuality(nq, nt, bad, amin, amax, ar, area)


def expected_area(rib, patches: list[RibPatch]) -> float:
    """Rib-face area implied by the geometry: patches minus cavities."""
    P = rib_points(rib)
    a = 0.0
    for p in patches:
        a += signed_area(P[p.ring])
        if p.inner is not None:
            a -= abs(signed_area(p.inner))
    return a


def free_edges(mesh: RibMesh) -> Counter:
    """Edges used by exactly one element (the mesh boundary)."""
    cnt: Counter = Counter()
    for e in mesh.elements:
        for i in range(len(e)):
            a, b = e[i], e[(i + 1) % len(e)]
            cnt[frozenset((a, b))] += 1
    return Counter({k: v for k, v in cnt.items() if v == 1})
