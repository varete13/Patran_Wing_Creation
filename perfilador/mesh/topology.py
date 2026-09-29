"""Rib-face topology shared by every rib mesher.

A rib face is split into *patches*: the standard LE / Box / TE sections, or
the sub-sections produced by ``inner_cuts``.  Each patch is a closed ring of
outer-profile nodes (local indices) plus an optional cavity polygon.

Ring edges are either **skin** edges (two consecutive profile nodes, shared
with the skin panels, always one element) or **web** edges (spar webs and cut
lines, straight, subdivided into ``n_web`` elements).  Web-edge interior nodes
are identified by a key that does not depend on the patch, so neighbouring
patches and the spar-web panels share them.

Node keys used by :class:`RibMesh`:

* ``("p", i)``            — outer profile node with local index *i*.
* ``("w", lo, hi, k)``    — *k*-th interior node (1…n_web-1) of the web edge
  between profile nodes *lo* < *hi*, counted from *lo*.
* ``("n", j)``            — mesher-created interior node.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Hashable

import numpy as np

if TYPE_CHECKING:
    from ..rib import Rib

NodeKey = Hashable


@dataclass
class RibPatch:
    """One meshable region of a rib face."""

    name: str
    ring: list[int]                 # local profile indices, counter-clockwise
    inner: np.ndarray | None = None  # (M, 2) cavity polygon, CCW, not closed


@dataclass
class RibMesh:
    """Mesher output for one rib face (2-D, rib-plane coordinates)."""

    nodes: dict[NodeKey, np.ndarray] = field(default_factory=dict)
    elements: list[tuple[NodeKey, ...]] = field(default_factory=list)
    info: list[str] = field(default_factory=list)

    def add_node(self, xy) -> NodeKey:
        key = ("n", len(self.nodes))
        while key in self.nodes:
            key = ("n", key[1] + 1)
        self.nodes[key] = np.asarray(xy, dtype=float)
        return key


def signed_area(pts: np.ndarray) -> float:
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def rib_points(rib: Rib) -> np.ndarray:
    """Unique outer-profile coordinates ``(n_unique, 2)`` of *rib*."""
    px, py = rib.profile
    if np.allclose(px[0], px[-1]) and np.allclose(py[0], py[-1]):
        px, py = px[:-1], py[:-1]
    return np.column_stack([px, py])


def is_skin_edge(i: int, j: int, n: int) -> bool:
    """True when profile nodes *i*, *j* are consecutive around the profile."""
    return abs(i - j) == 1 or {i, j} == {0, n - 1}


def web_key(i: int, j: int, k_from_i: int, n_web: int) -> NodeKey:
    """Key of the interior web-edge node *k_from_i* steps from *i* towards *j*."""
    if i < j:
        return ("w", i, j, k_from_i)
    return ("w", j, i, n_web - k_from_i)


def web_node_xy(P: np.ndarray, key: NodeKey, n_web: int) -> np.ndarray:
    _, lo, hi, k = key
    t = k / n_web
    return (1.0 - t) * P[lo] + t * P[hi]


def rib_patches(rib: Rib) -> list[RibPatch]:
    """Patches of *rib*: cut sub-sections if defined, else LE / Box / TE."""
    P = rib_points(rib)
    if rib.cut_sections is not None:
        named = [
            (name if len(subs) == 1 else f"{name}.{k + 1}", sub)
            for name, subs in rib.cut_sections.items()
            for k, sub in enumerate(subs)
        ]
        inners = rib.cut_inner_profiles or [None] * len(named)
    else:
        named = [
            (name, subs[0]) for name, subs in rib._apply_cuts([]).items()
        ]
        inners = rib.inner_profiles or [None] * len(named)

    patches: list[RibPatch] = []
    for (name, sub), inner in zip(named, inners):
        ring = list(dict.fromkeys(int(i) for i in sub))
        if signed_area(P[ring]) < 0:
            ring = ring[::-1]
        cav = None
        if inner is not None:
            cav = np.column_stack(inner)
            if np.allclose(cav[0], cav[-1]):
                cav = cav[:-1]
            if signed_area(cav) < 0:
                cav = cav[::-1]
        patches.append(RibPatch(name=name, ring=ring, inner=cav))
    return patches


def web_edges(rib: Rib) -> list[tuple[int, int]]:
    """All distinct web edges ``(lo, hi)`` of the rib face."""
    n = len(rib_points(rib))
    edges: set[tuple[int, int]] = set()
    for p in rib_patches(rib):
        m = len(p.ring)
        for k in range(m):
            i, j = p.ring[k], p.ring[(k + 1) % m]
            if not is_skin_edge(i, j, n):
                edges.add((min(i, j), max(i, j)))
    return sorted(edges)


# ---------------------------------------------------------------------------
# Boundary discretisation helpers
# ---------------------------------------------------------------------------

def boundary_nodes(
    patch: RibPatch, P: np.ndarray, n_web: int
) -> tuple[list[NodeKey], list[np.ndarray], list[int]]:
    """Discretised outer boundary of *patch* (CCW, not closed).

    Returns ``(keys, coords, ring_pos)`` where ``ring_pos[k]`` is the position
    in ``keys`` of ``patch.ring[k]``.
    """
    n = len(P)
    keys: list[NodeKey] = []
    xy: list[np.ndarray] = []
    ring_pos: list[int] = []
    m = len(patch.ring)
    for k in range(m):
        i, j = patch.ring[k], patch.ring[(k + 1) % m]
        ring_pos.append(len(keys))
        keys.append(("p", i))
        xy.append(P[i])
        if not is_skin_edge(i, j, n):
            for s in range(1, n_web):
                wk = web_key(i, j, s, n_web)
                keys.append(wk)
                xy.append(web_node_xy(P, wk, n_web))
    return keys, xy, ring_pos


def patch_corners(patch: RibPatch, P: np.ndarray, n_sp: int) -> list[int]:
    """Corner positions (indices into ``patch.ring``) of *patch*.

    Corners are the end points of web edges plus the leading-edge (local
    index 0) and trailing-edge (``n_sp - 1``) tips when present.  If fewer
    than three are found, extra corners are added at the ring nodes farthest
    (in arc length) from the existing ones.
    """
    n = len(P)
    ring = patch.ring
    m = len(ring)
    c: set[int] = set()
    for k in range(m):
        if not is_skin_edge(ring[k], ring[(k + 1) % m], n):
            c.update({k, (k + 1) % m})
    for tip in (0, n_sp - 1):
        if tip in ring:
            c.add(ring.index(tip))
    if len(c) < 3:
        R = P[ring]
        seg = np.hypot(*np.diff(np.vstack([R, R[:1]]), axis=0).T)
        arc = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
        L = arc[-1] + seg[-1]
        while len(c) < 3:
            def dist(k):
                return min(min(abs(arc[k] - arc[q]), L - abs(arc[k] - arc[q]))
                           for q in c) if c else 0.0
            c.add(max(range(m), key=dist))
    return sorted(c)
