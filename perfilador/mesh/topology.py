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


def ring_side(keys: list, a: int, b: int) -> list[int]:
    """Boundary positions from *a* to *b* (inclusive), going forward."""
    n = len(keys)
    out = [a]
    while out[-1] != b:
        out.append((out[-1] + 1) % n)
    return out


def trailing_edge_cap(patch_keys, xy, P, n_sp, cavity, h):
    """Find where to cut the trailing-edge wedge behind *cavity*.

    The cap joins the first upper / lower station pair behind the cavity
    tip (by at least ``h / 2``).  Returns ``(pos_upper, pos_lower, pos_te, cap_points)`` (boundary
    positions and the interior cap points) or ``None`` when the patch does
    not contain the trailing edge.
    """
    te = ("p", n_sp - 1)
    if te not in patch_keys:
        return None
    chord = P[n_sp - 1] - P[0]
    chord = chord / np.linalg.norm(chord)
    t_cav = float(np.max((cavity - P[0]) @ chord))
    # first station (upper index j, lower 2n_sp-2-j) at least half a skin
    # segment behind the cavity tip, so the block around the tip is not flat
    best = None
    for j in range(n_sp - 2, 0, -1):
        ku, kl = ("p", j), ("p", 2 * n_sp - 2 - j)
        if (ku in patch_keys and kl in patch_keys
                and (P[j] - P[0]) @ chord > t_cav + 0.5 * h):
            best = (ku, kl)
    if best is None:
        return None
    ku, kl = best
    pu, pl, pt = patch_keys.index(ku), patch_keys.index(kl), patch_keys.index(te)
    N = len(patch_keys)
    if (pt - pl) % N < (pu - pl) % N:   # CCW ring: lower → TE → upper
        pu, pl = pl, pu
    if not ((pt - pu) % N < (pl - pu) % N):
        return None
    cap_len = np.linalg.norm(xy[pl] - xy[pu])
    nc = max(2, int(round(cap_len / h)))
    cap_mid = [(1 - s / nc) * xy[pu] + (s / nc) * xy[pl] for s in range(1, nc)]
    return pu, pl, pt, cap_mid


def inner_corner_params(cavity: np.ndarray, xy: np.ndarray,
                        corners: list[int]) -> list[float]:
    """Arc-length positions on the *cavity* ring matching outer *corners*.

    Each outer corner is first projected onto the nearest cavity point; if
    those points are not in the same cyclic order as the corners, the corners
    are pushed inwards along their angle bisectors instead; if neither is
    ordered, the positions are spread evenly so the O-grid blocks never
    cross.  Coincident positions are separated by 2 % of the cavity length.
    """
    from shapely.geometry import LineString, LinearRing, Point

    ring = LinearRing(cavity)
    L = ring.length
    n = len(xy)
    reach = 4.0 * float(np.ptp(np.vstack([xy, cavity]), axis=0).max())

    def bisector_hit(c):
        p = xy[c]
        a = xy[c - 1] - p
        b = xy[(c + 1) % n] - p
        a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
        t = b - a
        normal = np.array([-t[1], t[0]]) / max(np.linalg.norm(t), 1e-14)
        bis = a + b
        v = bis / np.linalg.norm(bis) if np.linalg.norm(bis) > 0.3 else normal
        if v @ normal < 0:
            v = -v
        hit = LineString([p, p + reach * v]).intersection(ring)
        pts = [np.asarray(g.coords[0]) for g in getattr(hit, "geoms", [hit])
               if not g.is_empty]
        if not pts:
            return ring.project(Point(*p))
        return ring.project(Point(*min(pts, key=lambda s: np.linalg.norm(s - p))))

    def ordered(d):
        """*d* made strictly increasing (cyclically, gaps >= 2 % of L) by
        separating ties, or ``None`` if the corners are out of order."""
        eps = 0.02 * L
        u = [d[0]]
        for x in d[1:]:
            r = (x - d[0]) % L               # unwrap relative to the first
            x = d[0] + (r if r > 1e-9 else L)  # a tie with d[0] sits at +L
            if x < u[-1] - 1e-9 and u[-1] - x > eps:   # went backwards
                return None
            u.append(max(x, u[-1] + eps))
        if u[-1] > d[0] + L - eps:          # tie across the wrap
            u[-1] = d[0] + L - eps
            if len(u) > 1 and u[-1] < u[-2] + eps:
                return None
        return [x % L for x in u]

    for d in ([ring.project(Point(*xy[c])) for c in corners],
              [bisector_hit(c) for c in corners]):
        fixed = ordered(d)
        if fixed is not None:
            return fixed
    return [(d[0] + L * k / len(corners)) % L for k in range(len(corners))]
