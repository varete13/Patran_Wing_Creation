"""Option A — structured rib-face mesher in pure numpy / shapely.

Every patch is meshed with discrete transfinite interpolation (Coons patch):

* **Solid, four corners with matching opposite sides** (spar box): a fully
  mapped ``n_skin × n_web`` grid.
* **Solid, three corners** (leading-edge nose, trailing-edge wedge): a mapped
  grid collapsed onto the apex; the apex row becomes CTRIA3.
* **Solid, anything else** (asymmetric cut sub-sections): four corners are
  chosen on the boundary so that opposite sides match, preferring the
  sharpest boundary angles.  An odd boundary gets one extra node that is
  merged afterwards, leaving a single CTRIA3.
* **Hollow**: an O-grid of ``n_wall`` layers between the outer contour and the
  cavity, one mapped block per side.  When the patch contains the trailing
  edge, the thin wedge behind the cavity is cut off by a straight cap and
  meshed as a solid apex patch.

Interior nodes are finally relaxed with a Laplacian smoother that never
accepts a move that would invalidate an element.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LinearRing

from . import RIB_MESHERS
from .quality import element_metrics, free_edges
from .topology import (
    NodeKey,
    RibMesh,
    boundary_nodes,
    inner_corner_params,
    patch_corners,
    rib_patches,
    rib_points,
    ring_side,
    signed_area,
    trailing_edge_cap,
)

if TYPE_CHECKING:
    from ..rib import Rib


# ---------------------------------------------------------------------------
# Transfinite interpolation
# ---------------------------------------------------------------------------

def _arc01(pts: np.ndarray) -> np.ndarray:
    seg = np.hypot(*np.diff(pts, axis=0).T)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    return s / s[-1] if s[-1] > 0 else np.linspace(0.0, 1.0, len(pts))


def tfi(bottom, right, top, left) -> np.ndarray:
    """Coons-patch grid ``(nu + 1, nv + 1, 2)``.

    *bottom* runs c0→c1, *right* c1→c2, *top* c3→c2 and *left* c0→c3, with
    ``len(bottom) == len(top)`` and ``len(left) == len(right)``.
    """
    B, R, T, L = (np.asarray(a, dtype=float) for a in (bottom, right, top, left))
    sb, st, sl, sr = _arc01(B), _arc01(T), _arc01(L), _arc01(R)
    nu, nv = len(B), len(L)
    X = np.empty((nu, nv, 2))
    c0, c1, c2, c3 = B[0], B[-1], T[-1], T[0]
    for i in range(nu):
        for j in range(nv):
            du, dv = st[i] - sb[i], sr[j] - sl[j]
            u = (sb[i] + sl[j] * du) / (1.0 - du * dv)
            v = sl[j] + u * dv
            X[i, j] = (
                (1 - v) * B[i] + v * T[i] + (1 - u) * L[j] + u * R[j]
                - ((1 - u) * (1 - v) * c0 + u * (1 - v) * c1
                   + u * v * c2 + (1 - u) * v * c3)
            )
    return X


def _grid_elements(mesh: RibMesh, K: list[list[NodeKey]]) -> None:
    """Add the quads of key grid *K* (``K[i][j]``), dropping collapsed nodes."""
    for i in range(len(K) - 1):
        for j in range(len(K[0]) - 1):
            quad = [K[i][j], K[i + 1][j], K[i + 1][j + 1], K[i][j + 1]]
            elem = [k for n, k in enumerate(quad) if k not in quad[:n]]
            if len(elem) < 3:
                continue
            pts = np.array([mesh.nodes[k] for k in elem])
            if signed_area(pts) < 0:
                elem = elem[::-1]
            mesh.elements.append(tuple(elem))


def _fill(mesh: RibMesh, X: np.ndarray, bottom, right, top, left) -> None:
    """Create keys for the interior of grid *X* and add its elements."""
    nu, nv = X.shape[:2]
    K: list[list[NodeKey]] = [[None] * nv for _ in range(nu)]
    for i in range(nu):
        K[i][0], K[i][-1] = bottom[i], top[i]
    for j in range(nv):
        K[0][j], K[-1][j] = left[j], right[j]
    for i in range(1, nu - 1):
        for j in range(1, nv - 1):
            K[i][j] = mesh.add_node(X[i, j])
    _grid_elements(mesh, K)


# ---------------------------------------------------------------------------
# Solid patches
# ---------------------------------------------------------------------------

def _interior_angles(xy: np.ndarray) -> np.ndarray:
    """Interior angle (deg) at each vertex of a CCW polygon."""
    a = np.roll(xy, 1, axis=0) - xy
    b = np.roll(xy, -1, axis=0) - xy
    ang = np.degrees(np.arctan2(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0],
                                (a * b).sum(axis=1)))
    return np.mod(ang, 360.0)


#: corner sets (ranked by boundary angle) whose grids are actually evaluated
_N_CANDIDATES = 60


def _grid_score(X: np.ndarray) -> tuple[int, float]:
    """``(invalid elements, -min angle)`` of a grid (lower is better).

    Corners with a zero-length edge (collapsed apex rows) are ignored, as
    :func:`_grid_elements` drops them when it writes the elements.
    """
    Q = np.stack([X[:-1, :-1], X[1:, :-1], X[1:, 1:], X[:-1, 1:]], axis=2)
    a = np.roll(Q, -1, axis=2) - Q       # to next corner
    b = np.roll(Q, 1, axis=2) - Q        # to previous corner
    la, lb = np.linalg.norm(a, axis=-1), np.linalg.norm(b, axis=-1)
    live = (la > 1e-12) & (lb > 1e-12)
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    cosang = np.where(live, (a * b).sum(-1) / np.where(live, la * lb, 1.0), -1.0)
    ang = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
    bad = np.any(live & (cross <= 0), axis=-1)
    amin = float(np.min(np.where(live, ang, 180.0)))
    return int(bad.sum()), -amin


def _mapped_sides(keys, c: list[int]):
    """Boundary positions of the four sides for corners ``c0..c3`` (CCW)."""
    c0, c1, c2, c3 = c
    return (ring_side(keys, c0, c1), ring_side(keys, c1, c2),
            ring_side(keys, c2, c3)[::-1], ring_side(keys, c3, c0)[::-1])


def _apex_sides(keys, apex: int, c_next: int, c_prev: int):
    """Sides of a grid collapsed onto *apex* (base = c_next → c_prev)."""
    base = ring_side(keys, c_next, c_prev)
    return (ring_side(keys, apex, c_next), base,
            ring_side(keys, c_prev, apex)[::-1], [apex] * len(base))


class _Candidate:
    """A mapped grid over a (possibly augmented) boundary."""

    def __init__(self, kind, keys, xy, sides, virtual=None):
        self.kind, self.keys, self.xy, self.sides = kind, keys, xy, sides
        self.virtual = virtual  # (virtual key, key it merges into) or None
        self.X = tfi(*(xy[s] for s in sides))
        self.score = _grid_score(self.X)

    def apply(self, mesh: RibMesh) -> str:
        if self.virtual is not None:
            mesh.nodes[self.virtual[0]] = self.xy[self.keys.index(self.virtual[0])]
        _fill(mesh, self.X, *([self.keys[p] for p in s] for s in self.sides))
        if self.virtual is not None:
            _merge_node(mesh, *self.virtual)
        return self.kind


def _search_candidates(keys, xy) -> list[_Candidate]:
    """Four corners anywhere on the boundary with matching opposite sides.

    An odd boundary gets a virtual mid-node on its longest edge, merged into
    a neighbour afterwards (one CTRIA3 remains).
    """
    keys, xy = list(keys), np.asarray(xy)
    virtual = None
    if len(keys) % 2:
        seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
        k = int(np.argmax(seg))
        vkey = ("v", k, tuple(np.round(xy[k], 12)))
        mid = 0.5 * (xy[k] + xy[(k + 1) % len(xy)])
        virtual = (vkey, keys[k])  # merged into the lower-indexed neighbour
        keys.insert(k + 1, vkey)
        xy = np.insert(xy, k + 1, mid, axis=0)
    n, h = len(keys), len(keys) // 2
    w = _interior_angles(xy)
    # never put a grid corner inside a straight web edge or on the virtual node
    for p, k in enumerate(keys):
        if k[0] in ("w", "v"):
            w[p] = np.inf
    ranked = sorted(
        (w[c0] + w[c0 + d] + w[c0 + h] + w[(c0 + d + h) % n],
         [c0, c0 + d, c0 + h, (c0 + d + h) % n])
        for c0 in range(h) for d in range(1, h)
    )
    return [
        _Candidate("search", keys, xy, _mapped_sides(keys, c), virtual)
        for cost, c in ranked[:_N_CANDIDATES] if np.isfinite(cost)
    ]


def _merge_node(mesh: RibMesh, old: NodeKey, new: NodeKey) -> None:
    out = []
    for e in mesh.elements:
        if old in e:
            e = tuple(new if k == old else k for k in e)
            e = tuple(k for n, k in enumerate(e) if k not in e[:n])
            if len(e) < 3:
                continue
        out.append(e)
    mesh.elements = out
    del mesh.nodes[old]


def _mesh_solid(mesh: RibMesh, keys, xy, corners: list[int]) -> str:
    """Mesh a solid patch; returns the strategy used."""
    keys, xy = list(keys), np.asarray(xy)
    n, m = len(keys), len(corners)
    cnt = [(corners[(k + 1) % m] - corners[k]) % n for k in range(m)]
    if m == 4 and cnt[0] == cnt[2] and cnt[1] == cnt[3]:
        return _Candidate("mapped", keys, xy, _mapped_sides(keys, corners)).apply(mesh)
    cands: list[_Candidate] = []
    if m == 3:
        for k in range(3):
            if cnt[k - 1] == cnt[k]:  # equal sides into and out of corner k
                cands.append(_Candidate(
                    "apex", keys, xy,
                    _apex_sides(keys, corners[k], corners[(k + 1) % 3],
                                corners[k - 1]),
                ))
    cands += _search_candidates(keys, xy)
    # fewest invalid elements, then largest minimum angle (to the degree);
    # apex grids win ties
    return min(
        cands,
        key=lambda c: (c.score[0], round(c.score[1]), c.kind != "apex"),
    ).apply(mesh)


# ---------------------------------------------------------------------------
# Hollow patches (O-grid)
# ---------------------------------------------------------------------------

def _resample_like(ring: LinearRing, d0: float, d1: float, ref: np.ndarray) -> np.ndarray:
    """Points on *ring* from arc *d0* to *d1* spaced like polyline *ref*."""
    L = ring.length
    if d1 <= d0:
        d1 += L
    s = d0 + _arc01(ref) * (d1 - d0)
    return np.array([ring.interpolate(t % L).coords[0] for t in s])


def _mesh_ogrid(mesh: RibMesh, keys, xy, corners: list[int], cavity: np.ndarray,
                n_wall: int) -> None:
    xy = np.asarray(xy)
    ring = LinearRing(cavity)
    d = inner_corner_params(cavity, xy, corners)
    radial: dict[int, list[NodeKey]] = {}
    for c, dc in zip(corners, d):
        pin = np.array(ring.interpolate(dc).coords[0])
        col = [keys[c]]
        for s in range(1, n_wall + 1):
            t = s / n_wall
            col.append(mesh.add_node((1 - t) * xy[c] + t * pin))
        radial[c] = col
    m = len(corners)
    for k in range(m):
        a, b = corners[k], corners[(k + 1) % m]
        outer = ring_side(keys, a, b)
        pts_in = _resample_like(ring, d[k], d[(k + 1) % m], xy[outer])
        inner_keys = [radial[a][-1]] + [mesh.add_node(p) for p in pts_in[1:-1]] \
            + [radial[b][-1]]
        X = tfi(xy[outer], [mesh.nodes[q] for q in radial[b]],
                [mesh.nodes[q] for q in inner_keys],
                [mesh.nodes[q] for q in radial[a]])
        _fill(mesh, X, [keys[p] for p in outer], radial[b], inner_keys, radial[a])


def _mesh_hollow(mesh, keys, xy, corners, cavity, P, n_sp, n_wall) -> str:
    xy = np.asarray(xy)
    seg = np.hypot(*np.diff(np.vstack([xy, xy[:1]]), axis=0).T)
    tail = trailing_edge_cap(keys, xy, P, n_sp, cavity, float(np.median(seg)))
    if tail is None:
        _mesh_ogrid(mesh, keys, xy, corners, cavity, n_wall)
        return "o-grid"
    pu, pl, pt, cap_mid = tail
    cap_keys = [mesh.add_node(p) for p in cap_mid]
    # tail: pu → … → te → … → pl, then the cap back to pu
    tail_pos = ring_side(keys, pu, pl)
    t_keys = [keys[p] for p in tail_pos] + cap_keys[::-1]
    t_xy = np.vstack([xy[tail_pos], np.array(cap_mid[::-1]).reshape(-1, 2)])
    t_c = [0, tail_pos.index(pt), len(tail_pos) - 1]
    _mesh_solid(mesh, t_keys, t_xy, t_c)
    # main: pl → … → pu, then the cap from pu to pl
    main_pos = ring_side(keys, pl, pu)
    m_keys = [keys[p] for p in main_pos] + cap_keys
    m_xy = np.vstack([xy[main_pos], np.array(cap_mid).reshape(-1, 2)])
    m_c = sorted({main_pos.index(c) for c in corners if c in main_pos and c != pt}
                 | {0, len(main_pos) - 1})
    _mesh_ogrid(mesh, m_keys, m_xy, m_c, cavity, n_wall)
    return "o-grid+tail"


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

def _scaled_jacobian(pts: np.ndarray) -> float:
    """Minimum corner ``sin(angle)`` of a CCW polygon; negative if inverted."""
    k = len(pts)
    best = 1.0
    for i in range(k):
        a = pts[(i + 1) % k] - pts[i]
        b = pts[i - 1] - pts[i]
        den = np.linalg.norm(a) * np.linalg.norm(b)
        best = min(best, (a[0] * b[1] - a[1] * b[0]) / den if den > 0 else -1.0)
    return best


def smooth(mesh: RibMesh, movable: set, iterations: int = 20) -> None:
    """Relax *movable* nodes, then untangle any remaining poor elements.

    1. Laplacian smoothing; a move is rejected if it would invalidate one of
       the node's elements.
    2. Nodes of elements whose worst corner has ``sin(angle) < 0.2`` are
       moved to the best point of a small stencil (maximising the worst
       corner of their elements), which also untangles inverted elements.
    """
    nbr: dict = {k: set() for k in movable}
    elems_of: dict = {k: [] for k in movable}
    for e in mesh.elements:
        for i, k in enumerate(e):
            if k in movable:
                nbr[k].update({e[i - 1], e[(i + 1) % len(e)]})
                elems_of[k].append(e)
    nodes = mesh.nodes

    def local_q(k):
        return min(_scaled_jacobian(np.array([nodes[q] for q in e]))
                   for e in elems_of[k])

    order = sorted(movable, key=str)
    for _ in range(iterations):
        for k in order:
            if not nbr[k]:
                continue
            old, q_old = nodes[k], local_q(k)
            nodes[k] = np.mean([nodes[q] for q in nbr[k]], axis=0)
            if q_old > 0 and local_q(k) <= 0:
                nodes[k] = old

    for _ in range(iterations):
        poor = {k for e in mesh.elements
                if _scaled_jacobian(np.array([nodes[q] for q in e])) < 0.2
                for k in e if k in movable and nbr[k]}
        if not poor:
            break
        for k in sorted(poor, key=str):
            old, q_best = nodes[k], local_q(k)
            best = old
            h = 0.2 * np.mean([np.linalg.norm(nodes[q] - old) for q in nbr[k]])
            for t in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                for r in (h, 2 * h):
                    nodes[k] = old + r * np.array([np.cos(t), np.sin(t)])
                    q = local_q(k)
                    if q > q_best:
                        best, q_best = nodes[k], q
            nodes[k] = best


def split_bad_quads(mesh: RibMesh, max_angle: float = 175.0) -> int:
    """Split quads that are non-convex or nearly flat at one corner into two
    CTRIA3 through that corner.  Returns the number of quads split."""
    out, n_split = [], 0
    for e in mesh.elements:
        if len(e) == 4:
            pts = np.array([mesh.nodes[k] for k in e])
            valid, _, a_max, _, area = element_metrics(pts)
            if not valid or a_max > max_angle:
                ang = _interior_angles(pts) if area > 0 else None
                if ang is not None:
                    i = int(np.argmax(ang))
                    t1 = (e[i], e[(i + 1) % 4], e[(i + 2) % 4])
                    t2 = (e[i], e[(i + 2) % 4], e[(i + 3) % 4])
                    if all(signed_area(np.array([mesh.nodes[k] for k in t])) > 0
                           for t in (t1, t2)):
                        out += [t1, t2]
                        n_split += 1
                        continue
        out.append(e)
    mesh.elements = out
    return n_split


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def mesh_native(rib: Rib, n_web: int = 4, n_wall: int = 2) -> RibMesh:
    """Structured rib-face mesh without external meshing libraries."""
    P = rib_points(rib)
    n_sp = rib.n_skin_points
    mesh = RibMesh()
    fixed: set = set()
    for patch in rib_patches(rib):
        keys, xy, ring_pos = boundary_nodes(patch, P, n_web)
        xy = np.asarray(xy)
        for k, p in zip(keys, xy):
            mesh.nodes[k] = p
        fixed.update(keys)
        corners = [ring_pos[c] for c in patch_corners(patch, P, n_sp)]
        if patch.inner is None:
            kind = _mesh_solid(mesh, keys, xy, corners)
        else:
            kind = _mesh_hollow(mesh, keys, xy, corners, patch.inner, P, n_sp,
                                max(1, n_wall))
        mesh.info.append(f"{patch.name}: {kind}")
    # nodes on the cavity contours are on free edges and must not move
    on_boundary = {k for edge in free_edges(mesh) for k in edge}
    movable = {k for k in mesh.nodes if k not in fixed and k not in on_boundary}
    smooth(mesh, movable)
    n_split = split_bad_quads(mesh)
    if n_split:
        mesh.info.append(f"{n_split} distorted quad(s) split into CTRIA3")
    return mesh


RIB_MESHERS["native"] = mesh_native
