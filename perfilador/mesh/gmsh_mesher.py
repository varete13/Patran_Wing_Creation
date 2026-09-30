"""Option B — rib-face mesher built on gmsh.

The rib face is rebuilt as a 2-D gmsh model from the *same* boundary nodes
used everywhere else (profile nodes and web-edge nodes are gmsh points joined
by one-element lines), so the result is node-compatible with the skin panels
and the spar webs.  Per patch:

* **Solid, mappable** (four corners with matching opposite sides, or three
  corners with matching sides at the apex): transfinite surface → structured.
* **Solid, not mappable** (asymmetric cut sub-sections): gmsh's
  Frontal-Delaunay for quads + Blossom recombination → quad-dominant.
* **Hollow**: O-grid of transfinite blocks between the outer contour and the
  cavity (cavity sides are splines), with the trailing-edge wedge behind the
  cavity split off as a solid patch.

Requires the ``gmsh`` Python package (``pip install gmsh``).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from shapely.geometry import LinearRing

from . import RIB_MESHERS
from .topology import (
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

#: points sampled along each cavity side to build its spline
_SPLINE_SAMPLES = 24


class _RibModel:
    """Incremental builder of the gmsh geometry of one rib face."""

    def __init__(self, gmsh, mesh: RibMesh, h: float):
        self.gmsh, self.geo, self.mesh, self.h = gmsh, gmsh.model.geo, mesh, h
        self.point_of: dict = {}   # node key -> gmsh point tag
        self.key_of: dict = {}     # gmsh point tag -> node key
        self.lines: dict = {}      # (key_a, key_b) -> line tag
        self.surfaces: list[tuple[int, list[int] | None]] = []

    def point(self, key, xy) -> int:
        if key not in self.point_of:
            t = self.geo.addPoint(float(xy[0]), float(xy[1]), 0.0, self.h)
            self.point_of[key], self.key_of[t] = t, key
            self.mesh.nodes.setdefault(key, np.asarray(xy, dtype=float))
        return self.point_of[key]

    def line(self, ka, kb, n_elem: int = 1) -> int:
        """Straight line ka→kb (shared, signed tag) with *n_elem* elements."""
        if (ka, kb) in self.lines:
            return self.lines[(ka, kb)]
        if (kb, ka) in self.lines:
            return -self.lines[(kb, ka)]
        t = self.geo.addLine(self.point_of[ka], self.point_of[kb])
        self.geo.mesh.setTransfiniteCurve(t, n_elem + 1)
        self.lines[(ka, kb)] = t
        return t

    def polyline(self, keys: list) -> list[int]:
        return [self.line(a, b) for a, b in zip(keys, keys[1:])]

    def surface(self, loop: list[int], corners: list | None) -> int:
        s = self.geo.addPlaneSurface([self.geo.addCurveLoop(loop)])
        self.surfaces.append((s, corners))
        return s

    # -- regions -----------------------------------------------------------

    def solid(self, keys: list, xy: np.ndarray, corners: list[int]) -> str:
        n, m = len(keys), len(corners)
        for k, p in zip(keys, xy):
            self.point(k, p)
        loop = self.polyline(keys + keys[:1])
        cnt = [(corners[(k + 1) % m] - corners[k]) % n for k in range(m)]
        tf = None
        if m == 4 and cnt[0] == cnt[2] and cnt[1] == cnt[3]:
            tf, kind = [keys[c] for c in corners], "transfinite"
        elif m == 3:
            # apex grids only when the base is not longer than the sides,
            # otherwise the apex becomes a fan of slivers
            apex = [k for k in range(3)
                    if cnt[k - 1] == cnt[k] and cnt[(k + 1) % 3] <= cnt[k]]
            if apex:
                a = apex[0]
                tf = [keys[corners[a]], keys[corners[(a + 1) % 3]],
                      keys[corners[a - 1]]]
                kind = "transfinite-apex"
        if tf is None:
            kind = "quad-unstructured"
        self.surface(loop, tf)
        return kind

    def ogrid(self, keys: list, xy: np.ndarray, corners: list[int],
              cavity: np.ndarray, n_wall: int) -> None:
        for k, p in zip(keys, xy):
            self.point(k, p)
        ring = LinearRing(cavity)
        L = ring.length
        d = inner_corner_params(cavity, xy, corners)
        inner = {}
        for c, dc in zip(corners, d):
            key = self.mesh.add_node(ring.interpolate(dc).coords[0])
            self.point(key, self.mesh.nodes[key])
            inner[c] = key
        radial = {c: self.line(keys[c], inner[c], n_wall) for c in corners}
        m = len(corners)
        for k in range(m):
            a, b = corners[k], corners[(k + 1) % m]
            side = ring_side(keys, a, b)
            d0, d1 = d[k], d[(k + 1) % m]
            if d1 <= d0:
                d1 += L
            mids = []
            for t in np.linspace(d0, d1, _SPLINE_SAMPLES)[1:-1]:
                mids.append(self.geo.addPoint(
                    *ring.interpolate(t % L).coords[0], 0.0, self.h))
            arc = self.geo.addSpline(
                [self.point_of[inner[a]]] + mids + [self.point_of[inner[b]]])
            self.geo.mesh.setTransfiniteCurve(arc, len(side))
            loop = (self.polyline([keys[p] for p in side])
                    + [radial[b], -arc, -radial[a]])
            self.surface(loop, [keys[a], keys[b], inner[b], inner[a]])


def _generate(gmsh, model: _RibModel, transfinite: bool) -> None:
    geo = gmsh.model.geo
    for s, corners in model.surfaces:
        if transfinite and corners is not None:
            geo.mesh.setTransfiniteSurface(
                s, cornerTags=[model.point_of[k] for k in corners])
        geo.mesh.setRecombine(2, s)
    geo.synchronize()
    gmsh.model.mesh.generate(2)


def mesh_gmsh(rib: Rib, n_web: int = 4, n_wall: int = 2) -> RibMesh:
    """Rib-face mesh generated with gmsh (see module docstring)."""
    try:
        import gmsh
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise ImportError(
            "rib_mesher='gmsh' needs the gmsh package: pip install gmsh"
        ) from exc

    P = rib_points(rib)
    n_sp = rib.n_skin_points
    seg = np.hypot(*np.diff(np.vstack([P, P[:1]]), axis=0).T)
    h = float(np.median(seg))

    if not gmsh.isInitialized():
        gmsh.initialize(readConfigFiles=False, interruptible=False)
    gmsh.option.setNumber("General.Terminal", 0)
    for name, value in {
        "Mesh.Algorithm": 8,             # Frontal-Delaunay for quads
        "Mesh.RecombinationAlgorithm": 1,  # Blossom (keeps boundary nodes)
        "Mesh.RecombineAll": 1,
        "Mesh.Smoothing": 10,
        "Mesh.MeshSizeFromPoints": 1,
        "Mesh.MeshSizeExtendFromBoundary": 1,
    }.items():
        gmsh.option.setNumber(name, value)

    for transfinite in (True, False):
        gmsh.clear()
        gmsh.model.add(f"rib_{rib.ids.rib_index}")
        mesh = RibMesh()
        model = _RibModel(gmsh, mesh, h)
        for patch in rib_patches(rib):
            keys, xy, ring_pos = boundary_nodes(patch, P, n_web)
            xy = np.asarray(xy)
            corners = [ring_pos[c] for c in patch_corners(patch, P, n_sp)]
            if patch.inner is None:
                kind = model.solid(keys, xy, corners)
            else:
                cap = trailing_edge_cap(keys, xy, P, n_sp, patch.inner, h)
                if cap is None:
                    model.ogrid(keys, xy, corners, patch.inner, max(1, n_wall))
                    kind = "o-grid"
                else:
                    pu, pl, pt, cap_mid = cap
                    cap_keys = [mesh.add_node(p) for p in cap_mid]
                    for k, p in zip(keys, xy):
                        mesh.nodes.setdefault(k, p)
                    tail = ring_side(keys, pu, pl)
                    t_keys = [keys[p] for p in tail] + cap_keys[::-1]
                    t_xy = np.array([mesh.nodes[k] for k in t_keys])
                    model.solid(t_keys, t_xy, [0, tail.index(pt), len(tail) - 1])
                    main = ring_side(keys, pl, pu)
                    m_keys = [keys[p] for p in main] + cap_keys
                    m_xy = np.array([mesh.nodes[k] for k in m_keys])
                    m_c = sorted({main.index(c) for c in corners
                                  if c in main and c != pt}
                                 | {0, len(main) - 1})
                    model.ogrid(m_keys, m_xy, m_c, patch.inner, max(1, n_wall))
                    kind = "o-grid+tail"
            mesh.info.append(f"{patch.name}: {kind}")
        try:
            _generate(gmsh, model, transfinite)
            break
        except Exception as exc:  # transfinite constraints rejected
            if not transfinite:
                raise
            mesh.info.append(f"transfinite failed ({exc}); unstructured retry")

    # -- read the mesh back -----------------------------------------------
    key_of_node: dict[int, object] = {}
    for dim, tag in gmsh.model.getEntities(0):
        ntags, _, _ = gmsh.model.mesh.getNodes(dim, tag)
        if tag in model.key_of and len(ntags):
            key_of_node[int(ntags[0])] = model.key_of[tag]
    ntags, coords, _ = gmsh.model.mesh.getNodes()
    xyz = coords.reshape(-1, 3)
    for t, c in zip(ntags, xyz):
        if int(t) not in key_of_node:
            key_of_node[int(t)] = mesh.add_node(c[:2])
    etypes, _, enodes = gmsh.model.mesh.getElements(2)
    for et, en in zip(etypes, enodes):
        nper = {2: 3, 3: 4}.get(int(et))
        if nper is None:
            continue
        for conn in en.reshape(-1, nper):
            elem = tuple(key_of_node[int(t)] for t in conn)
            if signed_area(np.array([mesh.nodes[k] for k in elem])) < 0:
                elem = elem[::-1]
            mesh.elements.append(elem)
    gmsh.clear()

    # gmsh nodes on the web / profile points are the fixed boundary nodes
    used = {k for e in mesh.elements for k in e}
    mesh.nodes = {k: v for k, v in mesh.nodes.items() if k in used}
    return mesh


RIB_MESHERS["gmsh"] = mesh_gmsh
