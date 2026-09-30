from __future__ import annotations

import dataclasses


@dataclasses.dataclass
class IDScheme:
    """Base IDs and strides for Nastran/Patran entity numbering."""

    point_base: int = 100_000
    point_stride: int = 1_000

    surface_rib_base: int = 1_000
    surface_rib_stride: int = 10

    line_base: int = 0

    surface_spar_base: int = 2_000
    surface_spar_stride: int = 10

    surface_skin_base: int = 30_000
    surface_skin_stride: int = 100

    inner_point_base: int = 200_000
    inner_point_stride: int = 1_000

    surface_inner_base: int = 50_000
    surface_inner_stride: int = 10


@dataclasses.dataclass
class RibIDs:
    """Pre-allocated ID ranges for a single rib."""

    rib_index: int
    point_start: int
    point_end: int
    surface_base: int
    inner_point_start: int = 0
    inner_surface_base: int = 0


class IDManager:
    """Instance-owned ID allocator for a Wing.

    Replaces the class-level mutable counters in the original ``Ala`` class
    and the recursive ``check_name_basis`` with an iterative validator.
    """

    def __init__(
        self,
        n_ribs: int,
        n_skin_points: int,
        n_spars: int,
        has_inner: bool = False,
        n_cuts_per_rib: int = 0,
        scheme: IDScheme | None = None,
    ) -> None:
        # Copy so that widening never mutates a scheme shared by the caller
        self.scheme = dataclasses.replace(scheme) if scheme else IDScheme()
        self.n_skin_points = n_skin_points
        self.has_inner = has_inner
        self._validate_and_adjust(n_ribs, n_skin_points, n_spars, n_cuts_per_rib)
        self._rib_counter = 0

    def _validate_and_adjust(
        self, n_ribs: int, n_skin_points: int, n_spars: int, n_cuts_per_rib: int = 0
    ) -> None:
        """Widen strides and bases until every ID range is collision-free.

        Rib indices run from ``1`` to ``n_ribs`` (see :meth:`allocate_rib`),
        so a per-rib range ending at ``base + stride * n_ribs + stride - 1``
        stays below ``2 * base`` only when ``base >= stride * (n_ribs + 1)``.
        Spanwise entities (spar and skin panels) use the index of the first
        rib of each bay, which runs from ``1`` to ``n_ribs - 1``.

        After the per-range checks, ranges of the same Patran entity type
        (points, surfaces) are checked against each other and the base of
        the higher range is widened on overlap.
        """
        s = self.scheme
        # Maximum surfaces per rib: standard sections + one extra per cut
        max_surfaces_per_rib = n_spars + 1 + n_cuts_per_rib
        n_bays = max(n_ribs - 1, 1)
        changed = True
        while changed:
            changed = False
            if s.point_stride < 2 * n_skin_points:
                s.point_stride *= 10
                changed = True
            if s.point_base < s.point_stride * (n_ribs + 1):
                s.point_base *= 10
                changed = True
            if s.surface_rib_stride < max_surfaces_per_rib:
                s.surface_rib_stride *= 10
                changed = True
            if s.surface_rib_base < s.surface_rib_stride * (n_ribs + 1):
                s.surface_rib_base *= 10
                changed = True
            if s.surface_spar_stride < n_spars:
                s.surface_spar_stride *= 10
                changed = True
            if s.surface_spar_base < s.surface_spar_stride * (n_bays + 1):
                s.surface_spar_base *= 10
                changed = True
            if s.surface_skin_stride < 2 * n_skin_points:
                s.surface_skin_stride *= 10
                changed = True
            if s.surface_skin_base < s.surface_skin_stride * (n_bays + 1):
                s.surface_skin_base *= 10
                changed = True
            # Inner contour ID ranges
            if s.inner_point_stride < 2 * n_skin_points:
                s.inner_point_stride *= 10
                changed = True
            if s.inner_point_base < s.inner_point_stride * (n_ribs + 1):
                s.inner_point_base *= 10
                changed = True
            if s.surface_inner_stride < n_spars + 1:
                s.surface_inner_stride *= 10
                changed = True
            if s.surface_inner_base < s.surface_inner_stride * (n_ribs + 1):
                s.surface_inner_base *= 10
                changed = True
            if self._widen_overlapping_ranges(n_ribs, n_bays):
                changed = True

    def _widen_overlapping_ranges(self, n_ribs: int, n_bays: int) -> bool:
        """Widen one base if two ranges of the same entity type overlap.

        Returns ``True`` when a base was changed, so the caller re-validates.
        """
        s = self.scheme
        families = {
            "points": [
                ("point_base", s.point_base, s.point_stride, n_ribs),
                ("inner_point_base", s.inner_point_base, s.inner_point_stride, n_ribs),
            ],
            "surfaces": [
                ("surface_rib_base", s.surface_rib_base, s.surface_rib_stride, n_ribs),
                ("surface_spar_base", s.surface_spar_base, s.surface_spar_stride, n_bays),
                ("surface_skin_base", s.surface_skin_base, s.surface_skin_stride, n_bays),
                ("surface_inner_base", s.surface_inner_base, s.surface_inner_stride, n_ribs),
            ],
        }
        for ranges in families.values():
            # (first id, last id, attribute name) of every range, sorted
            spans = sorted(
                (base + stride, base + stride * (n_max + 1) - 1, name)
                for name, base, stride, n_max in ranges
            )
            for (_, hi, _), (lo_next, _, name_next) in zip(spans, spans[1:]):
                if lo_next <= hi:
                    setattr(s, name_next, getattr(s, name_next) * 10)
                    return True
        return False

    def allocate_rib(self) -> RibIDs:
        """Return ID ranges for the next rib and advance the counter."""
        self._rib_counter += 1
        idx = self._rib_counter
        s = self.scheme
        point_start = s.point_base + s.point_stride * idx
        point_end = point_start + 2 * self.n_skin_points - 3
        surface_base = s.surface_rib_base + s.surface_rib_stride * idx
        inner_point_start = s.inner_point_base + s.inner_point_stride * idx
        inner_surface_base = s.surface_inner_base + s.surface_inner_stride * idx
        return RibIDs(
            rib_index=idx,
            point_start=point_start,
            point_end=point_end,
            surface_base=surface_base,
            inner_point_start=inner_point_start,
            inner_surface_base=inner_surface_base,
        )
