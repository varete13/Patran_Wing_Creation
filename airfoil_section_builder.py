"""Visor interactivo y motor de cortes de secciones de ala.

Muestra las 3 secciones (LE, Box, TE) con los IDs de punto de Nastran.
Permite crear parejas de puntos dentro de la misma seccion para
definir un corte, obtener la lista de IDs de cada subseccion resultante
y visualizar el buffer interior con Shapely.

Uso programatico:
  wing = WingSection(...)
  wing.cut_sections()                               # visor interactivo
  wing.cut_sections([(id_a, id_b), ...])            # mismos cortes en todas las costillas
  wing.cut_sections({0: [...], 1: [...]})           # cortes distintos por costilla
  wing.apply_cut_cavities([(id_a, id_b), ...])      # cavidades interiores por subseccion

Controles del visor:
  - Click izquierdo : seleccionar punto (snap al mas cercano)
  - Enter           : aplicar cortes y mostrar subsecciones
  - S               : exportar diccionario de cortes para todas las costillas
  - R               : reiniciar seleccion
  - Q               : cerrar
"""
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon

from perfilador import NACAAirfoil, Rib, Wing, offset_polygon

SECTION_NAMES  = ["LE", "Box", "TE"]
SECTION_COLORS = ["steelblue", "seagreen", "tomato"]


# ======================================================================== #
#  WingSection — lógica de corte                                           #
# ======================================================================== #

class WingSection(Wing):
    """Wing con capacidades de corte de secciones, interactivo o programatico.

    Hereda de Wing sin modificar su constructor; solo añade métodos de corte.
    """

    def cut_sections(
        self,
        cuts: list[tuple[int, int]] | dict[int, list[tuple[int, int]]] | None = None,
        rib_index: int = 0,
        wall_thickness: float = 0.15,
    ) -> dict[int, dict[str, list[list[int]]]] | None:
        """Punto de entrada unificado para definir subsecciones.

        Parameters
        ----------
        cuts : None
            Abre el RibViewer interactivo para definir cortes con el raton.
        cuts : list of (id_a, id_b)
            Pares de IDs Nastran del rib ``rib_index``.  Se convierten a
            indices locales y se propagan a TODAS las costillas.
        cuts : dict {rib_idx: [(id_a, id_b), ...]}
            Pares de IDs Nastran especificos por costilla.

        Returns
        -------
        None (modo interactivo) o dict
            {rib_idx: {"LE": [[id, ...], ...], "Box": [...], "TE": [...]}}
        """
        if cuts is None:
            RibViewer(self, rib_index=rib_index, wall_thickness=wall_thickness)
            return None

        if isinstance(cuts, list):
            ref_rib = self.ribs[rib_index]
            pairs_local = [
                (id_a - ref_rib.ids.point_start, id_b - ref_rib.ids.point_start)
                for id_a, id_b in cuts
            ]
            return {
                ridx: self._compute_rib_cuts(rib, pairs_local)
                for ridx, rib in enumerate(self.ribs)
            }

        # dict case: per-rib Nastran IDs
        return {
            ridx: self._compute_rib_cuts(
                self.ribs[ridx],
                [
                    (
                        id_a - self.ribs[ridx].ids.point_start,
                        id_b - self.ribs[ridx].ids.point_start,
                    )
                    for id_a, id_b in rib_cuts
                ],
            )
            for ridx, rib_cuts in cuts.items()
        }

    def _compute_rib_cuts(
        self,
        rib: Rib,
        pairs_local: list[tuple[int, int]],
    ) -> dict[str, list[list[int]]]:
        """Aplica cortes a una costilla y devuelve IDs Nastran por subseccion.

        Parameters
        ----------
        rib : Rib
            Costilla a cortar.
        pairs_local : list of (idx_a, idx_b)
            Pares como indices en el array de perfil (no IDs Nastran).

        Returns
        -------
        dict {"LE": [[id,...], ...], "Box": [...], "TE": [...]}
            Una lista de subsecciones por seccion original; cada subseccion
            es una lista de IDs Nastran.
        """
        px, py = rib.profile
        if np.allclose(px[0], px[-1]) and np.allclose(py[0], py[-1]):
            px, py = px[:-1], py[:-1]
        n = len(px)
        ids = np.arange(rib.ids.point_start, rib.ids.point_start + n)

        si = [sid - rib.ids.point_start for sid in rib.spar_ids]
        i0, i1, i2, i3 = si[0], si[1], si[2], si[3]
        section_indices = [
            list(range(0, i0 + 1)) + list(range(i3, n)),
            list(range(i0, i1 + 1)) + list(range(i2, i3 + 1)),
            list(range(i1, i2 + 1)),
        ]
        assigned: dict[int, int] = {}
        for s, pts in enumerate(section_indices):
            for k in pts:
                if k not in assigned:
                    assigned[k] = s

        pools: dict[int, list[list[int]]] = {
            s: [list(idx)] for s, idx in enumerate(section_indices)
        }
        for ia, ib in pairs_local:
            sec_a = assigned.get(ia)
            sec_b = assigned.get(ib)
            if sec_a is None or sec_b is None or sec_a != sec_b or ia == ib:
                continue
            pool = pools[sec_a]
            for i, sub in enumerate(pool):
                if ia in sub and ib in sub:
                    pos_a, pos_b = sub.index(ia), sub.index(ib)
                    if pos_a > pos_b:
                        pos_a, pos_b = pos_b, pos_a
                    pool[i] = sub[pos_a: pos_b + 1]
                    pool.insert(i + 1, sub[pos_b:] + sub[:pos_a + 1])
                    break

        result: dict[str, list[list[int]]] = {}
        for sec, pool in pools.items():
            pool.sort(key=lambda sub: min(ids[j] for j in sub))
            result[SECTION_NAMES[sec]] = [[int(ids[j]) for j in sub] for sub in pool]
        return result

    def build_cuts_dict(
        self,
        pairs_local: list[tuple[int, int]],
    ) -> dict[int, list[tuple[int, int]]]:
        """Convierte indices locales a IDs Nastran por costilla.

        Parameters
        ----------
        pairs_local : list of (idx_a, idx_b)
            Indices locales del array de perfil (identicos en todos los ribs
            porque comparten el mismo x_stations).

        Returns
        -------
        dict {rib_idx: [(id_a, id_b), ...]}
        """
        return {
            ridx: [
                (int(rib.ids.point_start + ia), int(rib.ids.point_start + ib))
                for ia, ib in pairs_local
            ]
            for ridx, rib in enumerate(self.ribs)
        }

    def apply_cut_cavities(
        self,
        cuts: list[tuple[int, int]] | dict[int, list[tuple[int, int]]],
        rib_index: int = 0,
        wall_thickness: float = 0.15,
    ) -> dict[int, list[tuple[np.ndarray, np.ndarray] | None]]:
        """Genera cavidades interiores desde las subsecciones de corte.

        Para cada subseccion de cada costilla:
          1. Reconstruye el poligono desde las coordenadas del perfil.
          2. Aplica offset_polygon(coords_closed, wall_thickness).
          3. Almacena el resultado en self._cut_inner_profiles[rib_idx].
          4. Descarta rib._inner_profiles (establece None).

        Parameters
        ----------
        cuts : list of (id_a, id_b)
            IDs Nastran del rib ``rib_index``; se propagan a todas las costillas.
        cuts : dict {rib_idx: [(id_a, id_b), ...]}
            IDs Nastran especificos por costilla.
        wall_thickness : float
            Distancia de offset interior en metros.

        Returns
        -------
        dict {rib_idx: [inner_profile | None, ...]}
            Perfil interior de cada subseccion, en orden LE -> Box -> TE.
        """
        section_result = self.cut_sections(cuts, rib_index=rib_index)
        assert section_result is not None  # cuts != None garantiza un dict de retorno

        result: dict[int, list[tuple[np.ndarray, np.ndarray] | None]] = {}

        for ridx, sections in section_result.items():
            rib = self.ribs[ridx]

            # Coordenadas del perfil sin punto de cierre duplicado
            px, py = rib.profile
            if np.allclose(px[0], px[-1]) and np.allclose(py[0], py[-1]):
                px, py = px[:-1], py[:-1]
            pts = np.column_stack([px, py])
            point_start = rib.ids.point_start

            inner_profiles: list[tuple[np.ndarray, np.ndarray] | None] = []

            for sec_name in SECTION_NAMES:          # preserva orden LE, Box, TE
                for sub_ids in sections.get(sec_name, []):
                    local_idx = [id_ - point_start for id_ in sub_ids]
                    coords_arr = pts[local_idx]
                    coords_closed = np.vstack([coords_arr, coords_arr[0:1]])
                    inner = offset_polygon(coords_closed, wall_thickness)
                    inner_profiles.append(
                        (inner[:, 0], inner[:, 1]) if inner is not None else None
                    )

            rib._inner_profiles = None  # type: ignore[assignment]
            result[ridx] = inner_profiles

        self._cut_inner_profiles = result
        return result


# ======================================================================== #
#  RibViewer — visualizacion e interaccion                                 #
# ======================================================================== #

class RibViewer:
    def __init__(
        self,
        wing: WingSection,
        rib_index: int = 0,
        wall_thickness: float = 0.15,
    ) -> None:
        self._wing = wing
        self.rib = wing.ribs[rib_index]
        self.wall_thickness = wall_thickness
        self._setup_geometry()
        self._pending_idx: int | None = None
        self._pairs: list[tuple[int, int, int]] = []  # (idx_a, idx_b, sec)
        self._pair_artists: list = []
        self._build_figure()

    # ------------------------------------------------------------------ #
    #  Geometria                                                           #
    # ------------------------------------------------------------------ #
    def _setup_geometry(self) -> None:
        rib = self.rib
        px, py = rib.profile

        # Puntos unicos del contorno (el ultimo repite el primero)
        if np.allclose(px[0], px[-1]) and np.allclose(py[0], py[-1]):
            px, py = px[:-1], py[:-1]
        self._pts = np.column_stack([px, py])
        n = len(self._pts)
        self._ids = np.arange(rib.ids.point_start, rib.ids.point_start + n)

        # Indices de spar en el array del perfil
        # spar_ids = [upper_s1, upper_s2, lower_s2, lower_s1]
        si = [sid - rib.ids.point_start for sid in rib.spar_ids]
        i0, i1, i2, i3 = si[0], si[1], si[2], si[3]

        # Listas ordenadas de indices del contorno por seccion
        self._section_indices: list[list[int]] = [
            list(range(0, i0 + 1)) + list(range(i3, n)),        # LE
            list(range(i0, i1 + 1)) + list(range(i2, i3 + 1)),  # Box
            list(range(i1, i2 + 1)),                             # TE
        ]

        # Seccion de cada punto (la de menor indice que lo contiene)
        assigned: dict[int, int] = {}
        for s, pts in enumerate(self._section_indices):
            for k in pts:
                if k not in assigned:
                    assigned[k] = s
        self._section_of = np.array([assigned[k] for k in range(n)], dtype=int)

    # ------------------------------------------------------------------ #
    #  Figura principal                                                    #
    # ------------------------------------------------------------------ #
    def _build_figure(self) -> None:
        rib = self.rib
        self.fig, self.ax = plt.subplots(figsize=(13, 5))
        self.ax.set_aspect("equal")
        self.ax.set_title(
            f"NACA {rib.airfoil.designation}  |  "
            f"y = {rib.span_position:.2f} m  |  chord = {rib.chord:.2f} m\n"
            "[Click] seleccionar punto   [Enter] aplicar   [S] exportar   [R] reiniciar   [Q] cerrar"
        )
        self.ax.set_xlabel("X [m]")
        self.ax.set_ylabel("Z [m]")
        self.ax.grid(True, ls=":")

        # Relleno de secciones
        for indices, color, name in zip(
            self._section_indices, SECTION_COLORS, SECTION_NAMES
        ):
            self.ax.fill(
                self._pts[indices, 0], self._pts[indices, 1],
                alpha=0.12, color=color, label=name, zorder=1,
            )

        # Contorno exterior
        px, py = rib.profile
        self.ax.plot(px, py, "k-", lw=1.0, alpha=0.5, zorder=2)

        # Puntos coloreados por seccion
        for s, color in enumerate(SECTION_COLORS):
            mask = self._section_of == s
            self.ax.scatter(
                self._pts[mask, 0], self._pts[mask, 1],
                s=28, c=color, zorder=4,
            )

        # Anotaciones de IDs
        for k in range(len(self._pts)):
            self.ax.annotate(
                str(self._ids[k]),
                (self._pts[k, 0], self._pts[k, 1]),
                xytext=(3, 3), textcoords="offset points",
                fontsize=6, color="dimgray", zorder=5,
            )

        # Lineas de spar
        sv = rib.spar_vertices
        for i in range(1, len(rib.spar_positions) + 1):
            self.ax.plot(
                [sv[0][i - 1], sv[0][-i]],
                [sv[1][i - 1], sv[1][-i]],
                color="k", lw=2, alpha=0.35, ls="-.", zorder=3,
            )

        # Marcador del punto pendiente
        self._marker, = self.ax.plot([], [], "ro", ms=10, zorder=6)

        self.ax.legend(loc="upper right", fontsize=8)
        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("key_press_event",   self._on_key)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Eventos                                                             #
    # ------------------------------------------------------------------ #
    def _nearest(self, x: float, y: float) -> int:
        d = np.hypot(self._pts[:, 0] - x, self._pts[:, 1] - y)
        return int(np.argmin(d))

    def _on_click(self, event) -> None:
        if event.inaxes != self.ax or event.button != 1:
            return
        idx = self._nearest(event.xdata, event.ydata)

        if self._pending_idx is None:
            self._pending_idx = idx
            self._marker.set_data([self._pts[idx, 0]], [self._pts[idx, 1]])
            sec = self._section_of[idx]
            print(f"[1] ID {self._ids[idx]}  (seccion {SECTION_NAMES[sec]})")
        else:
            a, b = self._pending_idx, idx
            sec_a, sec_b = int(self._section_of[a]), int(self._section_of[b])

            if a == b:
                print("[!] Mismo punto -- selecciona otro")
            elif sec_a != sec_b:
                print(
                    f"[!] Secciones distintas "
                    f"({SECTION_NAMES[sec_a]} vs {SECTION_NAMES[sec_b]}) "
                    f"-- selecciona un punto de {SECTION_NAMES[sec_a]}"
                )
            else:
                self._pairs.append((a, b, sec_a))
                art, = self.ax.plot(
                    [self._pts[a, 0], self._pts[b, 0]],
                    [self._pts[a, 1], self._pts[b, 1]],
                    "r-", lw=2, zorder=5,
                )
                self._pair_artists.append(art)
                print(
                    f"[2] ID {self._ids[b]}  -> pareja "
                    f"({self._ids[a]}, {self._ids[b]})  "
                    f"seccion {SECTION_NAMES[sec_a]}  -- Enter para aplicar"
                )

            self._pending_idx = None
            self._marker.set_data([], [])

        self.fig.canvas.draw_idle()

    def _on_key(self, event) -> None:
        if event.key == "enter":
            self._apply_cuts()
        elif event.key == "s":
            self._export_cuts_dict()
        elif event.key == "r":
            for art in self._pair_artists:
                art.remove()
            self._pair_artists.clear()
            self._pairs.clear()
            self._pending_idx = None
            self._marker.set_data([], [])
            self.fig.canvas.draw_idle()
            print("[R] Seleccion reiniciada")
        elif event.key == "q":
            plt.close(self.fig)

    # ------------------------------------------------------------------ #
    #  Cortes — delega calculo a WingSection                              #
    # ------------------------------------------------------------------ #
    def _apply_cuts(self) -> None:
        pairs_local = [(ia, ib) for ia, ib, _ in self._pairs]
        result = self._wing._compute_rib_cuts(self.rib, pairs_local)

        print("\n" + "=" * 62)
        all_subs: list[tuple[str, list[int]]] = []
        for sec_name, subsections in result.items():
            n = len(subsections)
            label_suffix = "subseccion" if n == 1 else "subsecciones"
            print(f"\nSeccion {sec_name}  ->  {n} {label_suffix}:")
            for k, ids in enumerate(subsections, 1):
                label = f"{sec_name}-{k}"
                local_sub = [id_ - self.rib.ids.point_start for id_ in ids]
                print(f"  {label}  ({len(ids):2d} pts): {ids}")
                all_subs.append((label, local_sub))
        print("=" * 62 + "\n")

        self._plot_subsections(all_subs)

    # ------------------------------------------------------------------ #
    #  Figura de subsecciones con buffer                                  #
    # ------------------------------------------------------------------ #
    def _plot_subsections(
        self, subs: list[tuple[str, list[int]]]
    ) -> None:
        n = len(subs)
        colors = plt.colormaps["tab10"](np.linspace(0, 0.9, n))
        fig, axes = plt.subplots(1, n + 1, figsize=(4 * (n + 1), 4))

        ax0 = axes[0]
        ax0.set_aspect("equal")
        ax0.set_title("Vista global")
        ax0.grid(True, ls=":")

        for i, ((label, sub), color) in enumerate(zip(subs, colors)):
            coords_arr = self._pts[sub]                               # (n_sub, 2) abierto
            coords_closed = np.vstack([coords_arr, coords_arr[0:1]]) # (n_sub+1, 2) cerrado
            inner_arr = offset_polygon(coords_closed, self.wall_thickness)

            xo = np.append(coords_arr[:, 0], coords_arr[0, 0])
            yo = np.append(coords_arr[:, 1], coords_arr[0, 1])
            outer_area = Polygon(coords_arr).area
            xi = inner_arr[:, 0] if inner_arr is not None else None
            yi = inner_arr[:, 1] if inner_arr is not None else None

            # Vista global
            ax0.plot(xo, yo, color=color, lw=0.8, ls=":", alpha=0.6)
            if inner_arr is not None and xi is not None and yi is not None:
                ax0.fill(xi, yi, alpha=0.30, color=color)
                ax0.plot(xi, yi, color=color, lw=1.5)

            # Subplot individual
            ax = axes[i + 1]
            ax.set_aspect("equal")
            ax.grid(True, ls=":")
            ax.plot(xo, yo, color=color, lw=0.8, ls=":", alpha=0.6)
            if inner_arr is not None and xi is not None and yi is not None:
                inner_area = Polygon(inner_arr[:-1]).area
                ax.fill(xi, yi, alpha=0.30, color=color)
                ax.plot(xi, yi, color=color, lw=1.5)
                ax.set_title(
                    f"{label}\next={outer_area:.4f}  int={inner_area:.4f}"
                )
            else:
                ax.fill(xo, yo, alpha=0.15, color=color)
                ax.set_title(f"{label}\next={outer_area:.4f}\n(sin buffer)")

        fig.suptitle(
            f"NACA {self.rib.airfoil.designation}  |  "
            f"{n} subseccion(es)  |  thickness={self.wall_thickness} m",
            fontsize=11,
        )
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    #  Exportacion — delega a WingSection                                 #
    # ------------------------------------------------------------------ #
    def _export_cuts_dict(self) -> None:
        pairs_local = [(ia, ib) for ia, ib, _ in self._pairs]
        d = self._wing.build_cuts_dict(pairs_local)
        print("\n" + "=" * 62)
        print("[S] Diccionario de cortes por costilla")
        print("    cuts[rib_idx] = [(id_a, id_b), ...]")
        print("cuts = {")
        for rib_idx, pairs in d.items():
            print(f"    {rib_idx}: {pairs},")
        print("}")
        print("=" * 62 + "\n")


# ======================================================================== #
#  Helpers                                                                 #
# ======================================================================== #

def _build_wing_const(span: float = 35.0) -> WingSection:
    return WingSection(
        airfoil=NACAAirfoil("6412"),
        span=span,
        n_ribs=10,
        spar_positions=[0.2, 0.65],
        stringer_positions=np.linspace(0.05, 0.8, 8),
        chord_distribution=([0, 5.18, span], [9, 7, 2]),
        offset_distribution=([0, 5.18, span], [0, 2.42, 13.06]),
        twist_distribution=([0, span], [np.deg2rad(2), -np.deg2rad(4)]),
        elevation_distribution=([0, span], [0, span * np.sin(6 * np.pi / 180)]),
        n_skin_points=20,
    )


if __name__ == "__main__":
    wing = _build_wing_const()
    wing.cut_sections()  # modo interactivo
