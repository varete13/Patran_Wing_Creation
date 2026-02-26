# Patran Wing Creation

Parametric NACA 4-digit wing geometry generator for Nastran/Patran FEA.
Generates rib profiles, spar and stringer geometry, and exports to **Patran SES session files** (`.ses`) or **Nastran BDF** (`.bdf`).

---

## Features

- **NACA 4-digit airfoils** with full upper/lower surface discretisation.
- **Variable airfoil along the span** — NACA parameters are linearly interpolated between user-defined stations.
- **Spanwise distributions** for chord, sweep offset, twist and dihedral elevation (all driven by `scipy.interpolate.interp1d`).
- **Uniform or cosine point spacing** — cosine clusters points near the leading and trailing edges.
- **Automatic minimum-spacing filter** — removes profile points that would be closer than a physical threshold at the smallest chord station.
- **Entity budget solver** (`Wing.max_skin_points`) — computes the maximum skin resolution that keeps total entities within a Patran limit.
- **Hollow ribs** — each structural section (LE, spar box, TE) gets an independent wall thickness, constant or interpolated along the span.
- **Inner cuts** — subdivide any section of every rib or individual ribs with additional geometry lines.
- **Hollow cut sub-sections** — independent wall thickness per section for cut geometry.
- **Section map plot** (`Wing.section_map_plot`) — interactive colour-coded plot of local profile indices to assist in choosing `inner_cuts` pairs.
- **Patran SES export** — grids, PWL curves, trimmed surfaces, spar/stringer lines and skin quad panels written as Patran session commands.
- **Nastran BDF export** — GRID cards for standalone use.
- **Automatic ID management** — collision-free Nastran entity numbering that scales with rib count and skin resolution.

---

## Project structure

```
Patran_Wing_Creation/
├── main.py                        # Usage examples (run directly)
├── perfilador/
│   ├── airfoil.py                 # NACAAirfoil — geometry & section properties
│   ├── rib.py                     # Rib — single cross-section
│   ├── wing.py                    # Wing — full parametric wing
│   ├── geometry.py                # offset_polygon, hollow_rib_sections, …
│   ├── plotting.py                # plot_3d_view, plot_side_view, plot_rib, …
│   ├── id_manager.py              # IDManager, IDScheme, RibIDs
│   └── exporters/
│       ├── ses.py                 # save_ses — Patran session file
│       └── bdf.py                 # save_bdf — Nastran BDF
```

---

## Requirements

| Package | Purpose |
|---------|---------|
| `numpy` | Array maths |
| `scipy` | Spanwise interpolation (`interp1d`) and cross-section integrals |
| `matplotlib` | Visualisation |
| `shapely` | Polygon offset for hollow ribs |

Install with:

```bash
pip install numpy scipy matplotlib shapely
```

---

## Quick start

```python
import numpy as np
from perfilador import NACAAirfoil, Wing, save_ses, plot_3d_view
import matplotlib.pyplot as plt

span = 35.0

wing = Wing(
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

_, ax = plt.subplots(subplot_kw={"projection": "3d"})
plot_3d_view(wing, ax)
plt.show()

save_ses(wing, "my_wing.ses.01")
```

---

## Wing geometry

```
  Planform  (top view — schematic, not to scale)

       root (y = 0)                                     tip (y = b)
       (chord = c₀)                                 (chord = c_tip)

  LE   ●──────────────────────────────────────────────────●   ← LE (swept aft)
       │╲                                              ╱  │
       │  ─────────────────────────────────────────── │   ← front spar  (0.20 c)
       │    ╲                                       ╱ │
       │      ───────────────────────────────────── │   ← rear spar   (0.65 c)
       │   │    │    │    │    │    │    │    │    │ │   ← rib stations
  TE   ●──────────────────────────────────────────────────●   ← TE (tapered)
```

The diagonals (╲ ╱) indicate that each spar spans the full half-span but is
shorter in the chord direction at the tip due to taper.

```
  Elevation  (side view, looking inboard → outboard)

   z (elevation)
   ↑             rib n-1 (tip)
   │            ╱
   │    rib 1  ╱  dihedral angle
   │          ╱
   ●─────────╱───────────────────────────────────────── y (span)
   rib 0 (root)
```

---

## Wing parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `airfoil` | `NACAAirfoil` | Single airfoil for all stations. Mutually exclusive with `airfoil_distribution`. |
| `airfoil_distribution` | `dict[float, str]` | `{y: "NACA_code"}` — profile interpolated between stations. |
| `span` | `float` | Semi-span in metres. |
| `n_ribs` | `int` | Number of evenly-spaced rib stations root → tip. |
| `spar_positions` | `list[float]` | Chordwise spar positions as fraction of chord (0–1). |
| `stringer_positions` | `array-like` | Chordwise stringer positions as fraction of chord (0–1). |
| `chord_distribution` | `([y…], [c…])` | Chord length (m) vs spanwise station. |
| `offset_distribution` | `([y…], [x…])` | Leading-edge sweep offset (m) vs spanwise station. |
| `twist_distribution` | `([y…], [θ…])` | Geometric twist (rad) vs spanwise station. |
| `elevation_distribution` | `([y…], [z…])` | Dihedral elevation (m) vs spanwise station. |
| `n_skin_points` | `int` | Discretisation points per surface side (default `100`). |
| `point_spacing` | `"uniform"` \| `"cosine"` | Cosine clusters points near LE/TE. |
| `min_point_spacing` | `float` \| `None` | Minimum physical distance (m) between profile points. `None` disables. Default `0.005`. |
| `wall_thickness` | `list[float]` \| `dict[float, list[float]]` \| `None` | Wall thickness per section (LE, box, TE). `list` → constant; `dict` → interpolated along span. |
| `inner_cuts` | `list[tuple[int,int]]` \| `dict[int, list[tuple[int,int]]]` \| `None` | Additional geometry cuts within sections. See below. |
| `cut_wall_thickness` | `list[float]` \| `dict[float, list[float]]` \| `None` | Wall thickness for inner-cut sub-sections. Same dual format as `wall_thickness`. |
| `wall_thickness_kind` | `"linear"` \| `"quadratic"` \| `"cubic"` \| … | `scipy.interp1d` `kind` for spanwise wall-thickness interpolation. |
| `cut_wall_thickness_kind` | same as above | Interpolation kind for cut wall thickness. |

---

## Variable airfoil

Profiles transition smoothly between any number of spanwise stations:

```python
wing = Wing(
    airfoil_distribution={
        0:        "6415",   # root  — thick, high camber
        span / 2: "4412",   # mid
        span:     "2409",   # tip   — thin, low camber
    },
    ...
)
```

---

## Hollow ribs

Each rib is split into **three structural sections** by the spar planes.
Each section can have an independent wall thickness:

```
  Rib cross-section  (front view, spars at 0.20 c and 0.65 c)

           ●──────────────────────────────────────────────────────●
          ╱                                                         ╲
         ╱   ┌────────┐             ┌──────────────┐   ┌────────┐   ╲
        ●    │        │             │              │   │        │    ●
        │    │  void  │             │     void     │   │  void  │    │
        ●    │        │             │              │   │        │    ●
         ╲   └────────┘             └──────────────┘   └────────┘   ╱
          ╲                                                         ╱
           ●──────────────────────────────────────────────────────●
           │        │                             │               │
          LE      0.20 c                        0.65 c           TE
           │← LE section →│←────── spar box ─────→│← TE section →│
```

`wall_thickness=[t_LE, t_box, t_TE]` sets the wall thickness (m) for each
section. Setting a section's thickness to `None` or omitting a value leaves
it solid.

One thickness value per structural section (LE → spar box → TE):

```python
# Constant along the span
wing = Wing(..., wall_thickness=[0.08, 0.10, 0.06])

# Tapered from root to tip
wing = Wing(..., wall_thickness={
    0:    [0.10, 0.12, 0.08],   # root
    span: [0.04, 0.05, 0.03],   # tip
})
```

---

## Inner cuts

Subdivide any section with additional profile curves.
The `inner_cuts` parameter takes **local profile indices** (0-based):

```
  Profile index convention  (N = n_skin_points, total grid points = 2N-2)

  ╭──── upper surface: 0, 1, 2, …, N-2, N-1 ────╮
  ●─────────────────────────────────────────────────●
 ╱  0 (LE)                               N-1 (TE)    ╲
●                                                      ●
 ╲  2N-3 (near LE)              N (near TE)           ╱
  ●─────────────────────────────────────────────────●
  ╰──── lower surface: N, N+1, …, 2N-4, 2N-3 ─────╯

  Both indices of a cut pair must lie in the SAME section:
  ┌────────────────┬────────────────────────────────────────────┐
  │ Section        │ Typical index range (cosine, N ≈ 25)       │
  ├────────────────┼────────────────────────────────────────────┤
  │ LE  (upper)    │ 0 … i_spar0                                │
  │ Box (upper)    │ i_spar0+1 … i_spar1                        │
  │ TE  (upper)    │ i_spar1+1 … N-1                            │
  │ TE  (lower)    │ N … j_spar1                                │
  │ Box (lower)    │ j_spar1+1 … j_spar0                        │
  │ LE  (lower)    │ j_spar0+1 … 2N-3                           │
  └────────────────┴────────────────────────────────────────────┘
  Use Wing.section_map_plot() to identify exact boundaries.
```

Indices are **local profile indices** (0-based, shared by all ribs):

```python
# Global mode — same cuts on every rib
wing = Wing(..., inner_cuts=[(2, 43)])

# Per-rib mode — different cuts per rib
wing = Wing(..., inner_cuts={
    0: [(2, 43)],    # root rib
    5: [(3, 41)],    # rib index 5
})
```

Use `Wing.section_map_plot()` to identify local indices visually before constructing the wing:

```python
Wing.section_map_plot(
    airfoil=NACAAirfoil("6412"),
    spar_positions=[0.2, 0.65],
    stringer_positions=np.linspace(0.05, 0.8, 8),
    n_skin_points=25,
    point_spacing="cosine",
)
plt.show()
```

---

## Entity budget

Find the largest `n_skin_points` that stays within a Patran entity limit:

```python
n_sp = Wing.max_skin_points(
    n_ribs=10,
    n_spars=2,
    n_stringers=8,
    max_entities=1_190,
)
```

---

## Exports

```python
from perfilador import save_ses, save_bdf

save_ses(wing, "output.ses.01")   # Patran session file
save_bdf(wing, "output.bdf")      # Nastran BDF (GRID cards)
```

The SES file writes four Patran groups:

| Group name | Contents |
|------------|----------|
| `Secciones_Ala` | Rib trimmed surfaces |
| `Tramos_Largueros` | Spar lines and panels |
| `Tramos_Largerillos` | Stringer lines |
| `Secciones_Piel` | Skin quad-panel surfaces |

---

## Nastran ID scheme

| Entity | Base | Stride |
|--------|------|--------|
| Grid points (outer) | 100 000 | 1 000 per rib |
| Grid points (inner) | 200 000 | 1 000 per rib |
| Rib surfaces | 1 000 | 10 per rib |
| Spar panel surfaces | 2 000 | 10 per rib |
| Skin panel surfaces | 30 000 | 100 per rib pair |
| Inner surfaces | 50 000 | 10 per rib |

Strides are automatically widened if the skin resolution or rib count would cause collisions.

---

## Debug output

Entity summaries and geometry-collapse warnings are routed to Python's `logging` module at `DEBUG` level. To enable them:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## JSON serialisation

Any `Wing` can be saved to a JSON file and reloaded later without touching
the Python source:

```python
# Save
wing.to_json("configs/my_wing.json")

# Load — identical geometry, same Nastran IDs
wing2 = Wing.from_json("configs/my_wing.json")

# Or work with plain dicts
d = wing.to_dict()           # JSON-serialisable dict
wing3 = Wing.from_dict(d)
```

The JSON file stores every constructor parameter exactly as passed
(`n_skin_points` is the **requested** value, so `min_point_spacing` filtering
is re-applied identically on reload).  Numpy arrays are stored as plain lists;
float-keyed dicts (e.g. `wall_thickness`, `airfoil_distribution`) use string
keys to comply with JSON.

```json
{
  "airfoil": "6412",
  "span": 35.0,
  "n_ribs": 10,
  "spar_positions": [0.2, 0.65],
  "chord_distribution": {"y": [0, 5.18, 35.0], "values": [9, 7, 2]},
  "wall_thickness": {"0": [0.10, 0.12, 0.08], "35.0": [0.05, 0.05, 0.05]},
  "inner_cuts": [[2, 43]],
  ...
}
```

No extra dependencies — uses only Python's standard `json` and `pathlib`.

---

## Running the examples

```bash
python main.py
```

`main.py` builds ten wing configurations (constant profile, variable profile, cosine spacing, hollow ribs, inner cuts, combined) and exports a `.ses.01` file for each.
