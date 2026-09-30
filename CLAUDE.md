# CLAUDE.md

Guidance for AI assistants working in this repository.

## What this project is

A Python tool that builds a parametric wing (NACA 4-digit ribs, spars, stringers,
skin) and exports it for FEA:

- **Patran session files** (`.ses.01`) — geometry commands replayed inside Patran.
- **Nastran BDF** (`.bdf`) — a full structural mesh (MAT1, PSHELL, PBAR, GRID,
  CQUAD4, CBAR) written through pyNastran.

The domain is aerospace structures. Some names, comments and Patran group names
are in Spanish (`perfilador` = profiler, `ala` = wing, `costilla` = rib,
`larguero` = spar, `larguerillo` = stringer, `piel` = skin). Keep existing
Spanish identifiers as they are; new code and docstrings are written in English.

## Repository layout

```
Patran_Wing_Creation/
├── perfilador/                 # The library. All real logic lives here.
│   ├── __init__.py             # Public API re-exports (__all__)
│   ├── airfoil.py              # NACAAirfoil: NACA 4-digit geometry, area, CG
│   ├── rib.py                  # Rib: one cross-section, cached geometry, inner cuts
│   ├── wing.py                 # Wing: spanwise distributions, builds ribs, JSON I/O,
│   │                           #   entity budget (max_skin_points), section_map_plot,
│   │                           #   EntitySummary
│   ├── geometry.py             # Transforms, offset_polygon, extract_sections,
│   │                           #   hollow_rib_sections (shapely)
│   ├── id_manager.py           # IDScheme / RibIDs / IDManager (Nastran ID allocation)
│   ├── fea_props.py            # MatProps, BarProps, FEAProperties (BDF materials)
│   ├── plotting.py             # plot_rib, plot_rib_annotations, plot_side_view, plot_3d_view
│   └── exporters/
│       ├── ses.py              # save_ses(wing, filename)
│       └── bdf.py              # save_bdf(wing, filename, fea, n_rib_layers, n_span_div)
├── generate_configs.py         # Builds example Wing objects → configs/*.json
├── main.py                     # Loads configs/*.json → plots + SES/BDF exports
├── airfoil_section_builder.py  # Interactive matplotlib tool for picking inner cuts
│                               #   (WingSection subclass + RibViewer), Spanish docs
├── run_ses_in_patran.py        # Windows-only: drives a running Patran GUI to play a .ses.01
├── Perfilador_Ala_Nastran.py   # LEGACY monolithic script (class Ala). Do not extend.
└── README.md                   # User-facing docs (features, parameters, examples)
```

### Data flow

1. `Wing.__init__` validates inputs and builds `scipy.interpolate.interp1d`
   functions for chord, LE offset, twist (radians), elevation, airfoil
   parameters and wall thickness.
2. It builds chordwise stations (uniform or cosine). It then drops points closer
   than `min_point_spacing` at the smallest chord, so the effective
   `wing.n_skin_points` can be lower than the requested value.
3. An `IDManager` is created with the effective point count and widens its
   strides by powers of ten until no ID ranges can collide.
4. One `Rib` per evenly spaced station is built. Each rib computes and caches its
   profile, spar and stringer vertices snapped to profile points, hollow inner
   profiles, and cut sub-sections.
5. Exporters only read `wing.ribs`, `wing.id_scheme` and rib properties. They
   never mutate the wing.

## Environment and commands

Python 3.10+ is expected (the code uses `X | None` hints with
`from __future__ import annotations`; `ses.py` has a `pairwise` fallback for
3.9). There is no `requirements.txt`, `pyproject.toml`, test suite, linter
config or CI.

```bash
pip install numpy scipy matplotlib shapely pyNastran

python generate_configs.py   # writes configs/*.json (must run first)
python main.py               # loads configs, plots, writes *.ses.01 and wing_combined.bdf
```

- `configs/` does not exist in a fresh clone because `*.json` is gitignored.
  `main.py` fails with `FileNotFoundError` until `generate_configs.py` has run.
- `main.py` has `plt.show()` commented out. For headless runs set
  `MPLBACKEND=Agg`.
- Both scripts write their outputs into the current working directory.
- `run_ses_in_patran.py` needs Windows, a running Patran window, and
  `pyautogui`, `pygetwindow`, `pyperclip`. It cannot run in a Linux container.
- `airfoil_section_builder.py` is interactive and needs a GUI backend.

### Verifying a change

There are no automated tests. After changing library code, verify with:

```bash
python generate_configs.py && MPLBACKEND=Agg python main.py
```

Both must finish without exceptions and produce ten `.ses.01` files plus
`wing_combined.bdf`. For changes to ID allocation or exporters, diff the
generated `.ses.01` / `.bdf` against output from the previous commit. Nastran
IDs are part of the output contract, and existing Patran sessions depend on
them. Enable `logging.basicConfig(level=logging.DEBUG)` to see the entity
summary and geometry-collapse warnings.

## Key conventions

### Units and coordinates

- SI throughout: metres, pascals, kg/m³. Twist is in **radians** (examples use
  `np.deg2rad`).
- Chordwise and spar/stringer positions are chord fractions in `[0, 1]`.
- `span` is the semi-span. Ribs sit at `np.linspace(0, span, n_ribs)`.
- In exported files X1 is chordwise (LE → TE), X2 is the thickness direction
  and X3 is spanwise (root → tip).

### Profile indexing

With `N = n_skin_points`, each rib has `2N-2` profile points (the lower TE point
is skipped). Indices `0 … N-1` run along the upper surface from LE to TE.
Indices `N … 2N-3` run back along the lower surface from near the TE to near
the LE. Local index `i` maps to Nastran grid ID `rib.ids.point_start + i`.

`inner_cuts` pairs are these **local** indices and both ends must lie in the
same structural section (LE, spar box, TE). The valid pairs depend on
`n_skin_points` and `point_spacing`. The `megalodo` config shows that changing
the skin resolution means remapping cut indices, for example `(2, 43)` became
`(3, 85)`. Use `Wing.section_map_plot(...)` to find the boundaries.

### Sections and per-section lists

Spars split each rib into `len(spar_positions) + 1` sections, ordered
LE → TE. `wall_thickness` and `cut_wall_thickness` take either a list with one
value per section (constant along the span) or a dict `{y: [values…]}` that is
interpolated along the span with `wall_thickness_kind`. Lengths are validated
in both `Wing` and `Rib`.

### Nastran IDs

- `perfilador/id_manager.py` owns geometry IDs (points, inner points, rib,
  spar, skin and inner surfaces). Always allocate through `IDManager`. Never
  hardcode new ranges in exporters.
- `exporters/bdf.py` has its own fixed constants for material and property IDs
  and for element and extra-node ID ranges. The extra-node ranges start at
  `200_001` and `300_001`. Check for overlap with `IDScheme` whenever you touch
  either side.
- The README "Nastran ID scheme" table documents the default bases. Update it
  if you change `IDScheme` defaults.

### Wing serialisation

`Wing` stores its raw constructor arguments in `self._params`, and
`to_dict`/`from_dict` round-trip those, not derived state. When adding a
constructor parameter you must:

1. Add it to `self._params` in `__init__`.
2. Emit it in `to_dict` and read it in `from_dict` with a backward-compatible
   `d.get(key, default)`, so older JSON files still load.
3. Convert non-JSON types. Float-keyed dicts become string keys, tuples become
   lists and numpy arrays become lists.
4. Add or update a variant in `generate_configs.py` and document the parameter
   in the README "Wing parameters" table.

### Code style

- `from __future__ import annotations` at the top of every library module.
- Type hints on public functions and NumPy-style docstrings with a
  `Parameters` section.
- Geometry is computed once in `__init__` and exposed through read-only
  `@property` accessors. Keep state instance-owned. The legacy `Ala` class used
  class-level mutable counters, and the refactor deliberately removed them.
- Diagnostics go to `logging.debug`, not `print`, inside `perfilador/`. The
  top-level scripts may `print`.
- Anything public must be re-exported in `perfilador/__init__.py` and listed in
  `__all__`.
- Patran group names in `ses.py` (`Secciones_Ala`, `Tramos_Largueros`,
  `Tramos_Largerillos`, `Secciones_Piel`) stay unchanged for compatibility with
  existing sessions.

### Known gaps and pitfalls

- `perfilador/**/__pycache__/*.pyc` files are committed even though they
  should not be. Do not add new ones. Removing them and adding `__pycache__/`
  to `.gitignore` is a reasonable cleanup when asked.
- `.gitignore` also excludes `*.txt`, so a future `requirements.txt` would need
  an explicit exception.
- The README describes the BDF export as "GRID cards" and omits `pyNastran`
  from its requirements. The code writes a full mesh and needs pyNastran.
- The README says `main.py` builds the wings. It actually loads them from
  `configs/`, which `generate_configs.py` builds.
- `save_bdf` raises `ValueError` if neither `fea=` nor `wing.fea_properties`
  is provided.

## Git workflow

- The default branch is `main`. Work lands through pull requests from feature
  branches.
- Do not commit generated outputs. `*.ses.01`, `*.bdf`, `*.json`, `*.log` and
  `*.txt` are gitignored.
- Keep commits focused and describe the geometric or export behaviour that
  changed.
