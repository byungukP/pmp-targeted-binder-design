# Peripheral-membrane-protein (PMP) targeted binder design: structures, simulations, and interface analysis

**Author:** ByungUk Park\
**Date:** Oct 2, 2026

This repository contains the input and output for generated binders, scripts for computing complex geometry interface-neck scores, and interface analysis
code and results for de novo protein binders designed against three peripheral membrane protein targets. <!-- , as described in the following work: -->

<!-- 
B. Park and R. C. Van Lehn*. (2025). Decoding protein-membrane binding interfaces from surface-fingerprint-based geometric deep learning and molecular dynamics simulations. *bioRxiv*. https://doi.org/10.1101/2025.10.14.682447.
 -->


| Target directory | Target PMP (PDB ID) |
|---|---|
| `1CZT` | coagulation FV (1CZT) |
| `2B0M` | DHODH (2B0M) |
| `2OBI` | GPx4 (2OBI) |

Throughout, `ptn` denotes the target protein, `bN` denotes binder design *N* (i.e., 8 RFdiffusion3 binder designs `b0`–`b7`).

---

## Contents

```
.
├── rfd3/                      # RFdiffusion3 binder design models (8 per target)
├── interface_neck_score/      # Interface-neck score: code, complex structures, logs
├── interfacial_feat/          # MaSIF-style interfacial surface-feature analysis
└── edit_complexPDB4masif_pmp.py
```

### `rfd3/` — binder design models

Per target (`1CZT/`, `2B0M/`, `2OBI/`), 8 designs are stored as

* `*_model_N.cif.gz` — gzipped mmCIF coordinates of the target–binder complex
* `*_model_N.json` — design metadata, including the `diffused_index_map` relating
  diffused binder residue indices to chain/residue identifiers in the output model

### `interface_neck_score/` — geometric interface-neck score

A chain-aware descriptor of how constricted the target–binder interface is.
The axis joining the two chain centroids is used to slice a voxelized
solvent-excluded body perpendicular to that axis, giving a cross-sectional-area
profile *A(t)*. The score is

```
S_interface-neck = Q10( A(t) over the inter-centroid region )
                   / min( max(A_left), max(A_right) )
```

where `Q10` is the 10th percentile (robust to a single noisy slice). Values near 1
indicate a broad, smooth connection; small values indicate a narrow neck.

* `scripts/cif_interface_neck.py` — implementation, mmCIF input
* `scripts/pdb_interface_neck.py` — PDB-input wrapper around the same routines
* `complex_files/<target>_rfd/` — the 8 complexes scored per target (mmCIF, or PDB for 2B0M)
* `s_int_neck_<target>-binder.log` — per-complex results: chain IDs, centroid distance,
  interface area, left/right lobe maximum areas, reference lobe area, and `S_interface_neck`

Usage:

```bash
python scripts/cif_interface_neck.py complex.cif --chain-a A --chain-b B
```

```bash
python scripts/pdb_interface_neck.py complex.pdb --chain-a A --chain-b B
```

Chain IDs are auto-selected when omitted. Key options (defaults shown):
`--model 1`, `--spacing 0.5` Å voxel grid, `--probe-radius 1.4` Å,
`--slab-width 0.5` Å, `--smooth-sigma 1.0`, `--endpoint-exclusion 0.10`,
`--quantile 0.10`, `--include-hydrogen` (off by default).
Requires Python 3 with `numpy` and `scipy`.

### `interfacial_feat/` — interfacial surface features

Per target (`1CZT/`, `2B0M/`, `2OBI/`):

* `structures/` — target (`<TARGET>.pdb`), `binder.pdb`, and `complex.pdb`
* `surfaces/` — MSMS molecular-surface meshes (`.ply`) for target, binder, and complex
* `surface_feat/` — pre-computed MaSIF feature arrays (`.npy`), shape
  `(N_vertices, 100, 5)`; each patch of 100 vertices is centred on one surface vertex,
  and the five channels are shape index, distance-dependent curvature, hydrogen-bond
  potential, Poisson–Boltzmann electrostatics, and hydrophobicity

`interface_analysis.ipynb` reproduces the analysis and figures.

Interface vertices are those whose squared distance to the nearest complex-surface
vertex exceeds 2.0 Å² (≈1.41 Å), i.e. vertices buried upon complex formation; each is
assigned to its nearest heavy atom and hence to a parent residue. Downstream analysis
retains four interpretable descriptors (shape index, hydrogen-bond potential,
Poisson–Boltzmann electrostatics, hydrophobicity); distance-dependent curvature is excluded.

Notebook dependencies: `numpy`, `pandas`, `scipy`, `matplotlib`, `pymesh`, `pyflann`.

### `edit_complexPDB4masif_pmp.py`

Prepares a CHARMM-GUI-processed complex PDB for MASIF-PMP: assigns a single chain ID
(`c`) to all `ATOM`/`HETATM`/`TER` records and renumbers the residues after the first
`TER` so that the second chain continues the numbering of the first.

```bash
python edit_complexPDB4masif_pmp.py input.pdb output.pdb
```

---

## Data availability

This repository holds the design models, analysis code, and small input/output files.
The molecular-dynamics and umbrella-sampling data underlying the binding free-energy
analysis are too large to version-control (≈31 GB) and are deposited separately on
Dryad, DOI: `10.5061/dryad.XXXXXXX`:

* **Unbiased MD** — GROMACS inputs (`.mdp`, `sys.gro`, `sys.top`, `toppar/`, `index.ndx`)
  and trajectories (`.xtc`, `.tpr`) for the target–binder complex in bulk water, and for
  the complex, the isolated target, and the isolated binders in a highly mobile membrane
  mimetic (HMMM) bilayer; three replicas per system.
* **Steered MD and umbrella sampling** — cylinder-pulling inputs and trajectories, the
  per-window umbrella inputs and outputs, and the WHAM results (PMF, window histograms,
  and bootstrap profiles).

## Software

* [MaSIF-PMP](https://github.com/byungukP/masif_pmp): surface-feature computation & membrane-binding region prediction
* Python (3.8) with `numpy`, `scipy`, `pandas`, `matplotlib`, `pymesh`, `pyflann`
* [GROMACS](https://www.gromacs.org/) and [CHARMM-GUI](https://www.charmm-gui.org/): simulations and system building (Dryad deposit)

## License

Released under the MIT License — see [LICENSE](LICENSE).

<!-- 
## Citation

If you use this code, please cite the work using the BibTeX entry provided in [citation.bib](citation.bib)
 -->
