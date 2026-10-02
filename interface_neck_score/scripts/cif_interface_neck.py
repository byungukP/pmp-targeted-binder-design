#!/usr/bin/env python3
"""
Created on Wed Jul 15 2026

@author: byunguk park

Compute a chain-aware interface-neck score for a two-chain mmCIF complex.

Definition
----------
The axis joins the geometric centroids of the two selected chains.  The
voxelized solvent-excluded body is sliced perpendicular to this axis, giving
a cross-sectional-area profile A(t).  The score is

    S_interface-neck = Q10(A(t) in the inter-centroid region)
                       / min(max(A_left), max(A_right))

Q10 makes the interface estimate robust to a single noisy slice.  Values near
1 indicate a broad, smooth connection; small values indicate a narrow neck.

Usage:
    Run with explicit chain IDs:
        $ python cif_interface_neck.py structure.cif --chain-a A --chain-b B
    If the CIF contains exactly two chains, the chain IDs can be omitted.
        $ python cif_interface_neck.py structure.cif
    Optional higher-resolution calculation:
        $ python cif_interface_neck.py structure.cif --spacing 0.25

"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_closing, gaussian_filter1d


VDW_RADII = {
    "H": 1.20, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47,
    "P": 1.80, "S": 1.80, "CL": 1.75, "BR": 1.85, "I": 1.98,
    "FE": 1.80, "ZN": 1.39, "MG": 1.73, "CA": 2.31, "NA": 2.27,
    "K": 2.75, "CU": 1.40, "MN": 1.79, "CO": 1.67, "NI": 1.63,
    "SE": 1.90,
}
# The above radii are from Bondi, A. (1964) J. Phys. Chem., 68, 441-451.
# https://pubs.acs.org/doi/abs/10.1021/j100785a001, https://pubs.acs.org/doi/10.1021/jp8111556


def _clean_element(value: str) -> str:
    value = "".join(c for c in value if c.isalpha()).upper()
    # Two-letter elements that occur commonly in biomolecular structures.
    two = {"CL", "BR", "FE", "ZN", "MG", "CA", "NA", "CU", "MN",
           "CO", "NI", "SE"}
    return value[:2] if value[:2] in two else value[:1]


def read_mmcif_atoms_and_chains(path: Path, model: int = 1,
                                include_hydrogen: bool = False):
    """Return coordinates, elements, and chain IDs from the atom_site loop."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    for start, line in enumerate(lines):
        if line.strip() != "loop_":
            continue
        headers, i = [], start + 1
        while i < len(lines) and lines[i].lstrip().startswith("_atom_site."):
            headers.append(lines[i].strip())
            i += 1
        required = {"_atom_site.Cartn_x", "_atom_site.Cartn_y",
                    "_atom_site.Cartn_z"}
        if not headers or not required.issubset(headers):
            continue

        col = {name: j for j, name in enumerate(headers)}
        chain_name = ("_atom_site.auth_asym_id"
                      if "_atom_site.auth_asym_id" in col
                      else "_atom_site.label_asym_id")
        element_name = ("_atom_site.type_symbol"
                        if "_atom_site.type_symbol" in col
                        else "_atom_site.label_atom_id")
        model_name = "_atom_site.pdbx_PDB_model_num"
        rows, pending = [], []
        while i < len(lines):
            stripped = lines[i].strip()
            if stripped == "#" or stripped == "loop_" or stripped.startswith("_"):
                break
            if stripped:
                pending.extend(shlex.split(lines[i], comments=False, posix=True))
                while len(pending) >= len(headers):
                    rows.append(pending[:len(headers)])
                    pending = pending[len(headers):]
            i += 1

        xyz, elements, chains = [], [], []
        for row in rows:
            if model_name in col and row[col[model_name]] not in (".", "?"):
                if int(float(row[col[model_name]])) != model:
                    continue
            element = _clean_element(row[col[element_name]])
            if element == "H" and not include_hydrogen:
                continue
            if not element:
                continue
            xyz.append([float(row[col["_atom_site.Cartn_x"]]),
                        float(row[col["_atom_site.Cartn_y"]]),
                        float(row[col["_atom_site.Cartn_z"]])])
            elements.append(element)
            chains.append(row[col[chain_name]])
        if not xyz:
            raise ValueError(f"No atoms found for model {model}")
        return np.asarray(xyz, float), elements, np.asarray(chains, str)
    raise ValueError("No suitable atom_site loop was found")


def _ball(radius_voxels: int) -> np.ndarray:
    a = np.arange(-radius_voxels, radius_voxels + 1)
    x, y, z = np.meshgrid(a, a, a, indexing="ij")
    return x*x + y*y + z*z <= radius_voxels*radius_voxels


def voxelized_ses(coords, elements, spacing=0.5, probe_radius=1.4,
                  default_radius=1.7, max_voxels=150_000_000):
    """Voxelize vdW atoms, then apply probe-radius morphological closing."""
    radii = np.asarray([VDW_RADII.get(e, default_radius) for e in elements])
    pad = radii.max() + probe_radius + 2.0 * spacing
    lo = coords.min(axis=0) - pad
    hi = coords.max(axis=0) + pad
    shape = np.ceil((hi - lo) / spacing).astype(int) + 1
    nvox = int(np.prod(shape, dtype=np.int64))
    if nvox > max_voxels:
        raise MemoryError(
            f"Grid would contain {nvox:,} voxels ({tuple(shape)}). "
            "Increase --spacing or --max-voxels."
        )
    mask = np.zeros(tuple(shape), dtype=bool)
    for center, radius in zip(coords, radii):
        c = (center - lo) / spacing
        r = int(np.ceil(radius / spacing))
        lower = np.maximum(np.floor(c - r).astype(int), 0)
        upper = np.minimum(np.ceil(c + r).astype(int) + 1, shape)
        ix = np.arange(lower[0], upper[0])
        iy = np.arange(lower[1], upper[1])
        iz = np.arange(lower[2], upper[2])
        d2 = ((ix[:, None, None] - c[0])**2 +
              (iy[None, :, None] - c[1])**2 +
              (iz[None, None, :] - c[2])**2) * spacing**2
        mask[np.ix_(ix, iy, iz)] |= d2 <= radius**2
    rp = max(1, int(round(probe_radius / spacing)))
    ses = binary_closing(mask, structure=_ball(rp), border_value=0)
    return ses, lo


def interface_neck_score(mask: np.ndarray, origin: np.ndarray, spacing: float,
                         centroid_a: np.ndarray, centroid_b: np.ndarray,
                         slab_width: float = 0.5, smooth_sigma: float = 1.0,
                         endpoint_exclusion: float = 0.10,
                         quantile: float = 0.10):
    """Calculate the score and diagnostic cross-sectional profile."""
    delta = centroid_b - centroid_a
    distance = float(np.linalg.norm(delta))
    if distance == 0:
        raise ValueError("The two chain centroids coincide; axis is undefined")
    axis = delta / distance

    points = np.argwhere(mask).astype(float) * spacing + origin
    midpoint = 0.5 * (centroid_a + centroid_b)
    t = (points - midpoint) @ axis
    ta = float((centroid_a - midpoint) @ axis)
    tb = float((centroid_b - midpoint) @ axis)
    if ta > tb:
        ta, tb = tb, ta

    edges = np.arange(np.floor(t.min() / slab_width) * slab_width,
                      np.ceil(t.max() / slab_width) * slab_width + slab_width,
                      slab_width)
    counts, _ = np.histogram(t, bins=edges)
    # Each voxel contributes volume spacing^3. Dividing slab volume by slab
    # width gives the mean occupied cross-sectional area of that slab.
    raw_area = counts.astype(float) * spacing**3 / slab_width
    sigma_bins = smooth_sigma / slab_width
    area = gaussian_filter1d(raw_area, sigma=sigma_bins, mode="nearest")
    positions = 0.5 * (edges[:-1] + edges[1:])

    margin = endpoint_exclusion * (tb - ta)
    interface_mask = (positions >= ta + margin) & (positions <= tb - margin)
    if interface_mask.sum() < 3:
        raise ValueError("Too few slabs between chain centroids")
    interface_values = area[interface_mask]
    interface_area = float(np.quantile(interface_values, quantile))

    # Locate the representative neck at the point nearest the chosen robust
    # interface area, then measure the largest lobe on either side.
    candidate_indices = np.flatnonzero(interface_mask)
    neck_index = int(candidate_indices[np.argmin(
        np.abs(area[candidate_indices] - interface_area))])
    left_max = float(area[:neck_index + 1].max())
    right_max = float(area[neck_index:].max())
    reference_area = min(left_max, right_max)
    if reference_area <= 0:
        raise ValueError("Could not determine positive lobe cross-sectional areas")
    score = min(interface_area / reference_area, 1.0)
    return {
        "score": score,
        "interface_area": interface_area,
        "left_lobe_max_area": left_max,
        "right_lobe_max_area": right_max,
        "reference_area": reference_area,
        "neck_position": float(positions[neck_index]),
        "centroid_distance": distance,
    }


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute a two-chain interface-neck score from mmCIF")
    p.add_argument("cif", type=Path)
    p.add_argument("--chain-a", help="first chain ID; auto-selected if omitted")
    p.add_argument("--chain-b", help="second chain ID; auto-selected if omitted")
    p.add_argument("--model", type=int, default=1)
    p.add_argument("--spacing", type=float, default=0.5,
                   help="voxel spacing in A (default: 0.5)")
    p.add_argument("--probe-radius", type=float, default=1.4)
    p.add_argument("--slab-width", type=float, default=0.5)
    p.add_argument("--smooth-sigma", type=float, default=1.0,
                   help="Gaussian smoothing sigma in A (default: 1.0)")
    p.add_argument("--endpoint-exclusion", type=float, default=0.10,
                   help="fraction excluded near each centroid (default: 0.10)")
    p.add_argument("--quantile", type=float, default=0.10,
                   help="robust interface-area quantile (default: 0.10)")
    p.add_argument("--include-hydrogen", action="store_true")
    args = p.parse_args()

    if args.spacing <= 0 or args.slab_width <= 0 or args.smooth_sigma < 0:
        p.error("spacing/slab width must be positive and sigma nonnegative")
    if not 0 <= args.endpoint_exclusion < 0.5 or not 0 <= args.quantile <= 1:
        p.error("invalid endpoint exclusion or quantile")
    try:
        coords, elements, chains = read_mmcif_atoms_and_chains(
            args.cif, args.model, args.include_hydrogen)
        unique, counts = np.unique(chains, return_counts=True)
        order = unique[np.argsort(counts)[::-1]]
        if args.chain_a is None and args.chain_b is None:
            if len(order) != 2:
                raise ValueError(
                    f"Found chains {list(unique)}; specify --chain-a and --chain-b")
            chain_a, chain_b = order[0], order[1]
        elif args.chain_a is not None and args.chain_b is not None:
            chain_a, chain_b = args.chain_a, args.chain_b
        else:
            raise ValueError("Provide both --chain-a and --chain-b, or neither")
        if chain_a == chain_b or chain_a not in unique or chain_b not in unique:
            raise ValueError(f"Invalid chain selection; available chains: {list(unique)}")

        selected = (chains == chain_a) | (chains == chain_b)
        selected_coords = coords[selected]
        selected_elements = [e for e, keep in zip(elements, selected) if keep]
        mask, origin = voxelized_ses(selected_coords, selected_elements,
                                     args.spacing, args.probe_radius)
        ca = coords[chains == chain_a].mean(axis=0)
        cb = coords[chains == chain_b].mean(axis=0)
        result = interface_neck_score(
            mask, origin, args.spacing, ca, cb, args.slab_width,
            args.smooth_sigma, args.endpoint_exclusion, args.quantile)
    except (OSError, ValueError, MemoryError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    print(f"chain_a: {chain_a}")
    print(f"chain_b: {chain_b}")
    print(f"chain_centroid_distance_A: {result['centroid_distance']:.6f}")
    print(f"interface_area_A2: {result['interface_area']:.6f}")
    print(f"left_lobe_max_area_A2: {result['left_lobe_max_area']:.6f}")
    print(f"right_lobe_max_area_A2: {result['right_lobe_max_area']:.6f}")
    print(f"reference_lobe_area_A2: {result['reference_area']:.6f}")
    print(f"S_interface_neck: {result['score']:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
