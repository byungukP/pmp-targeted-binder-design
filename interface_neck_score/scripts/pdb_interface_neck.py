#!/usr/bin/env python3
"""
PDB-input wrapper for the chain-aware interface-neck score.


"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from cif_interface_neck import VDW_RADII, interface_neck_score, voxelized_ses


TWO_LETTER_ELEMENTS = {
    "CL", "BR", "FE", "ZN", "MG", "CA", "NA", "CU", "MN", "CO",
    "NI", "SE",
}


def infer_element(atom_name: str, element_field: str) -> str:
    """Read the PDB element field or infer it conservatively from atom name."""
    element = "".join(c for c in element_field if c.isalpha()).upper()
    if element:
        return element[:2] if element[:2] in TWO_LETTER_ELEMENTS else element[:1]

    # In standard PDB formatting, one-letter protein elements have a leading
    # space in the four-character atom-name field; two-letter elements do not.
    raw = atom_name[:4].ljust(4)
    letters = "".join(c for c in raw if c.isalpha()).upper()
    if raw[0] != " " and letters[:2] in TWO_LETTER_ELEMENTS:
        return letters[:2]
    return letters[:1]


def read_pdb_atoms_and_chains(path: Path, model: int = 1,
                              include_hydrogen: bool = False,
                              include_hetero: bool = True):
    """Return coordinates, elements, and chain IDs from a PDB file."""
    coords, elements, chains = [], [], []
    current_model, saw_model = 1, False

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, 1):
            record = line[0:6].strip().upper()
            if record == "MODEL":
                saw_model = True
                try:
                    current_model = int(line[10:14].strip())
                except ValueError:
                    fields = line.split()
                    current_model = int(fields[1]) if len(fields) > 1 else 1
                continue
            if record == "ENDMDL":
                continue
            if record not in {"ATOM", "HETATM"}:
                continue
            if record == "HETATM" and not include_hetero:
                continue
            if saw_model and current_model != model:
                continue

            # Retain the blank/A conformer and discard duplicated alternatives.
            altloc = line[16:17] if len(line) >= 17 else " "
            if altloc not in {" ", "A", "1"}:
                continue
            try:
                xyz = [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            except (ValueError, IndexError) as exc:
                raise ValueError(f"Invalid coordinates at PDB line {line_number}") from exc

            atom_name = line[12:16] if len(line) >= 16 else ""
            element_field = line[76:78] if len(line) >= 78 else ""
            element = infer_element(atom_name, element_field)
            if not element or (element == "H" and not include_hydrogen):
                continue
            chain = line[21:22].strip() if len(line) >= 22 else ""
            chain = chain or "_"  # explicit label for a blank PDB chain ID
            coords.append(xyz)
            elements.append(element)
            chains.append(chain)

    if not coords:
        raise ValueError(f"No atoms found for model {model}")
    return np.asarray(coords, float), elements, np.asarray(chains, str)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Compute a two-chain interface-neck score from PDB")
    p.add_argument("pdb", type=Path, help="input PDB file")
    p.add_argument("--chain-a", help="first chain ID; auto-selected if omitted")
    p.add_argument("--chain-b", help="second chain ID; auto-selected if omitted")
    p.add_argument("--model", type=int, default=1)
    p.add_argument("--spacing", type=float, default=0.5,
                   help="voxel spacing in A (default: 0.5)")
    p.add_argument("--probe-radius", type=float, default=1.4)
    p.add_argument("--slab-width", type=float, default=0.5)
    p.add_argument("--smooth-sigma", type=float, default=1.0,
                   help="Gaussian smoothing sigma in A (default: 1.0)")
    p.add_argument("--endpoint-exclusion", type=float, default=0.10)
    p.add_argument("--quantile", type=float, default=0.10)
    p.add_argument("--include-hydrogen", action="store_true")
    p.add_argument("--exclude-hetero", action="store_true",
                   help="exclude HETATM records")
    args = p.parse_args()

    if args.spacing <= 0 or args.slab_width <= 0 or args.smooth_sigma < 0:
        p.error("spacing/slab width must be positive and sigma nonnegative")
    if not 0 <= args.endpoint_exclusion < 0.5 or not 0 <= args.quantile <= 1:
        p.error("invalid endpoint exclusion or quantile")

    try:
        coords, elements, chains = read_pdb_atoms_and_chains(
            args.pdb, args.model, args.include_hydrogen,
            not args.exclude_hetero)
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
        centroid_a = coords[chains == chain_a].mean(axis=0)
        centroid_b = coords[chains == chain_b].mean(axis=0)
        result = interface_neck_score(
            mask, origin, args.spacing, centroid_a, centroid_b,
            args.slab_width, args.smooth_sigma,
            args.endpoint_exclusion, args.quantile)
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
