# -*- coding: utf-8 -*-
# written by ByungUk Park 01/19/2026

R"""
edit a pdb file of protein-binder complex from RFD3 for MASIF-PMP
- Input: pdb file from CHARMM-GUI preprocessing (RFD3 outputs cif files)
- Sets chain ID (col 22) for ATOM/HETATM/TER records. --> 'c' for the complex
- Treats the FIRST 'TER' line as delimiter between chain 1 and chain 2.
- Renumbers residues in chain 2 so they continue from the last residue ID of chain 1.

"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional
import sys

def edit_chain_and_renumber_after_ter(
    input_pdb: str,
    output_pdb: str,
    chain_id_first: str = "c",
    chain_id_second: str = "c",
) -> None:
    """
    - Sets chain ID (col 22) for ATOM/HETATM/TER records.
    - Treats the FIRST 'TER' line as delimiter between chain 1 and chain 2.
    - Renumbers residues in chain 2 so they continue from the last residue ID of chain 1.
      (Based on residue key: (orig_resSeq, iCode). Preserves iCode.)
    - Does NOT modify SEGID (cols 73-76) or any other fields besides chain ID + resSeq.

    Notes:
    - PDB residue sequence number (resSeq) is columns 23-26 (1-based), i.e. line[22:26].
    - Insertion code (iCode) is column 27 (1-based), i.e. line[26].
    - This assumes one delimiter TER between two chains (as you described).
    """

    in_path = Path(input_pdb)
    out_path = Path(output_pdb)

    after_ter = False
    last_resseq_chain1: Optional[int] = None

    # Map original residue identifiers in chain2 -> new resSeq
    # Keyed by (orig_resSeq, iCode) so each residue stays consistent across atoms.
    chain2_resmap: Dict[Tuple[int, str], int] = {}
    next_resseq_chain2: Optional[int] = None

    def _get_resseq_icode(line: str) -> Tuple[int, str]:
        # resSeq is cols 23-26 (1-based) => [22:26]
        resseq_str = line[22:26]
        # iCode is col 27 (1-based) => [26]
        icode = line[26] if len(line) > 26 else " "
        return int(resseq_str), icode

    def _set_chain_id(line: str, new_chain: str) -> str:
        # chain ID is col 22 (1-based) => index 21
        if len(new_chain) != 1:
            raise ValueError("chain_id must be exactly 1 character for PDB format.")
        if len(line) < 22:
            return line
        return line[:21] + new_chain + line[22:]

    def _set_resseq(line: str, new_resseq: int) -> str:
        # Keep width 4, right-justified; preserve iCode at col 27
        if len(line) < 26:
            return line
        new_field = f"{new_resseq:4d}"
        return line[:22] + new_field + line[26:]

    with in_path.open("r") as fin, out_path.open("w") as fout:
        for line in fin:
            rec = line[:6].strip()

            # Detect delimiter: first TER splits chain1 and chain2
            if rec == "TER":
                # TER also has chain/resSeq fields; set chain ID as appropriate
                # TER record typically includes resSeq of the terminating residue
                if not after_ter:
                    # Before TER => chain1
                    try:
                        resseq, _ = _get_resseq_icode(line)
                        last_resseq_chain1 = resseq
                    except Exception:
                        pass

                    line = _set_chain_id(line, chain_id_first)
                    after_ter = True

                    # initialize chain2 starting resseq
                    if last_resseq_chain1 is None:
                        # fallback if TER didn't have resseq: start at 1
                        last_resseq_chain1 = 0
                    next_resseq_chain2 = last_resseq_chain1 + 1
                else:
                    # If there are extra TERs later, just apply second chain ID
                    line = _set_chain_id(line, chain_id_second)

                fout.write(line)
                continue

            if rec in ("ATOM", "HETATM"):
                if not after_ter:
                    # Chain 1: set chain ID; track last residue ID
                    line = _set_chain_id(line, chain_id_first)
                    try:
                        resseq, _ = _get_resseq_icode(line)
                        last_resseq_chain1 = resseq if (last_resseq_chain1 is None) else max(last_resseq_chain1, resseq)
                    except Exception:
                        pass
                else:
                    # Chain 2: set chain ID; renumber residues continuously
                    line = _set_chain_id(line, chain_id_second)

                    if next_resseq_chain2 is None:
                        # If TER was missing for some reason, fall back
                        next_resseq_chain2 = 1

                    try:
                        orig_resseq, icode = _get_resseq_icode(line)
                        key = (orig_resseq, icode)
                        if key not in chain2_resmap:
                            chain2_resmap[key] = next_resseq_chain2
                            next_resseq_chain2 += 1
                        line = _set_resseq(line, chain2_resmap[key])
                    except Exception:
                        # If parsing fails, write line as-is after chain edit
                        pass

                fout.write(line)
                continue

            # For all other records: write as-is
            fout.write(line)


# Executable section
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python edit_complexPdb4masif_pmp.py <input_pdb> <output_pdb>")
        sys.exit(1)
    edit_chain_and_renumber_after_ter(
        input_pdb=sys.argv[1],
        output_pdb=sys.argv[2],
        chain_id_first="c",
        chain_id_second="c",
    )
