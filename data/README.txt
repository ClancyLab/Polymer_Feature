The polymer solvation dataset of the paper: put combined_svp_ml_input.xlsx
here. configs/polymer_solvation_paper.yaml reads it from this folder.

One row per polymer-solvent pair: 9,126 pairs, from 351 DKP/DPP-based
repeat units in 26 solvents. 21 descriptor columns, computed with ORCA and
Multiwfn: 11 of the polymer repeat unit (names starting Polymer-) and 10 of
the solvent (Solvent-). The target, Total_eV, is the SMD solvation free
energy in eV.
