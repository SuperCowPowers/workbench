import importlib.resources
import math
from copy import deepcopy
from typing import Dict, List, Union

import numpy as np
import pandas as pd
from rdkit import Chem


def _float_or_zero(val) -> float:
    """Convert a value to float, returning 0.0 for NaN or unparseable values."""
    try:
        v = float(val)
        return 0.0 if math.isnan(v) else v
    except (ValueError, TypeError):
        return 0.0


with importlib.resources.open_text("monroe.assets", "periodic_table.csv") as f:
    PERIODIC_TABLE = pd.read_csv(f)
PERIODIC_TABLE = PERIODIC_TABLE.set_index("AtomicNumber")

ELECTRONEGATIVITY = [_float_or_zero(elem) for elem in PERIODIC_TABLE["Electronegativity"]]
FIRST_IONIZATION = [_float_or_zero(elem) for elem in PERIODIC_TABLE["FirstIonization"]]
MELTING_POINT = [_float_or_zero(elem) for elem in PERIODIC_TABLE["MeltingPoint"]]
PHASE = list(PERIODIC_TABLE["Phase"].values)
PHASE_SET = list(set(PHASE))
GROUP = deepcopy(PERIODIC_TABLE["Group"].values)
GROUP[np.isnan(GROUP)] = 19
GROUP_SET = list(set(GROUP))
PERIOD = list(PERIODIC_TABLE["Period"].values)
PERIOD_SET = list(set(PERIOD))

ATOM_LIST = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "Mg", "Na", "Ca", "Fe", "As", "Al", "I",
    "B", "V", "K", "Tl", "Yb", "Sb", "Sn", "Ag", "Pd", "Co", "Se", "Ti", "Zn", "H", "Li", "Ge",
    "Cu", "Au", "Ni", "Cd", "In", "Mn", "Zr", "Cr", "Pt", "Hg", "Pb"
]

ATOM_NUM_H       = list(range(0, 9))
ATOM_DEGREE_LIST = list(range(0, 13))
VALENCE          = list(range(0, 13))
CHARGE_LIST      = list(range(-8, 9))
RADICAL_E_LIST   = list(range(0, 9))     

HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.names[k]
    for k in sorted(Chem.rdchem.HybridizationType.names.keys(), reverse=True)
    if k != "OTHER"
]

CHIRALITY_LIST = ["R", "S"]

BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

BOND_STEREO = [
    Chem.rdchem.BondStereo.STEREONONE,
    Chem.rdchem.BondStereo.STEREOANY,
    Chem.rdchem.BondStereo.STEREOZ,
    Chem.rdchem.BondStereo.STEREOE,
    Chem.rdchem.BondStereo.STEREOCIS,
    Chem.rdchem.BondStereo.STEREOTRANS,
    "chi_star",
    "chi_circle"
]

NODE_FEAT_LIST_ONE_HOT: Dict[str, List[Union[int, str]]] = {
    "valence_implicit": VALENCE,
    "valence_total": VALENCE,
    "hybridization": HYBRIDIZATION_LIST,
    "atomic_number": ATOM_LIST,
    "total_num_hs": ATOM_NUM_H,
    "is_aromatic": [1],
    "is_in_ring": [1],
    "chirality": CHIRALITY_LIST,
    "period": PERIOD_SET,
    "degree": ATOM_DEGREE_LIST,
    "group": GROUP_SET,
}

NODE_FEAT_LIST_FLOAT = [
    "num_radical_electrons",
    "electronegativity",
    "first_ionization",
    "covalent_radius",
    "melting_point",
    "formal_charge",
    "vdw_radius",
    "weight",
]

# Feature indices in NODE_FEAT_LIST_FLOAT that use 0.0 as sentinel for missing
# periodic table entries. Real values are always > 0 for these features:
#   electronegativity >= 0.7, first_ionization >= 3.89 eV, melting_point >= 14.2 K
NODE_FLOAT_MISSING_IDXS = (1, 2, 4)

EDGE_FEAT_LIST_ONE_HOT: Dict[str, List[Union[int, str]]] = {
    "conjugation": [1],
    "is_in_ring": [1],
    "bond_type": BOND_TYPES,
    "stereo": BOND_STEREO,
}

# Derived indices used for stereo edge filtering (avoids hardcoding in grit.py)
BOND_TYPE_FEAT_IDX = list(EDGE_FEAT_LIST_ONE_HOT.keys()).index("bond_type")
BOND_TYPE_OTHER_CODE = len(BOND_TYPES)  # "other" bucket follows the known types
