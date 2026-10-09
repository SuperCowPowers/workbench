from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import AllChem, rdchem, rdDetermineBonds, rdDistGeom, rdForceFieldHelpers, rdmolops
from rdkit.Geometry import Point3D

from monroe.model.constants import (
    BOND_TYPES,
    EDGE_FEAT_LIST_ONE_HOT,
    ELECTRONEGATIVITY,
    FIRST_IONIZATION,
    GROUP,
    MELTING_POINT,
    NODE_FEAT_LIST_FLOAT,
    NODE_FEAT_LIST_ONE_HOT,
    PERIOD,
)

RDLogger.DisableLog("rdApp.warning")
RDLogger.DisableLog("rdApp.error")
RDLogger.logger().setLevel(RDLogger.CRITICAL)
rdBase.DisableLog("rdApp.warning")

ptable = Chem.GetPeriodicTable()


def encode_code(val: Any, classes: Iterable[Any]) -> np.uint8:
    cl = list(classes)
    try:
        idx = cl.index(val)
    except ValueError:
        idx = len(cl)
    return np.uint8(idx)


def get_node_features_codes(mol: Chem.Mol):
    atoms = mol.GetAtoms()
    N = len(atoms)

    feat_float = np.zeros((N, len(NODE_FEAT_LIST_FLOAT)), dtype=np.float32)
    feat_codes = np.zeros((N, len(NODE_FEAT_LIST_ONE_HOT)), dtype=np.uint8)

    for i, atom in enumerate(atoms):
        z = atom.GetAtomicNum()
        # float features
        feat_float[i, 0] = atom.GetNumRadicalElectrons()
        feat_float[i, 1] = ELECTRONEGATIVITY[z - 1]
        feat_float[i, 2] = FIRST_IONIZATION[z - 1]
        feat_float[i, 3] = ptable.GetRcovalent(z)
        feat_float[i, 4] = MELTING_POINT[z - 1]
        feat_float[i, 5] = atom.GetFormalCharge()
        feat_float[i, 6] = ptable.GetRvdw(z)
        feat_float[i, 7] = atom.GetMass()

        # categorical
        if hasattr(Chem.rdchem, "ValenceType"):
            v_imp = atom.GetValence(which=Chem.rdchem.ValenceType.IMPLICIT)
            v_exp = atom.GetValence(which=Chem.rdchem.ValenceType.EXPLICIT)
        else:
            v_imp = atom.GetImplicitValence()
            v_exp = atom.GetExplicitValence()
        v_tot = v_imp + v_exp

        chirality = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None

        vals = dict(
            valence_total=v_tot,
            valence_implicit=v_imp,
            hybridization=atom.GetHybridization(),
            atomic_number=atom.GetSymbol(),
            total_num_hs=atom.GetTotalNumHs(),
            is_aromatic=int(atom.GetIsAromatic()),
            is_in_ring=int(atom.IsInRing()),
            chirality=chirality,
            period=PERIOD[z - 1],
            degree=atom.GetDegree(),
            group=GROUP[z - 1],
        )
        for j, k in enumerate(NODE_FEAT_LIST_ONE_HOT):
            feat_codes[i, j] = encode_code(vals[k], NODE_FEAT_LIST_ONE_HOT[k])

    mol_name = mol.GetProp("_Name") if mol.HasProp("_Name") else "unknown"
    assert not np.isnan(feat_float).any(), f"NaN found in feat_float for {mol_name}"
    return feat_float, feat_codes  # [N,8] float32, [N,10] uint8


def get_edge_features_codes(mol: Chem.Mol):
    E = mol.GetNumBonds()
    edge_index = np.empty((2, E), dtype=np.int32)
    edge_codes = np.empty((E, len(EDGE_FEAT_LIST_ONE_HOT)), dtype=np.uint8)

    for bi, bond in enumerate(mol.GetBonds()):
        edge_index[0, bi] = bond.GetBeginAtomIdx()
        edge_index[1, bi] = bond.GetEndAtomIdx()
        vals = dict(
            conjugation=bond.GetIsConjugated(),
            is_in_ring=bond.IsInRing(),
            bond_type=bond.GetBondType(),
            stereo=bond.GetStereo(),
        )
        for j, k in enumerate(EDGE_FEAT_LIST_ONE_HOT):
            edge_codes[bi, j] = encode_code(vals[k], EDGE_FEAT_LIST_ONE_HOT[k])
    return edge_codes, edge_index  # [E,4] uint8, [2,E] int32


def _encode_edge_row_for_virtual(stereo_val: Any) -> np.ndarray:
    vals = dict(
        conjugation=0,  # falls into "other" bucket via encode_code
        is_in_ring=0,  # falls into "other"
        bond_type=None,  # falls into "other"
        stereo=stereo_val,  # use RDKit BondStereo value
    )
    row = np.empty((len(EDGE_FEAT_LIST_ONE_HOT),), dtype=np.uint8)
    for j, k in enumerate(EDGE_FEAT_LIST_ONE_HOT):
        row[j] = encode_code(vals[k], EDGE_FEAT_LIST_ONE_HOT[k])
    return row


def get_stereo_virtual_edges(mol: Chem.Mol) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        edge_index_extra: [2, E_s] int32
        edge_codes_extra: [E_s, 4] uint8
    """
    # Real edges to avoid duplicating bonds
    real_pairs = set()
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        real_pairs.add((i, j))
        real_pairs.add((j, i))

    result_pairs: List[Tuple[int, int]] = []
    result_codes: List[np.ndarray] = []
    seen = set()  # avoid duplicates inside the virtual set

    def add_edge(u: int, v: int, stereo_val: Any, bidirectional: bool = True):
        try:
            stereo_key = int(stereo_val)
        except (TypeError, ValueError):
            stereo_key = stereo_val
        key = (u, v, stereo_key)
        if (u, v) in real_pairs or key in seen:
            return
        seen.add(key)
        row = _encode_edge_row_for_virtual(stereo_val)
        result_pairs.append((u, v))
        result_codes.append(row)
        if bidirectional:
            result_pairs.append((v, u))
            result_codes.append(row.copy())

    # --- E/Z edges around double bonds ---
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        stereo = bond.GetStereo()
        if stereo not in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ):
            continue

        idx_3, idx_4 = bond.GetStereoAtoms()
        if idx_3 < 0 or idx_4 < 0:
            continue

        a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
        i1, i2 = a1.GetIdx(), a2.GetIdx()

        idx_5 = [nbr.GetIdx() for nbr in a1.GetNeighbors() if nbr.GetIdx() not in {i2, idx_3}]
        idx_6 = [nbr.GetIdx() for nbr in a2.GetNeighbors() if nbr.GetIdx() not in {i1, idx_4}]

        inv = Chem.BondStereo.STEREOE if stereo == Chem.BondStereo.STEREOZ else Chem.BondStereo.STEREOZ

        # diagonal across the double bond
        add_edge(idx_3, idx_4, stereo)
        # cross links
        if idx_5:
            add_edge(idx_5[0], idx_4, inv)
        if idx_6:
            add_edge(idx_3, idx_6[0], inv)
        # parallel neighbors across the bond
        if idx_5 and idx_6:
            add_edge(idx_5[0], idx_6[0], stereo)

    # --- R/S local orientation edges ---
    for atom in mol.GetAtoms():
        if not atom.HasProp("_CIPCode"):
            continue
        chi = atom.GetProp("_CIPCode")
        nbrs = atom.GetNeighbors()
        if not nbrs or not all(n.HasProp("_CIPRank") for n in nbrs):
            continue

        sorted_neighbors = sorted(nbrs, key=lambda x: int(x.GetProp("_CIPRank")), reverse=True)
        if len(sorted_neighbors) < 4:
            continue

        if chi == "R":
            a, b, c = [n.GetIdx() for n in sorted_neighbors[:3]]
        else:  # "S"
            a, b, c = [n.GetIdx() for n in sorted_neighbors[:3]][::-1]
        d = sorted_neighbors[3].GetIdx()

        # star spokes and orientation cycle, use STEREOANY as neutral label
        add_edge(a, d, "chi_star")
        add_edge(b, d, "chi_star")
        add_edge(c, d, "chi_star")
        add_edge(b, a, "chi_circle", bidirectional=False)
        add_edge(c, b, "chi_circle", bidirectional=False)
        add_edge(a, c, "chi_circle", bidirectional=False)

    if not result_pairs:
        return (np.empty((2, 0), dtype=np.int32), np.empty((0, len(EDGE_FEAT_LIST_ONE_HOT)), dtype=np.uint8))

    edge_index_extra = np.asarray(result_pairs, dtype=np.int32).T
    edge_codes_extra = np.asarray(result_codes, dtype=np.uint8)
    return edge_index_extra, edge_codes_extra


def symmetrize_edges(
    edge_index: np.ndarray,
    edge_codes: np.ndarray,
    bond_type_idx: int = None,
    stereo_code: int = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Symmetrize edges by adding reverse direction for non-stereo edges.

    Stereo edges (bond_type == stereo_code) are already bidirectional from
    get_stereo_virtual_edges and should not be duplicated. Regular bonds are
    stored once per bond and need their reverse direction added for message
    passing.

    Args:
        edge_index: Edge indices of shape (2, E)
        edge_codes: Edge feature codes of shape (E, num_features)
        bond_type_idx: Column index for bond_type in edge_codes. If None,
            derived from EDGE_FEAT_LIST_ONE_HOT.
        stereo_code: The code value for "other" bond type indicating stereo
            edges. If None, derived as len(BOND_TYPES).

    Returns:
        Tuple of (symmetrized_edge_index, symmetrized_edge_codes)
    """
    if bond_type_idx is None:
        bond_type_idx = list(EDGE_FEAT_LIST_ONE_HOT.keys()).index("bond_type")
    if stereo_code is None:
        stereo_code = len(BOND_TYPES)

    is_stereo = edge_codes[:, bond_type_idx] == stereo_code
    non_stereo_mask = ~is_stereo

    # Get non-stereo edges that need to be reversed
    non_stereo_ei = edge_index[:, non_stereo_mask]
    non_stereo_ec = edge_codes[non_stereo_mask]

    # Add reverse direction for non-stereo edges
    rev_ei = non_stereo_ei[[1, 0], :]  # swap src and dst

    # Concatenate: original edges + reversed non-stereo edges
    sym_ei = np.concatenate([edge_index, rev_ei], axis=1)
    sym_ec = np.concatenate([edge_codes, non_stereo_ec], axis=0)

    return sym_ei, sym_ec


def _two_orb_cap(Zs):
    cap = 0
    for z in Zs:
        z = int(z)
        if z <= 2:
            cap += 1  # 1s
        elif z <= 18:
            cap += 4  # 2s2p
        else:
            cap += 9  # 3s3p3d
    return 2 * cap


def _align_conformers(
    reference: np.ndarray,
    moving: np.ndarray,
) -> np.ndarray:
    """
    Rigidly align `moving` conformer onto `reference` (same atom order) via Kabsch,
    preserving the reference frame and returning the aligned moving coordinates.
    """
    if reference.shape != moving.shape:
        raise ValueError("Cannot align conformers with different shapes.")
    ref = np.asarray(reference, dtype=np.float64)
    mob = np.asarray(moving, dtype=np.float64)

    ref_centroid = ref.mean(axis=0, keepdims=True)
    mob_centroid = mob.mean(axis=0, keepdims=True)

    ref_c = ref - ref_centroid
    mob_c = mob - mob_centroid

    H = mob_c.T @ ref_c
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    mob_aligned = mob_c @ R.T
    return (mob_aligned + ref_centroid).astype(moving.dtype, copy=False)


def determine_mol_from_coords(
    inchi: str,
    atomic_numbers: List[int],
    coords: List[float],
    charge: int,
) -> Chem.Mol:

    rw = rdchem.RWMol()
    for z in atomic_numbers:
        rw.AddAtom(rdchem.Atom(int(z)))
    mol = rw.GetMol()
    N = len(atomic_numbers)

    conf = rdchem.Conformer(N)
    for i in range(N):
        x, y, z = coords[3 * i : 3 * i + 3]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)

    Zs = [int(z) for z in atomic_numbers]
    two_cap = _two_orb_cap(Zs)
    e = sum(Zs) - charge
    if e < 0 or e > two_cap:
        raise ValueError(f"num_electrons is greater than twice the num_orbs for {inchi}")

    for use_hueckel in [True, False]:
        try:
            rdDetermineBonds.DetermineBonds(mol, useHueckel=use_hueckel, maxIterations=40_000)
            rdmolops.SanitizeMol(mol)
            Chem.AssignStereochemistryFrom3D(mol, replaceExistingTags=True)
            return mol
        except Exception:
            mol.RemoveAllConformers()
            mol = rw.GetMol()
            mol.AddConformer(conf, assignId=True)
    raise ValueError(f"Failed to determine bonds for {inchi}")


def build_from_mol_and_coords(
    mol: Chem.Mol,
    pos_rdkit: np.ndarray,
    pos: Optional[np.ndarray] = None,
    stereo_augmentation: Optional[bool] = True,
    symmetrize: bool = False,
) -> Dict[str, np.ndarray]:
    node_float, node_codes = get_node_features_codes(mol)
    edge_codes, edge_index = get_edge_features_codes(mol)

    if stereo_augmentation:
        edge_index_extra, edge_codes_extra = get_stereo_virtual_edges(mol)
        if edge_index_extra.size > 0:
            edge_index = np.concatenate([edge_index, edge_index_extra], axis=1)
            edge_codes = np.concatenate([edge_codes, edge_codes_extra], axis=0)

    if symmetrize:
        edge_index, edge_codes = symmetrize_edges(edge_index, edge_codes)

    return dict(
        node_float=node_float.astype(np.float32),  # [N,8]
        node_codes=node_codes.astype(np.uint8),  # [N,10]
        pos_rdkit=pos_rdkit.astype(np.float32),  # [N,3]
        pos=pos.astype(np.float32) if pos is not None else None,  # [N,3]
        edge_index=edge_index.astype(np.int32),  # [2,E]
        edge_codes=edge_codes.astype(np.uint8),  # [E,4]
    )


def predict_structure(mol: Chem.Mol, n_confs: int = 1) -> Chem.Conformer:
    params = getattr(rdDistGeom, "ETKDGv3")()
    params.enforceChirality = True
    params.useRandomCoords = True
    params.numThreads = 1
    params.maxIterations = 500
    params.randomSeed = 42  # Workbench: a SMILES always gets the same conformer
    params.timeout = 10  # Workbench: seconds, enforced inside RDKit (works off the main thread)
    mol.RemoveAllConformers()
    confs = rdDistGeom.EmbedMultipleConfs(mol, numConfs=n_confs, params=params)
    if len(confs) == 0:
        raise ValueError("Failed to predict structure with ETKDGv3.")

    # Prefer MMFF94s — best accuracy for organic small molecules. For molecules
    # MMFF94 cannot parameterize (metal complexes, salts, unusual oxidation
    # states, radicals), fall back to UFF which covers the full periodic table
    # at some accuracy cost. This keeps exotic molecules (empirically ~14% of
    # PCBA) in the dataset with real 3D coords instead of degenerating into a
    # flat 2D layout in build_single_graph's outer except.
    mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, "MMFF94s")
    if mp is not None:
        ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, mp)
        results = rdForceFieldHelpers.OptimizeMoleculeConfs(mol, ff, maxIters=200, numThreads=1)
    elif rdForceFieldHelpers.UFFHasAllMoleculeParams(mol):
        results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(mol, maxIters=200, numThreads=1)
    else:
        raise ValueError("MMFF and UFF parameters not available.")

    energies = np.array([energy for _, energy in results])
    best_idx = int(min(range(len(energies)), key=energies.__getitem__))
    best_conf_id = int(confs[best_idx])
    return mol.GetConformer(best_conf_id)


def build_single_graph(
    inchi: str,
    atomic_numbers: Optional[List[int]] = None,
    coords: Optional[List[float]] = None,
    charge: Optional[int] = None,
    stereo_augmentation: Optional[bool] = True,
    allow_2d_fallback_on_timeout: bool = False,
    symmetrize: bool = False,
) -> Dict[str, np.ndarray]:

    # inference time: build from InChi
    if atomic_numbers is None and coords is None and charge is None:
        mol = Chem.MolFromInchi(inchi, sanitize=False)
        if mol is None:
            raise ValueError(f"Skipping - failed to rebuild molecule from InChI for {inchi}")
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol, addCoords=False)
        Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
        pos = None
    else:  # training time: build from existing coords
        try:
            mol = determine_mol_from_coords(inchi, atomic_numbers, coords, charge)
            conf = mol.GetConformer(0)
            pos = np.asarray(conf.GetPositions(), dtype=np.float32)
        except Exception:
            raise ValueError(f"Skipping - failed to determine bonds for {inchi}")

    try:  # use ETKDG + MMFF94
        conf = predict_structure(mol)
        pos_rdkit = np.asarray(conf.GetPositions(), dtype=np.float32)
    except Exception as exc:
        if isinstance(exc, TimeoutError) and not allow_2d_fallback_on_timeout:
            raise
        print(f"Failed to predict structure for {inchi}: {exc}. " "Falling back to 2D conformation.")
        try:
            AllChem.Compute2DCoords(mol)
            conf = mol.GetConformer(0)
            pos_rdkit = np.asarray(conf.GetPositions(), dtype=np.float32)
        except Exception as exc:
            raise ValueError(f"Skipping - failed to compute 2D coordinates for {inchi}: {exc}")

    # Align RDKit prediction to provided coordinates so frames match.
    if pos is not None:
        pos_rdkit = _align_conformers(pos, pos_rdkit)

    return build_from_mol_and_coords(
        mol=mol,
        pos=pos,
        pos_rdkit=pos_rdkit,
        stereo_augmentation=stereo_augmentation,
        symmetrize=symmetrize,
    )
