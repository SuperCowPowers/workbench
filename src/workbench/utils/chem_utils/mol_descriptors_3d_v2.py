"""
mol_descriptors_3d_v2.py - curated, xTB-powered 3D descriptors (v2)

Why v2:
    The v1 set (74 features) never beat 2D — even on non-PXR ADMET assays. It is
    dominated by 43 collinear Gasteiger-charge CPSA descriptors and ~22 shape
    descriptors that mostly re-encode size/logP (already in the 2D set). The
    conformer engine and Boltzmann averaging are sound; the *descriptor choice*
    was the weak link.

What changed:
    Same engine (ETKDGv3 -> MMFF opt -> GFN2-xTB ranking -> Boltzmann weights,
    all reused from mol_descriptors_3d), a deliberately small descriptor layer
    (~25), and every feature chosen to add signal orthogonal to 2D:

      - Electronic (xTB): dipole, HOMO/LUMO/gap, hardness, electrophilicity,
        partial-charge summaries. Harvested from the SAME GFN2-xTB single point
        we already run for energy ranking — pure quantum signal, absent from 2D,
        essentially free. This is the headline of v2.
      - Surface: SASA total / polar / apolar / fraction, plus a charge-weighted
        polar surface area using xTB charges. Replaces 43 Gasteiger CPSA columns
        with a handful of robust ones.
      - Shape: NPR1/NPR2, asphericity, radius of gyration, spherocity, PBF.
      - Pharmacophore geometry: amphiphilic moment, charge-centroid offset,
        H-bond-acceptor-centroid offset.
      - Flexibility: conformational flexibility index, conformers in window.

    All per-conformer descriptors are Boltzmann-weighted over the energy window,
    exactly as v1.

tblite note:
    The electronic block needs tblite (GFN2-xTB), present only in the 3D
    inference image. When tblite is unavailable (or a molecule fails), the
    electronic features are NaN and the geometry/surface/shape blocks still
    compute — same graceful-degradation contract as v1. The exact tblite
    Result keys are read defensively (per-property try/except) and the set that
    succeeded is logged once, so a missing key degrades a single feature rather
    than the whole molecule.
"""

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors3D, rdMolDescriptors
from rdkit.Chem import rdFreeSASA

# Reuse the v1 engine verbatim — only the descriptor layer is new. Relative
# import so it resolves under either package name this module loads as:
# `workbench.utils.chem_utils` (library) or the symlinked `molecular_utils`
# namespace package (model-script runtime) — without requiring `workbench`
# itself to be importable in the endpoint, matching the v1 script's convention.
from .mol_descriptors_3d import (
    BOHR_PER_ANGSTROM,
    BOLTZMANN_FORCE_TOL,
    HARTREE_TO_KCAL,
    TBLITE_AVAILABLE,
    adaptive_n_conformers,
    boltzmann_weights,
    check_complexity,
    compute_amphiphilic_moment,
    compute_charge_centroid_distance,
    compute_conformer_statistics,
    compute_hba_centroid_distance,
    conformer_energies_and_method,
    generate_conformers,
    _stereo_preserved,
)

logger = logging.getLogger("workbench")

# tblite is optional (3D inference image only). Guarded import mirrors v1.
try:
    from tblite.interface import Calculator as XTBCalculator
except ImportError:  # pragma: no cover - exercised only outside the 3D image
    XTBCalculator = None

# Unit conversions for the electronic block.
HARTREE_TO_EV = 27.211386245988
AU_DIPOLE_TO_DEBYE = 2.541746473  # 1 e·a0 -> Debye
OCC_THRESHOLD = 1e-6  # an orbital counts as occupied above this occupation

# Logged once after the first successful xTB property extraction so we can
# confirm — on the inference image — exactly which Result keys are available.
_XTB_KEYS_LOGGED = False


# =============================================================================
# Electronic block (GFN2-xTB single point — energy + properties in one pass)
# =============================================================================


def _electronic_from_result(res, n_atoms: int) -> Tuple[Dict[str, float], Optional[np.ndarray]]:
    """Pull electronic descriptors from one tblite Result, defensively.

    Every property is read in its own try/except so a missing/renamed tblite
    key NaNs a single feature instead of failing the conformer. Returns
    (feature_dict, raw_charges) — the raw per-atom charge vector (or None) is
    handed back separately so the surface block can charge-weight SASA without
    polluting the averaged feature set.
    """
    global _XTB_KEYS_LOGGED
    out = _nan_electronic()
    available: List[str] = []

    # Dipole: tblite returns the dipole vector in atomic units (e·a0).
    try:
        dipole = np.asarray(res.get("dipole"), dtype=float).ravel()
        if dipole.size >= 3:
            out["elec_dipole"] = float(np.linalg.norm(dipole[:3])) * AU_DIPOLE_TO_DEBYE
            available.append("dipole")
    except Exception as e:
        logger.debug(f"xTB dipole unavailable: {e}")

    # Quadrupole: 6 packed components of the traceless symmetric tensor
    # [xx, xy, yy, xz, yz, zz]. Report the rotationally-invariant Frobenius norm.
    # Orthogonal to the dipole (e.g. benzene: zero dipole, large quadrupole).
    try:
        q6 = np.asarray(res.get("quadrupole"), dtype=float).ravel()
        if q6.size >= 6:
            xx, xy, yy, xz, yz, zz = q6[:6]
            frob = np.sqrt(xx * xx + yy * yy + zz * zz + 2.0 * (xy * xy + xz * xz + yz * yz))
            out["elec_quadrupole"] = float(frob)
            available.append("quadrupole")
    except Exception as e:
        logger.debug(f"xTB quadrupole unavailable: {e}")

    # Frontier orbitals from eigenvalues + occupations (Hartree -> eV).
    try:
        energies = np.asarray(res.get("orbital-energies"), dtype=float).ravel()
        occ = np.asarray(res.get("orbital-occupations"), dtype=float).ravel()
        occupied = energies[occ > OCC_THRESHOLD]
        virtual = energies[occ <= OCC_THRESHOLD]
        if occupied.size and virtual.size:
            homo = float(occupied.max()) * HARTREE_TO_EV
            lumo = float(virtual.min()) * HARTREE_TO_EV
            gap = lumo - homo
            out["elec_homo"] = homo
            out["elec_lumo"] = lumo
            out["elec_gap"] = gap
            # Conceptual DFT: hardness eta = gap/2; chemical potential
            # mu = (homo+lumo)/2; electrophilicity omega = mu^2/(2 eta).
            eta = gap / 2.0
            mu = (homo + lumo) / 2.0
            out["elec_hardness"] = eta
            if eta > 1e-6:
                out["elec_electrophilicity"] = (mu * mu) / (2.0 * eta)
            available.append("orbital-energies")
    except Exception as e:
        logger.debug(f"xTB orbital energies unavailable: {e}")

    # Partial-charge summaries (Mulliken-like, from xTB — far better than the
    # Gasteiger charges v1's CPSA relied on). Keep the raw vector for the
    # charge-weighted surface area.
    raw_charges: Optional[np.ndarray] = None
    try:
        q = np.asarray(res.get("charges"), dtype=float).ravel()
        if q.size:
            out["elec_qmax"] = float(q.max())
            out["elec_qmin"] = float(q.min())
            out["elec_qabs_mean"] = float(np.abs(q).mean())
            raw_charges = q
            available.append("charges")
    except Exception as e:
        logger.debug(f"xTB charges unavailable: {e}")

    if not _XTB_KEYS_LOGGED and available:
        logger.info(f"xTB v2 electronic keys available: {sorted(available)}")
        _XTB_KEYS_LOGGED = True

    return out, raw_charges


def xtb_singlepoint_properties(
    mol: Chem.Mol,
) -> Tuple[List[float], List[Dict[str, float]], List[Optional[np.ndarray]]]:
    """Per-conformer GFN2-xTB energies (kcal/mol), electronic descriptors, charges.

    One single point per conformer yields the energy (for Boltzmann weighting),
    the electronic properties, and the raw partial-charge vector — no second xTB
    pass. Returns (energies, props, charges) aligned by conformer index. When
    tblite is unavailable the energies are all-NaN (caller falls back to the
    force-field ranking), every electronic dict is all-NaN, and charges are None.
    """
    n_confs = mol.GetNumConformers()
    if not TBLITE_AVAILABLE or XTBCalculator is None:
        return [np.nan] * n_confs, [_nan_electronic() for _ in range(n_confs)], [None] * n_confs

    numbers = np.array([a.GetAtomicNum() for a in mol.GetAtoms()])
    charge = Chem.GetFormalCharge(mol)
    n_atoms = mol.GetNumAtoms()

    energies: List[float] = []
    props: List[Dict[str, float]] = []
    charges: List[Optional[np.ndarray]] = []
    for conf_id in range(n_confs):
        try:
            pos_bohr = mol.GetConformer(conf_id).GetPositions() * BOHR_PER_ANGSTROM
            calc = XTBCalculator("GFN2-xTB", numbers, pos_bohr, charge=charge)
            calc.set("verbosity", 0)
            res = calc.singlepoint()
            energies.append(float(res.get("energy")) * HARTREE_TO_KCAL)
            prop, q = _electronic_from_result(res, n_atoms)
            props.append(prop)
            charges.append(q)
        except Exception as e:
            logger.debug(f"GFN2-xTB single point failed for conf {conf_id}: {e}")
            energies.append(np.nan)
            props.append(_nan_electronic())
            charges.append(None)
    return energies, props, charges


def _nan_electronic() -> Dict[str, float]:
    return {
        "elec_dipole": np.nan,
        "elec_quadrupole": np.nan,
        "elec_homo": np.nan,
        "elec_lumo": np.nan,
        "elec_gap": np.nan,
        "elec_hardness": np.nan,
        "elec_electrophilicity": np.nan,
        "elec_qmax": np.nan,
        "elec_qmin": np.nan,
        "elec_qabs_mean": np.nan,
    }


# =============================================================================
# Surface block (rdFreeSASA, charge-weighted with xTB partial charges)
# =============================================================================


def compute_surface_descriptors(
    mol: Chem.Mol, conf_id: int = 0, charges: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """Solvent-accessible surface area split into polar / apolar.

    Uses RDKit's free-SASA (Shrake-Rupley). ``classifyAtoms`` only assigns radii
    (its Polar/Apolar tags are left "Unclassified" by the default classifier),
    so the polar/apolar split is done deterministically by element: N, O, S, P
    and the H atoms bonded to them are polar; everything else (C and its H) is
    apolar — the 3D analogue of TPSA. ``CalcSASA`` writes per-atom 'SASA' props.
    When xTB ``charges`` are supplied, also returns a charge-weighted polar
    surface area (sum of per-atom SASA * |q|) — the proper-charge replacement
    for v1's Gasteiger CPSA.
    """
    nan_result = {
        "surf_sasa_total": np.nan,
        "surf_sasa_polar": np.nan,
        "surf_sasa_apolar": np.nan,
        "surf_frac_apolar": np.nan,
        "surf_psa_charge": np.nan,
    }
    if mol is None or mol.GetNumConformers() == 0:
        return nan_result

    try:
        radii = rdFreeSASA.classifyAtoms(mol)
        total = float(rdFreeSASA.CalcSASA(mol, radii, confIdx=conf_id))
        polar = 0.0
        psa_charge = 0.0
        polar_elements = {"N", "O", "S", "P"}
        for atom in mol.GetAtoms():
            sasa = float(atom.GetPropsAsDict().get("SASA", 0.0))
            symbol = atom.GetSymbol()
            is_polar = symbol in polar_elements or (
                symbol == "H" and any(n.GetSymbol() in polar_elements for n in atom.GetNeighbors())
            )
            if is_polar:
                polar += sasa
            if charges is not None and atom.GetIdx() < len(charges):
                psa_charge += sasa * abs(float(charges[atom.GetIdx()]))
        apolar = total - polar
        return {
            "surf_sasa_total": total,
            "surf_sasa_polar": polar,
            "surf_sasa_apolar": apolar,
            "surf_frac_apolar": (apolar / total) if total > 0 else np.nan,
            "surf_psa_charge": psa_charge if charges is not None else np.nan,
        }
    except Exception as e:
        logger.debug(f"SASA computation failed: {e}")
        return nan_result


# =============================================================================
# Shape block (curated subset, no size-redundant / collinear descriptors)
# =============================================================================


def compute_shape_descriptors(mol: Chem.Mol, conf_id: int = 0) -> Dict[str, float]:
    """Six shape descriptors: NPR1/NPR2, asphericity, Rg, spherocity, PBF.

    Deliberately drops PMI1-3 (size-redundant with MW), and the Mordred
    gravitational/geometrical indices (collinear with these).
    """
    nan_result = {
        "shape_npr1": np.nan,
        "shape_npr2": np.nan,
        "shape_asphericity": np.nan,
        "shape_rgyr": np.nan,
        "shape_spherocity": np.nan,
        "shape_pbf": np.nan,
    }
    if mol is None or mol.GetNumConformers() == 0:
        return nan_result
    try:
        return {
            "shape_npr1": Descriptors3D.NPR1(mol, confId=conf_id),
            "shape_npr2": Descriptors3D.NPR2(mol, confId=conf_id),
            "shape_asphericity": Descriptors3D.Asphericity(mol, confId=conf_id),
            "shape_rgyr": Descriptors3D.RadiusOfGyration(mol, confId=conf_id),
            "shape_spherocity": Descriptors3D.SpherocityIndex(mol, confId=conf_id),
            "shape_pbf": rdMolDescriptors.CalcPBF(mol, confId=conf_id),
        }
    except Exception as e:
        logger.debug(f"Shape descriptor computation failed: {e}")
        return nan_result


# =============================================================================
# Pharmacophore-geometry block (the genuinely-3D spatial separations)
# =============================================================================


def compute_pharm_geometry(mol: Chem.Mol, conf_id: int = 0) -> Dict[str, float]:
    """Three 3D spatial separations, reusing the vetted v1 implementations."""
    return {
        "pharm_amphiphilic_moment": compute_amphiphilic_moment(mol, conf_id),
        "pharm_charge_centroid_dist": compute_charge_centroid_distance(mol, conf_id),
        "pharm_hba_centroid_dist": compute_hba_centroid_distance(mol, conf_id),
    }


# =============================================================================
# Feature / diagnostic name registries
# =============================================================================

_ELECTRONIC_NAMES = [
    "elec_dipole",
    "elec_quadrupole",
    "elec_homo",
    "elec_lumo",
    "elec_gap",
    "elec_hardness",
    "elec_electrophilicity",
    "elec_qmax",
    "elec_qmin",
    "elec_qabs_mean",
]
_SURFACE_NAMES = ["surf_sasa_total", "surf_sasa_polar", "surf_sasa_apolar", "surf_frac_apolar", "surf_psa_charge"]
_SHAPE_NAMES = ["shape_npr1", "shape_npr2", "shape_asphericity", "shape_rgyr", "shape_spherocity", "shape_pbf"]
_PHARM_NAMES = ["pharm_amphiphilic_moment", "pharm_charge_centroid_dist", "pharm_hba_centroid_dist"]
_FLEX_NAMES = ["flex_index", "flex_confs_in_window"]


def get_3d_v2_feature_names() -> List[str]:
    """All v2 feature names (25 total)."""
    return _ELECTRONIC_NAMES + _SURFACE_NAMES + _SHAPE_NAMES + _PHARM_NAMES + _FLEX_NAMES


def get_3d_v2_diagnostic_names() -> List[str]:
    """Diagnostic columns (prefixed desc3d_), mirroring v1's contract."""
    return [
        "desc3d_status",
        "desc3d_conf_count",
        "desc3d_confs_requested",
        "desc3d_confs_in_window",
        "desc3d_embed_tier",
        "desc3d_force_field",
        "desc3d_energy_method",
        "desc3d_compute_time_s",
        "desc3d_stereo_preserved",
    ]


# =============================================================================
# Boltzmann-weighted ensemble assembly
# =============================================================================


def _compute_descriptors_v2_boltzmann(mol: Chem.Mol) -> Tuple[Dict[str, float], int, str]:
    """Boltzmann-weighted v2 descriptors over the energy window.

    Energies (and electronic properties) come from one GFN2-xTB pass; if xTB is
    unavailable the energies fall back to the force field for weighting and the
    electronic block stays NaN. Surface/shape/pharm are computed per in-window
    conformer and Boltzmann-averaged; flexibility is an ensemble statistic.
    """
    energies, elec_props, charges_list = xtb_singlepoint_properties(mol)
    energy_method = "GFN2-xTB"
    if np.all(np.isnan(energies)):
        # Fall back to the force-field ranking (and keep electronic NaN).
        energies, energy_method = conformer_energies_and_method(mol, energy_method="MMFF94s")

    conf_ids, weights = boltzmann_weights(energies)

    per_conf: List[Dict[str, float]] = []
    keys: Optional[List[str]] = None
    for cid in conf_ids:
        charges = charges_list[cid] if cid < len(charges_list) else None
        d: Dict[str, float] = {}
        d.update(elec_props[cid] if cid < len(elec_props) else _nan_electronic())
        d.update(compute_surface_descriptors(mol, cid, charges=charges))
        d.update(compute_shape_descriptors(mol, cid))
        d.update(compute_pharm_geometry(mol, cid))
        per_conf.append(d)
        if keys is None:
            keys = list(d.keys())

    features: Dict[str, float] = {}
    for key in keys or []:
        values = np.array([d[key] for d in per_conf], dtype=float)
        if np.all(np.isnan(values)):
            features[key] = np.nan
            continue
        nan_mask = np.isnan(values)
        w = weights.copy()
        w[nan_mask] = 0.0
        w_sum = w.sum()
        features[key] = float(np.dot(w / w_sum, np.where(nan_mask, 0.0, values))) if w_sum > 0 else np.nan

    # Flexibility (ensemble-level): reuse v1 conformer stats for the index,
    # and report how many conformers actually drove the Boltzmann average.
    stats = compute_conformer_statistics(mol, energies=energies)
    features["flex_index"] = stats.get("conformational_flexibility", np.nan)
    features["flex_confs_in_window"] = float(len(conf_ids))

    return features, len(conf_ids), energy_method


# =============================================================================
# Entry point
# =============================================================================


def compute_descriptors_3d_v2(
    df: pd.DataFrame,
    optimize: bool = True,
    random_seed: int = 42,
    complexity_check: bool = True,
) -> pd.DataFrame:
    """Compute v2 3D descriptors for a DataFrame with a 'smiles' column.

    Mirrors compute_descriptors_3d's contract (adaptive 50/200 conformers,
    Boltzmann window, desc3d_* diagnostics, NaN on skip/failure) but emits the
    curated ~25 v2 features instead of the 74 v1 features.
    """
    smiles_column = next((c for c in df.columns if c.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    n_molecules = len(df)
    logger.info(f"Computing 3D v2 descriptors for {n_molecules} molecules...")

    feature_names = get_3d_v2_feature_names()
    diag_names = get_3d_v2_diagnostic_names()
    object_diags = {"desc3d_status", "desc3d_force_field", "desc3d_energy_method", "desc3d_stereo_preserved"}
    new_columns: Dict[str, pd.Series] = {}
    for col in feature_names:
        new_columns[col] = pd.Series(np.full(n_molecules, np.nan, dtype=np.float64), index=df.index)
    for col in diag_names:
        if col in object_diags:
            new_columns[col] = pd.Series([pd.NA] * n_molecules, dtype=object, index=df.index)
        else:
            new_columns[col] = pd.Series(np.full(n_molecules, np.nan, dtype=np.float64), index=df.index)
    result = pd.concat([df.copy(), pd.DataFrame(new_columns, index=df.index)], axis=1)
    result["desc3d_status"] = "skipped"

    start = time.time()
    for idx, row in result.iterrows():
        smiles = row[smiles_column]
        mol_start = time.time()

        if pd.isna(smiles) or smiles == "":
            result.at[idx, "desc3d_status"] = "skip:empty"
            continue
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                result.at[idx, "desc3d_status"] = "skip:parse"
                continue

            n_confs = adaptive_n_conformers(mol)
            if complexity_check:
                status = check_complexity(mol, n_conformers=n_confs)
                if status is not None:
                    result.at[idx, "desc3d_status"] = status
                    continue

            input_chirality = frozenset(
                Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
            )
            mol = Chem.AddHs(mol)
            result.at[idx, "desc3d_confs_requested"] = n_confs

            mol, gen_info = generate_conformers(
                mol, n_conformers=n_confs, random_seed=random_seed, optimize=optimize, force_tol=BOLTZMANN_FORCE_TOL
            )
            result.at[idx, "desc3d_embed_tier"] = gen_info["embed_tier"]
            result.at[idx, "desc3d_force_field"] = gen_info["force_field"]

            if mol is None or mol.GetNumConformers() == 0:
                result.at[idx, "desc3d_status"] = "skip:embed"
                continue

            result.at[idx, "desc3d_conf_count"] = mol.GetNumConformers()
            result.at[idx, "desc3d_stereo_preserved"] = _stereo_preserved(mol, input_chirality)

            features, confs_in_window, energy_method = _compute_descriptors_v2_boltzmann(mol)
            for name, value in features.items():
                if name in result.columns:
                    result.at[idx, name] = value

            result.at[idx, "desc3d_confs_in_window"] = confs_in_window
            result.at[idx, "desc3d_energy_method"] = energy_method
            result.at[idx, "desc3d_status"] = "ok"
            result.at[idx, "desc3d_compute_time_s"] = round(time.time() - mol_start, 3)

        except Exception as e:
            logger.debug(f"3D v2 descriptor calculation failed for index {idx}: {e}")
            result.at[idx, "desc3d_status"] = f"error:{type(e).__name__}"
            continue

    elapsed = time.time() - start
    valid = (result["desc3d_status"] == "ok").sum()
    logger.info(f"Computed 3D v2 descriptors for {valid}/{n_molecules} molecules in {elapsed:.2f}s")
    return result
