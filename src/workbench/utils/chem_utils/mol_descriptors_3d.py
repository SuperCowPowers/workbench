"""
mol_descriptors_3d.py - 3D molecular descriptor computation for ADMET modeling

Purpose:
    Computes 3D conformer-based molecular descriptors for ADMET property prediction.
    These features capture molecular shape, size, and spatial distribution of chemical
    properties that cannot be determined from 2D structure alone.

    This module complements mol_descriptors.py (2D/topological features) by adding
    geometry-dependent features that are relevant for:
    - Membrane permeability
    - Transporter interactions (P-gp, BCRP, MRP, OATP)
    - Protein-ligand binding
    - Solubility (shape-dependent solvation)
    - General ADMET property prediction

Conformer Generation:
    Uses RDKit's ETKDGv3 (Experimental Torsion Knowledge Distance Geometry v3):
    - Distance geometry with experimental torsion angle preferences
    - Improved handling of small rings and macrocycles
    - Optional MMFF94 force field optimization

    By default generates multiple conformers to capture conformational flexibility.

Descriptor Categories:
    1. RDKit 3D Shape Descriptors (10 descriptors)
       - Principal Moments of Inertia (PMI1, PMI2, PMI3)
       - Normalized PMI Ratios (NPR1, NPR2) - rod vs disc vs sphere
       - Asphericity, Eccentricity, Spherocity Index
       - Radius of Gyration, Inertial Shape Factor

    2. Mordred 3D Descriptors (52 descriptors)
       - CPSA (43): Charged Partial Surface Area - electrostatic surface properties
       - GeometricalIndex (4): Petitjean shape indices
       - GravitationalIndex (4): Mass-weighted distance descriptors
       - PBF (1): Plane of Best Fit - molecular planarity

    3. Pharmacophore 3D Descriptors (8 descriptors)
       - Spatial distribution of pharmacophoric features
       - Amphiphilic moment (polar/nonpolar separation)
       - Intramolecular H-bond potential (chameleonicity)
       - Relevant for transporters, permeability, protein binding

    4. Conformer Ensemble Statistics (5 descriptors)
       - Energy statistics (min, range, std)
       - Conformational flexibility index
       - Conformer count

Pipeline Integration:
    This module can be used standalone or combined with mol_descriptors.py:

    Option 1: Separate endpoint (recommended for performance)
        df = compute_descriptors_3d(df)  # 3D features only

    Option 2: Combined with 2D descriptors
        df = compute_descriptors(df)     # 2D features (fast)
        df = compute_descriptors_3d(df)  # 3D features (slower)

Performance Notes:
    - Conformer generation: ~100-500ms per molecule (dominates runtime)
    - Single conformer mode: ~100ms per molecule
    - Multi-conformer mode (n=10): ~300-500ms per molecule
    - Descriptor calculation: ~1-5ms per conformer
    - Memory: ~10MB per 1000 molecules with conformers

    For high-throughput screening, consider:
    - Using single conformer mode (n_conformers=1)
    - Skipping MMFF optimization (optimize=False)
    - Using serverless endpoint with higher memory

Special Considerations:
    - Molecules that fail conformer generation get NaN values
    - Very flexible molecules (>15 rotatable bonds) may need more conformers
    - Macrocycles use specialized ETKDGv3 parameters
    - 3D features are conformer-dependent; results may vary slightly with random seed

Example Usage:
    from mol_descriptors_3d import compute_descriptors_3d

    # Standard usage (10 conformers, optimized)
    df = compute_descriptors_3d(df)

    # Fast mode (single conformer, no optimization)
    df = compute_descriptors_3d(df, n_conformers=1, optimize=False)

    # High-accuracy mode (more conformers)
    df = compute_descriptors_3d(df, n_conformers=50)

    # Get feature names
    from mol_descriptors_3d import get_3d_feature_names
    feature_names = get_3d_feature_names()

References:
    - ETKDGv3: https://doi.org/10.1021/acs.jcim.0c00025
    - RDKit 3D Descriptors: https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html
    - Mordred CPSA: https://doi.org/10.1021/ci00057a009
    - Mordred documentation: https://mordred-descriptor.github.io/documentation/
    - Amphiphilic moment: https://doi.org/10.1021/ci00057a005
    - Intramolecular H-bonds and permeability: https://doi.org/10.1021/acs.jmedchem.5b00911
"""

import logging
import re
import pandas as pd
import numpy as np
import time
from typing import Optional, List, Dict, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D, rdMolDescriptors, rdMolTransforms
from mordred import Calculator as MordredCalculator
from mordred import CPSA, GeometricalIndex, GravitationalIndex, PBF
from scipy.spatial.distance import pdist

# Per-conformer wall-clock timeout (seconds, int) enforced inside RDKit's
# EmbedMultipleConfs. Requires RDKit >= 2025.03.1.
CONFORMER_TIMEOUT_SECONDS = 10

# ---------------------------------------------------------------------------
# Boltzmann-mode constants
# ---------------------------------------------------------------------------
# Greg Landrum's RDKit blog: tighter force tolerance gives ~20% speedup
# with negligible geometry loss.  Used only in Boltzmann mode so fast mode
# stays bit-for-bit identical.
BOLTZMANN_FORCE_TOL = 0.0135

# Only conformers within this window of the MMFF minimum are included in
# the Boltzmann-weighted average (kcal/mol).
BOLTZMANN_ENERGY_WINDOW_KCAL = 5.0

# Standard temperature for Boltzmann weights.
BOLTZMANN_TEMPERATURE_K = 298.0

# Adaptive conformer counts keyed by rotatable-bond thresholds.
# The tier is selected by the first (threshold, n_confs) whose threshold is
# strictly greater than the molecule's rotatable-bond count.
# rot < 8 → 50, 8 ≤ rot ≤ 12 → 300, rot ≥ 13 → 500.
#
# The upper tiers are bumped above the datamol-style 200/300 to reduce the
# stochastic seed variance we measured on very flexible molecules (~20%
# NPR1 spread across seeds at 300 conformers for 13+ rot-bond chains).
# This is the documented path for reducing single-seed Boltzmann variance —
# more independent samples from the same seed.
#
# Future Note: multi-seed conformer pooling (run k seeds × N/k conformers,
# merge before Boltzmann averaging) would cut seed variance by ~sqrt(k)
# at the same total conformer count, but it's not documented practice in
# the small-molecule conformer-generation literature we checked. Worth
# revisiting if bigger conformer tiers still leave too much variance, or
# if we switch embedding algorithms (CONFORGE / Lyrebird) where the
# sampling trade-offs may differ.
ADAPTIVE_CONFORMER_TIERS = [(8, 50), (13, 300)]
ADAPTIVE_CONFORMER_DEFAULT = 500

logger = logging.getLogger("workbench")


# =============================================================================
# Molecular Complexity Check
# =============================================================================

# Thresholds for skipping 3D computation. Sized for the async endpoint's
# 15-minute invocation budget in Boltzmann mode (adaptive 50-300 conformers);
# the realtime endpoint's own 60s SageMaker timeout is the tighter practical
# ceiling for fast-mode calls, so we size these for the async/batch case
# and let realtime fail its own timeout on pathological inputs.
#
# Per-conformer wall-clock is still capped by CONFORMER_TIMEOUT_SECONDS
# (10s), so worst-case per-molecule = 10s × n_conformers. In practice most
# conformers finish in <1s and the theoretical ceiling doesn't materialize.
MAX_HEAVY_ATOMS = 150
MAX_ROTATABLE_BONDS = 50
MAX_RING_SYSTEMS = 10
MAX_RING_COMPLEXITY = 15  # rings + bridgehead + spiro atoms (backstop for polycyclic cages)


def check_complexity(mol: Chem.Mol) -> Optional[str]:
    """Check if a molecule is too complex for 3D conformer generation.

    Screens against size and topology thresholds (heavy atoms, rotatable bonds,
    ring count, ring complexity). Molecules that exceed any threshold get NaN
    features instead of risking excessive compute time.

    Args:
        mol: RDKit molecule object

    Returns:
        None if the molecule passes all checks, or a status string describing
        the specific failure (e.g. ``"skip:heavy_atoms"``).
    """
    if mol is None:
        return "skip:parse"

    n_heavy = mol.GetNumHeavyAtoms()
    if n_heavy > MAX_HEAVY_ATOMS:
        logger.warning(f"Skipping molecule: {n_heavy} heavy atoms > {MAX_HEAVY_ATOMS}")
        return "skip:heavy_atoms"

    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    if n_rot > MAX_ROTATABLE_BONDS:
        logger.warning(f"Skipping molecule: {n_rot} rotatable bonds > {MAX_ROTATABLE_BONDS}")
        return "skip:rot_bonds"

    ring_info = mol.GetRingInfo()
    n_rings = ring_info.NumRings()
    if n_rings > MAX_RING_SYSTEMS:
        logger.warning(f"Skipping molecule: {n_rings} rings > {MAX_RING_SYSTEMS}")
        return "skip:rings"

    # Ring complexity backstop for dense polycyclic cages.
    # Typical drug scaffolds score 2-5.
    n_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    n_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    ring_complexity = n_rings + n_bridgehead + n_spiro
    if ring_complexity > MAX_RING_COMPLEXITY:
        logger.warning(
            f"Skipping molecule: ring_complexity={ring_complexity} "
            f"(rings={n_rings} + bridgehead={n_bridgehead} + spiro={n_spiro}) "
            f"> {MAX_RING_COMPLEXITY}"
        )
        return "skip:ring_complexity"

    return None


# =============================================================================
# Conformer Generation
# =============================================================================


def generate_conformers(
    mol: Chem.Mol,
    n_conformers: int = 5,
    random_seed: int = 42,
    optimize: bool = True,
    force_tol: Optional[float] = None,
) -> Tuple[Optional[Chem.Mol], Dict[str, any]]:
    """
    Generate 3D conformers using ETKDGv3 with tiered fallback.

    Embedding strategy (3 tiers):
        1. ETKDGv3 with experimental torsion preferences
        2. ETKDGv3 with random coordinates (for difficult molecules)
        3. ETKDGv3 with random coordinates + relaxed ring constraints (for strained bridged systems)

    Optimization: MMFF94s preferred (better planar N handling), UFF fallback
    for molecules with unsupported MMFF atom types.

    Note: Expects mol with explicit Hs already added (via Chem.AddHs).
          Returns mol with Hs preserved (caller manages Hs).

    Args:
        mol: RDKit molecule with explicit Hs
        n_conformers: Number of conformers to generate
        random_seed: Random seed for reproducibility
        optimize: Whether to run force field optimization
        force_tol: Optional optimizer force tolerance for ETKDGv3.
            When set (e.g. 0.0135), gives ~20% speedup with negligible
            geometry loss. None keeps the RDKit default.

    Returns:
        Tuple of (mol, info) where mol is the molecule with conformers
        (or None on failure) and info is a dict with diagnostic fields:
        ``embed_tier`` (int), ``force_field`` (str).
    """
    info = {"embed_tier": 0, "force_field": "none", "embed_failures": 0, "timeout_failures": 0}

    if mol is None:
        return None, info

    try:
        # Silence UFFTYPER warnings during embedding
        rdkit_logger = RDLogger.logger()
        rdkit_logger.setLevel(RDLogger.ERROR)

        # Each tier relaxes constraints if the previous tier returned nothing
        embedding_tiers = [
            ("standard ETKDGv3", {}),
            ("random coordinates", {"useRandomCoords": True}),
            ("relaxed ring constraints", {"useRandomCoords": True, "useBasicKnowledge": False}),
        ]

        conf_ids = []
        for tier_idx, (tier_name, overrides) in enumerate(embedding_tiers, start=1):
            params = AllChem.ETKDGv3()
            params.randomSeed = random_seed
            params.useSmallRingTorsions = True
            params.numThreads = 0  # all available cores
            params.pruneRmsThresh = 0.5
            params.trackFailures = True
            params.timeout = CONFORMER_TIMEOUT_SECONDS  # per-conformer, C++-enforced
            if force_tol is not None:
                params.optimizerForceTol = force_tol
            for attr, value in overrides.items():
                setattr(params, attr, value)
            try:
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
            except RuntimeError as e:
                logger.debug(f"Embedding tier '{tier_name}' raised {e}")
                conf_ids = []
            if len(conf_ids) > 0:
                info["embed_tier"] = tier_idx
                # Capture embedding failure counts from this tier
                failure_counts = params.GetFailureCounts()
                info["embed_failures"] = sum(failure_counts)
                # EXCEEDED_TIMEOUT is the last entry in EmbedFailureCauses
                info["timeout_failures"] = failure_counts[-1] if failure_counts else 0
                break
            logger.debug(f"Embedding tier '{tier_name}' failed, trying next")

        if len(conf_ids) == 0:
            rdkit_logger.setLevel(RDLogger.WARNING)
            logger.warning(
                f"Conformer generation returned 0 conformers "
                f"(likely timeout >{CONFORMER_TIMEOUT_SECONDS}s on every attempt)"
            )
            return None, info

        if optimize:
            info["force_field"] = _optimize_conformers(mol)

        # Restore RDKit logging
        rdkit_logger.setLevel(RDLogger.WARNING)

        return mol, info

    except Exception as e:
        logger.warning(f"Conformer generation failed: {e}")
        RDLogger.logger().setLevel(RDLogger.WARNING)
        return None, info


def _optimize_conformers(mol: Chem.Mol) -> str:
    """Optimize all conformers using MMFF94s (preferred) or UFF fallback.

    Returns:
        Name of the force field used: ``"MMFF94s"``, ``"UFF"``, or ``"none"``.
    """
    if AllChem.MMFFHasAllMoleculeParams(mol):
        for conf_id in range(mol.GetNumConformers()):
            try:
                AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s", maxIters=200, confId=conf_id)
            except Exception as e:
                logger.debug(f"MMFF94s optimization failed for conformer {conf_id}: {e}")
        return "MMFF94s"
    elif AllChem.UFFHasAllMoleculeParams(mol):
        logger.debug("MMFF94s params unavailable, falling back to UFF")
        for conf_id in range(mol.GetNumConformers()):
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200, confId=conf_id)
            except Exception as e:
                logger.debug(f"UFF optimization failed for conformer {conf_id}: {e}")
        return "UFF"
    else:
        logger.debug("No force field params available, skipping optimization")
        return "none"


def get_conformer_energies(mol: Chem.Mol) -> List[float]:
    """
    Calculate force field energies for all conformers.

    Uses MMFF94s (preferred for drug-like molecules), falls back to UFF
    for molecules with unsupported MMFF atom types.

    Args:
        mol: RDKit molecule with conformers and explicit Hs

    Returns:
        List of energies (kcal/mol), NaN for failed calculations
    """
    if mol is None or mol.GetNumConformers() == 0:
        return []

    n_confs = mol.GetNumConformers()
    energies = []

    # Try MMFF94s first (better handling of planar nitrogen centers)
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol, mmffVariant="MMFF94s")
    if mmff_props is not None:
        for conf_id in range(n_confs):
            try:
                ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
                energies.append(ff.CalcEnergy() if ff is not None else np.nan)
            except Exception:
                energies.append(np.nan)
        return energies

    # Fall back to UFF if it has params for this molecule
    if AllChem.UFFHasAllMoleculeParams(mol):
        logger.debug("MMFF94s params unavailable for energy calc, falling back to UFF")
        for conf_id in range(n_confs):
            try:
                ff = AllChem.UFFGetMoleculeForceField(mol, confId=conf_id)
                energies.append(ff.CalcEnergy() if ff is not None else np.nan)
            except Exception:
                energies.append(np.nan)
        return energies

    # Neither force field can handle this molecule
    logger.debug("No force field params available for energy calc")
    return [np.nan] * n_confs


# =============================================================================
# Boltzmann Ensemble Helpers
# =============================================================================


def adaptive_n_conformers(mol: Chem.Mol) -> int:
    """Return the conformer count for Boltzmann mode based on rotatable bonds.

    Tiering follows the datamol-style ladder but with bumped upper tiers
    (300 / 500 instead of 200 / 300) to reduce the stochastic seed variance
    observed in single-seed Boltzmann runs on flexible molecules:
        rot_bonds < 8        → 50
        rot_bonds 8..12      → 300
        rot_bonds ≥ 13       → 500

    Args:
        mol: RDKit molecule (Hs not required for this calculation)

    Returns:
        Target number of conformers
    """
    n_rot = rdMolDescriptors.CalcNumRotatableBonds(mol)
    for threshold, n_confs in ADAPTIVE_CONFORMER_TIERS:
        if n_rot < threshold:
            return n_confs
    return ADAPTIVE_CONFORMER_DEFAULT


def boltzmann_weights(
    energies: List[float],
    e_max_kcal: float = BOLTZMANN_ENERGY_WINDOW_KCAL,
    T: float = BOLTZMANN_TEMPERATURE_K,
) -> Tuple[List[int], np.ndarray]:
    """Compute Boltzmann weights for conformers within an energy window.

    Filters conformers to those within ``e_max_kcal`` of the minimum energy,
    then returns normalized Boltzmann weights: w_i = exp(-(E_i - E_min) / kT).

    Edge cases:
        - All energies NaN → fallback to conformer 0 with weight 1.0
        - Only 1 valid energy → single conformer with weight 1.0

    Args:
        energies: Per-conformer energies in kcal/mol (NaN for failures)
        e_max_kcal: Energy window above minimum (kcal/mol)
        T: Temperature in Kelvin

    Returns:
        Tuple of (conf_indices, normalized_weights) where conf_indices are
        integer conformer IDs and normalized_weights is a numpy array that
        sums to 1.0.
    """
    # Filter valid energies
    valid = [(i, e) for i, e in enumerate(energies) if not np.isnan(e)]
    if not valid:
        return [0], np.array([1.0])

    e_min = min(e for _, e in valid)

    # Filter to energy window
    in_window = [(i, e) for i, e in valid if (e - e_min) <= e_max_kcal]
    if not in_window:
        # Shouldn't happen (minimum is always in window), but be defensive
        in_window = [min(valid, key=lambda x: x[1])]

    indices = [i for i, _ in in_window]
    deltas = np.array([e - e_min for _, e in in_window])

    # kT in kcal/mol (R = 1.987e-3 kcal/(mol·K))
    kT = 1.987e-3 * T

    raw_weights = np.exp(-deltas / kT)
    normalized = raw_weights / raw_weights.sum()

    return indices, normalized


# =============================================================================
# RDKit 3D Shape Descriptors
# =============================================================================


def compute_rdkit_3d_descriptors(mol: Chem.Mol, conf_id: int = 0) -> Dict[str, float]:
    """
    Compute RDKit's built-in 3D shape descriptors.

    Args:
        mol: RDKit molecule with conformer
        conf_id: Conformer ID to use

    Returns:
        Dictionary of descriptor name -> value
    """
    nan_result = {
        "pmi1": np.nan,
        "pmi2": np.nan,
        "pmi3": np.nan,
        "npr1": np.nan,
        "npr2": np.nan,
        "asphericity": np.nan,
        "eccentricity": np.nan,
        "inertial_shape_factor": np.nan,
        "radius_of_gyration": np.nan,
        "spherocity_index": np.nan,
    }

    if mol is None or mol.GetNumConformers() == 0:
        return nan_result

    try:
        return {
            "pmi1": Descriptors3D.PMI1(mol, confId=conf_id),
            "pmi2": Descriptors3D.PMI2(mol, confId=conf_id),
            "pmi3": Descriptors3D.PMI3(mol, confId=conf_id),
            "npr1": Descriptors3D.NPR1(mol, confId=conf_id),
            "npr2": Descriptors3D.NPR2(mol, confId=conf_id),
            "asphericity": Descriptors3D.Asphericity(mol, confId=conf_id),
            "eccentricity": Descriptors3D.Eccentricity(mol, confId=conf_id),
            "inertial_shape_factor": Descriptors3D.InertialShapeFactor(mol, confId=conf_id),
            "radius_of_gyration": Descriptors3D.RadiusOfGyration(mol, confId=conf_id),
            "spherocity_index": Descriptors3D.SpherocityIndex(mol, confId=conf_id),
        }
    except Exception as e:
        logger.debug(f"RDKit 3D descriptor calculation failed: {e}")
        return nan_result


# =============================================================================
# Mordred 3D Descriptors
# =============================================================================


def get_mordred_3d_calculator() -> MordredCalculator:
    """
    Create a Mordred calculator with 3D descriptor modules.

    Includes:
    - CPSA (43): Charged Partial Surface Area
    - GeometricalIndex (4): Petitjean shape indices
    - GravitationalIndex (4): Mass-weighted distances
    - PBF (1): Plane of Best Fit

    Returns:
        Configured Mordred Calculator
    """
    calc = MordredCalculator()
    calc.register(CPSA)
    calc.register(GeometricalIndex)
    calc.register(GravitationalIndex)
    calc.register(PBF)
    return calc


def compute_mordred_3d_descriptors(mol: Chem.Mol, conf_id: int = 0) -> Dict[str, float]:
    """
    Compute Mordred 3D descriptors on a specific conformer.

    Note: Expects mol with explicit Hs (needed for CPSA partial charge calculations).
    Mordred doesn't expose a confId parameter, so we create a single-conformer
    copy of the molecule to ensure the correct geometry is used.

    Args:
        mol: RDKit molecule with conformer(s) and explicit Hs
        conf_id: Conformer ID to compute descriptors on (default 0)

    Returns:
        Dictionary of descriptor name -> value
    """
    calc = get_mordred_3d_calculator()

    if mol is None or mol.GetNumConformers() == 0:
        # Return NaN for all descriptors
        return {str(desc): np.nan for desc in calc.descriptors}

    try:
        # Mordred doesn't accept a confId argument — it uses whatever
        # conformer(s) the molecule has.  Create a single-conformer copy
        # so the requested geometry is used.
        if mol.GetNumConformers() > 1 or conf_id != 0:
            mol_copy = Chem.RWMol(mol)
            conf = mol.GetConformer(conf_id)
            mol_copy.RemoveAllConformers()
            mol_copy.AddConformer(Chem.Conformer(conf), assignId=True)
            mol = mol_copy.GetMol()

        result_df = calc.pandas([mol], nproc=1, quiet=True)

        # Convert to dictionary with sanitized column names
        result = {}
        for col in result_df.columns:
            value = result_df[col].iloc[0]
            # Handle Mordred error values
            if hasattr(value, "error"):
                value = np.nan
            else:
                try:
                    value = float(value)
                except (ValueError, TypeError):
                    value = np.nan
            # Sanitize column name and add m3d_ prefix to avoid collisions with 2D descriptors
            # (e.g., tpsa exists in both 2D RDKit and 3D Mordred CPSA)
            clean_name = re.sub(r"[^a-z0-9_]", "_", str(col).lower())
            clean_name = re.sub(r"_+", "_", clean_name)
            clean_name = f"m3d_{clean_name}"
            result[clean_name] = value

        return result

    except Exception as e:
        logger.debug(f"Mordred 3D descriptor calculation failed: {e}")
        return {str(desc): np.nan for desc in calc.descriptors}


def get_mordred_3d_feature_names() -> List[str]:
    """Get sanitized names of all Mordred 3D descriptors (with m3d_ prefix)."""
    calc = get_mordred_3d_calculator()
    names = []
    for desc in calc.descriptors:
        clean_name = re.sub(r"[^a-z0-9_]", "_", str(desc).lower())
        clean_name = re.sub(r"_+", "_", clean_name)
        clean_name = f"m3d_{clean_name}"
        names.append(clean_name)
    return names


# =============================================================================
# Pharmacophore 3D Descriptors
# =============================================================================


_NITRO_SMARTS = Chem.MolFromSmarts("[#7+](=[#8])[#8-]")


def _nitro_atom_indices(mol: Chem.Mol) -> set:
    """Return set of atom indices that belong to nitro groups (the N+ and both O's).

    The nitro N has no lone pair available — both are consumed in the N=O /
    N-O[-] bonds — so it cannot accept H-bonds even though a naive 'N with
    zero H' filter would include it. The two oxygens are likewise
    electron-deficient and are not treated as classical H-bond acceptors
    by medicinal-chemistry convention. Used to exclude these atoms from
    compute_hba_centroid_distance and compute_intramolecular_hbond_potential.
    """
    indices = set()
    if _NITRO_SMARTS is None:
        return indices
    for match in mol.GetSubstructMatches(_NITRO_SMARTS):
        indices.update(match)
    return indices


def _get_atom_positions_and_masses(mol: Chem.Mol, conf_id: int = 0, include_hs: bool = False) -> tuple:
    """
    Get atom positions and masses as numpy arrays.

    The pipeline keeps Hs attached because Mordred 3D descriptors (CPSA)
    need them for partial-charge calculations. Most pharmacophore
    descriptors use the cheminformatics convention of heavy-atom-only
    geometry (axis length, CoM centroids), so the default filters Hs.

    Pass ``include_hs=True`` for descriptors where H positions are
    physically meaningful — notably molecular volume (the convex hull
    of atom centers is a crude approximation of van der Waals volume
    and needs H atoms to be non-degenerate for small molecules).

    Returns:
        Tuple of (positions array [N, 3], masses array [N]), or (None, None) if failed
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None, None

    conf = mol.GetConformer(conf_id)
    positions = []
    masses = []

    for atom in mol.GetAtoms():
        if not include_hs and atom.GetAtomicNum() == 1:
            continue
        pos = conf.GetAtomPosition(atom.GetIdx())
        positions.append([pos.x, pos.y, pos.z])
        masses.append(atom.GetMass())

    if not positions:
        return None, None

    return np.array(positions), np.array(masses)


def compute_molecular_axis_length(mol: Chem.Mol, conf_id: int = 0) -> float:
    """
    Calculate the maximum distance between any two heavy atoms.

    This captures molecular elongation, which is relevant for:
    - Transporter binding (P-gp prefers elongated molecules ~25-30 Angstroms)
    - Membrane spanning
    - Binding site fit
    """
    positions, _ = _get_atom_positions_and_masses(mol, conf_id)
    if positions is None or len(positions) < 2:
        return np.nan

    distances = pdist(positions, metric="euclidean")
    return float(np.max(distances))


def compute_molecular_volume_3d(mol: Chem.Mol, conf_id: int = 0) -> float:
    """
    Calculate molecular volume using RDKit's grid-based van der Waals volume.

    Uses ``AllChem.ComputeMolVolume`` at 0.5 Angstrom grid spacing: places
    every atom's van der Waals sphere on the 3D conformer and counts grid
    cells inside the union. 0.5 A gives ML-quality volume estimates at ~4x
    the speed of the 0.2 A default; the finer grid is only needed for
    docking-accuracy work.

    Relevant for:
    - Binding site fit
    - Transporter size constraints
    - Solubility prediction
    """
    if mol is None or mol.GetNumConformers() == 0:
        return np.nan
    try:
        return float(AllChem.ComputeMolVolume(mol, confId=conf_id, gridSpacing=0.5))
    except Exception:
        return np.nan


def compute_amphiphilic_moment(mol: Chem.Mol, conf_id: int = 0) -> float:
    """
    Calculate distance between centroids of polar and nonpolar atom regions.

    Polar atoms: N, O, S, P, and halogens (F, Cl, Br, I).
    Nonpolar atoms: carbons not adjacent to N, O, S, or P (halogen-bonded
    carbons are classified as nonpolar — C-F/C-Cl are only weakly polarized
    and would otherwise bias the amphiphilic signal on heavily halogenated
    scaffolds).

    Higher values indicate clearer amphiphilic character, which affects:
    - Membrane partitioning and orientation
    - Transporter recognition (P-gp substrates are typically amphipathic)
    - Permeability
    """
    if mol is None or mol.GetNumConformers() == 0:
        return np.nan

    conf = mol.GetConformer(conf_id)
    polar_positions = []
    nonpolar_positions = []

    for atom in mol.GetAtoms():
        pos = conf.GetAtomPosition(atom.GetIdx())
        pos_array = np.array([pos.x, pos.y, pos.z])
        symbol = atom.GetSymbol()

        # Polar atoms: N, O, S, P, halogens
        if symbol in ["N", "O", "S", "P", "F", "Cl", "Br", "I"]:
            polar_positions.append(pos_array)
        elif symbol == "C":
            # Carbon is nonpolar if not adjacent to heteroatoms
            neighbors = atom.GetNeighbors()
            has_polar_neighbor = any(n.GetSymbol() in ["N", "O", "S", "P"] for n in neighbors)
            if not has_polar_neighbor:
                nonpolar_positions.append(pos_array)

    if not polar_positions or not nonpolar_positions:
        return 0.0

    polar_centroid = np.mean(polar_positions, axis=0)
    nonpolar_centroid = np.mean(nonpolar_positions, axis=0)

    return float(np.linalg.norm(polar_centroid - nonpolar_centroid))


def compute_charge_centroid_distance(mol: Chem.Mol, conf_id: int = 0) -> float:
    """
    Distance from the molecular center of mass to the centroid of potential
    charge-bearing nitrogens.

    The charge-site filter selects nitrogens that are either (a) already
    carrying a positive formal charge (quaternary ammonium, pyridinium,
    permanent cations that survive standardize's ChargeParent), (b) carrying
    at least one H (primary/secondary/tertiary amine, amide, N-H azole —
    potential protonation sites), or (c) aromatic (often basic, e.g.
    pyridine). This is a structural proxy for "N that bears or could bear
    positive charge" — it's broader than strictly-basic amines (also catches
    amide N-H, aromatic non-basic N like pyrrole, etc.). For strict basicity,
    use pKa prediction as a preprocessing step.

    The center-of-mass calculation includes H atoms so it matches the physical
    definition (and stays consistent with RDKit's Descriptors3D.PMI /
    RadiusOfGyration, which also use all atoms). The centroid of charge-site
    nitrogens uses only the N atom positions (heavy-atom only, by construction).

    Returns 0.0 when no qualifying nitrogens are present.

    Captures whether potential charge centers are peripheral or central, which
    affects:
    - Transporter recognition
    - Binding orientation
    - Membrane interaction
    """
    positions, masses = _get_atom_positions_and_masses(mol, conf_id, include_hs=True)
    if positions is None:
        return np.nan

    conf = mol.GetConformer(conf_id)
    com = np.average(positions, axis=0, weights=masses)

    # Charge-site nitrogens: (a) already-cationic N (formal charge > 0,
    # e.g. quaternary ammonium, pyridinium, permanent cations that survive
    # standardize's ChargeParent); (b) N carrying at least one H
    # (primary/secondary/tertiary amine, amide, N-H azole — potential
    # protonation sites); (c) aromatic N (often basic, e.g. pyridine).
    # includeNeighbors=True is required because the pipeline calls Chem.AddHs
    # before descriptors run — the default GetTotalNumHs(False) returns 0 for
    # every N in that mol, which would miss aliphatic amines entirely.
    n_positions = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() != "N":
            continue
        is_charge_site = (
            atom.GetFormalCharge() > 0 or atom.GetTotalNumHs(includeNeighbors=True) > 0 or atom.GetIsAromatic()
        )
        if is_charge_site:
            pos = conf.GetAtomPosition(atom.GetIdx())
            n_positions.append([pos.x, pos.y, pos.z])

    if len(n_positions) == 0:
        return 0.0

    n_centroid = np.mean(n_positions, axis=0)
    return float(np.linalg.norm(com - n_centroid))


def compute_nitrogen_span(mol: Chem.Mol, conf_id: int = 0) -> float:
    """
    Maximum pairwise distance between any two nitrogen atoms in the molecule.

    This includes ALL nitrogens — no filter on basicity, protonation state,
    or bonding pattern. Amide, nitro, cyano, aromatic, and aliphatic N are
    all counted equally. For chemistry-aware filtering, use
    compute_charge_centroid_distance (protonation proxy) or a pharmacophore
    feature-match library.

    Returns 0.0 when fewer than two N atoms are present.

    Captures the overall spatial spread of nitrogen-containing groups,
    relevant for:
    - Multi-point binding interactions
    - Transporter recognition patterns
    """
    if mol is None or mol.GetNumConformers() == 0:
        return np.nan

    conf = mol.GetConformer(conf_id)
    n_positions = []

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N":
            pos = conf.GetAtomPosition(atom.GetIdx())
            n_positions.append([pos.x, pos.y, pos.z])

    if len(n_positions) < 2:
        return 0.0

    distances = pdist(np.array(n_positions), metric="euclidean")
    return float(np.max(distances))


def compute_hba_centroid_distance(mol: Chem.Mol, conf_id: int = 0) -> float:
    """
    Distance from the molecular center of mass to the centroid of pure
    H-bond acceptors.

    The acceptor filter selects all oxygens (any O — alcohol, ether, carbonyl)
    plus only those nitrogens with zero attached H atoms (pyridine-like
    aromatic N, tertiary amines, nitriles, imines). This deliberately excludes
    primary/secondary amines, amides, and N-H azoles — they have lone pairs
    and CAN accept H-bonds, but they also donate, so the descriptor's intent
    is to capture pure-acceptor spatial distribution, distinct from the mixed
    donor-acceptor nitrogens counted by compute_charge_centroid_distance.

    The center-of-mass calculation includes H atoms so it matches the physical
    definition (consistent with RDKit's Descriptors3D.PMI / RadiusOfGyration).
    The acceptor centroid uses only the heavy-atom (O or no-H-N) positions.

    Returns 0.0 when no qualifying acceptors are present.

    Captures spatial distribution of pure H-bond acceptors, relevant for:
    - Binding interactions
    - Solubility
    - Permeability
    """
    positions, masses = _get_atom_positions_and_masses(mol, conf_id, include_hs=True)
    if positions is None:
        return np.nan

    conf = mol.GetConformer(conf_id)
    com = np.average(positions, axis=0, weights=masses)

    # Find H-bond acceptors: all O, plus N without any H (pure acceptor like
    # tertiary/aromatic N). Excluded:
    #   - Nitro N and its two O's (lone pairs consumed by N+=O / N-O- bonds)
    #   - Any N with positive formal charge (quaternary ammonium, protonated
    #     species) — the lone pair is consumed by the extra bond/proton
    # includeNeighbors=True is required because the pipeline calls
    # Chem.AddHs before descriptors run — default GetTotalNumHs returns
    # 0 for every N in a mol with explicit graph Hs.
    nitro_indices = _nitro_atom_indices(mol)
    hba_positions = []
    for atom in mol.GetAtoms():
        if atom.GetIdx() in nitro_indices:
            continue
        symbol = atom.GetSymbol()
        if symbol == "O":
            pos = conf.GetAtomPosition(atom.GetIdx())
            hba_positions.append([pos.x, pos.y, pos.z])
        elif symbol == "N" and atom.GetTotalNumHs(includeNeighbors=True) == 0 and atom.GetFormalCharge() <= 0:
            pos = conf.GetAtomPosition(atom.GetIdx())
            hba_positions.append([pos.x, pos.y, pos.z])

    if len(hba_positions) == 0:
        return 0.0

    hba_centroid = np.mean(hba_positions, axis=0)
    return float(np.linalg.norm(com - hba_centroid))


IMHB_MIN_ANGLE_DEG = 120.0


def compute_intramolecular_hbond_potential(mol: Chem.Mol, conf_id: int = 0) -> int:
    """
    Estimate potential for intramolecular hydrogen bonds.

    Counts donor-acceptor pairs that satisfy all three geometric criteria:
      1. D...A heavy-atom distance 2.5-3.5 Angstroms
      2. Topological separation 4-6 bonds (favors 6-, 7-, 8-membered IMHB
         pseudo-rings)
      3. D-H...A angle >= 120 degrees (rules out geometrically-close but
         non-linear pairs that can't form an H-bond)

    The angle check requires explicit Hs on the molecule; the pipeline
    calls Chem.AddHs before descriptors run, so this precondition holds.
    Donors with no explicit H neighbors are silently skipped.

    Intramolecular H-bonds enable "chameleonic" behavior:
    - Molecules can mask polar groups in nonpolar membrane environments
    - This increases membrane permeability
    - And can reduce transporter recognition

    Higher values indicate greater potential for conformational polarity masking.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return 0

    conf = mol.GetConformer(conf_id)

    # HBD atoms (N-H, O-H)
    hbd_smarts = Chem.MolFromSmarts("[#7H1,#7H2,#7H3,#8H1]")
    hbd_matches = mol.GetSubstructMatches(hbd_smarts) if hbd_smarts else []
    hbd_indices = [m[0] for m in hbd_matches]

    # HBA atoms (O, N with lone pairs). Excluded:
    #   - Nitro N/O (lone pairs consumed by N+=O / N-O- bonds)
    #   - Any atom with positive formal charge (quaternary ammonium N+,
    #     protonated species, oxocarbenium O+, etc. — no available lone pair)
    hba_smarts = Chem.MolFromSmarts("[#7X2,#7X3,#8X1,#8X2]")
    hba_matches = mol.GetSubstructMatches(hba_smarts) if hba_smarts else []
    nitro_indices = _nitro_atom_indices(mol)
    hba_indices = [
        m[0] for m in hba_matches if m[0] not in nitro_indices and mol.GetAtomWithIdx(m[0]).GetFormalCharge() <= 0
    ]

    if not hbd_indices or not hba_indices:
        return 0

    imhb_count = 0

    for d_idx in hbd_indices:
        d_atom = mol.GetAtomWithIdx(d_idx)
        d_pos = conf.GetAtomPosition(d_idx)
        h_neighbors = [n.GetIdx() for n in d_atom.GetNeighbors() if n.GetAtomicNum() == 1]

        for a_idx in hba_indices:
            # Skip same atom or directly bonded
            if d_idx == a_idx:
                continue

            bond = mol.GetBondBetweenAtoms(d_idx, a_idx)
            if bond is not None:
                continue

            a_pos = conf.GetAtomPosition(a_idx)
            dist = d_pos.Distance(a_pos)
            if not (2.5 <= dist <= 3.5):
                continue

            # Topological separation: GetShortestPath returns atoms, so a
            # path of N atoms = N-1 bonds. 4-6 bonds covers the favorable
            # 6-, 7-, and 8-membered IMHB pseudo-rings (salicylate-like
            # geometry and the β-blocker-style 7-membered case).
            path = Chem.GetShortestPath(mol, d_idx, a_idx)
            if not path or not (5 <= len(path) <= 7):
                continue

            # Angle check: at least one H on the donor must point toward the
            # acceptor with a D-H...A angle >= IMHB_MIN_ANGLE_DEG.
            angle_ok = any(
                rdMolTransforms.GetAngleDeg(conf, d_idx, h_idx, a_idx) >= IMHB_MIN_ANGLE_DEG
                for h_idx in h_neighbors
            )
            if angle_ok:
                imhb_count += 1

    return imhb_count


def compute_pharmacophore_3d_descriptors(mol: Chem.Mol, conf_id: int = 0) -> Dict[str, float]:
    """
    Compute all Pharmacophore 3D descriptors.

    These capture spatial distribution of pharmacophoric features relevant for:
    - Transporter interactions (P-gp, BCRP, etc.)
    - Membrane permeability
    - Protein binding

    Args:
        mol: RDKit molecule with conformer
        conf_id: Conformer ID to use

    Returns:
        Dictionary of descriptor name -> value (8 descriptors)
    """
    axis = compute_molecular_axis_length(mol, conf_id)
    volume = compute_molecular_volume_3d(mol, conf_id)
    if np.isnan(axis) or np.isnan(volume) or volume <= 0:
        elongation = np.nan
    else:
        elongation = axis / (volume ** (1 / 3))

    return {
        "pharm3d_molecular_axis": axis,
        "pharm3d_molecular_volume": volume,
        "pharm3d_amphiphilic_moment": compute_amphiphilic_moment(mol, conf_id),
        "pharm3d_charge_centroid_dist": compute_charge_centroid_distance(mol, conf_id),
        "pharm3d_nitrogen_span": compute_nitrogen_span(mol, conf_id),
        "pharm3d_hba_centroid_dist": compute_hba_centroid_distance(mol, conf_id),
        "pharm3d_imhb_potential": compute_intramolecular_hbond_potential(mol, conf_id),
        "pharm3d_elongation": elongation,
    }


def get_pharmacophore_3d_feature_names() -> List[str]:
    """Get names of all Pharmacophore 3D descriptors."""
    return [
        "pharm3d_molecular_axis",
        "pharm3d_molecular_volume",
        "pharm3d_amphiphilic_moment",
        "pharm3d_charge_centroid_dist",
        "pharm3d_nitrogen_span",
        "pharm3d_hba_centroid_dist",
        "pharm3d_imhb_potential",
        "pharm3d_elongation",
    ]


# =============================================================================
# Conformer Ensemble Statistics
# =============================================================================


def compute_conformer_statistics(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute statistics across the conformer ensemble.

    These capture conformational flexibility which affects binding and permeability.
    """
    if mol is None or mol.GetNumConformers() == 0:
        return {
            "conf_energy_min": np.nan,
            "conf_energy_range": np.nan,
            "conf_energy_std": np.nan,
            "conformational_flexibility": np.nan,
        }

    energies = get_conformer_energies(mol)
    valid_energies = [e for e in energies if not np.isnan(e)]

    if not valid_energies:
        return {
            "conf_energy_min": np.nan,
            "conf_energy_range": np.nan,
            "conf_energy_std": np.nan,
            "conformational_flexibility": np.nan,
        }

    energy_min = min(valid_energies)
    energy_max = max(valid_energies)
    energy_range = energy_max - energy_min
    energy_std = np.std(valid_energies) if len(valid_energies) > 1 else 0.0

    # Flexibility index: normalized energy range
    # Higher values = more conformational flexibility
    flexibility = energy_range / (1.0 + abs(energy_min)) if energy_min != 0 else energy_range

    return {
        "conf_energy_min": energy_min,
        "conf_energy_range": energy_range,
        "conf_energy_std": energy_std,
        "conformational_flexibility": flexibility,
    }


# =============================================================================
# Main Descriptor Computation Function
# =============================================================================


def get_3d_feature_names() -> List[str]:
    """
    Return list of all 3D feature names computed by this module.

    Returns:
        List of feature column names (74 total):
        - 10 RDKit 3D shape
        - 52 Mordred 3D (CPSA, Geometrical, Gravitational, PBF)
        - 8 Pharmacophore 3D
        - 4 Conformer statistics
    """
    rdkit_names = [
        "pmi1",
        "pmi2",
        "pmi3",
        "npr1",
        "npr2",
        "asphericity",
        "eccentricity",
        "inertial_shape_factor",
        "radius_of_gyration",
        "spherocity_index",
    ]

    mordred_names = get_mordred_3d_feature_names()

    pharmacophore_names = get_pharmacophore_3d_feature_names()

    conformer_names = [
        "conf_energy_min",
        "conf_energy_range",
        "conf_energy_std",
        "conformational_flexibility",
    ]

    return rdkit_names + mordred_names + pharmacophore_names + conformer_names


def get_3d_diagnostic_names() -> List[str]:
    """Return the diagnostic column names added alongside the 74 features.

    These columns are prefixed with ``desc3d_`` to distinguish them from
    model-input features. They can be filtered out with:
    ``[c for c in df.columns if not c.startswith("desc3d_")]``
    """
    return [
        "desc3d_status",
        "desc3d_mode",
        "desc3d_conf_count",
        "desc3d_confs_requested",
        "desc3d_confs_in_window",
        "desc3d_embed_failures",
        "desc3d_timeout_failures",
        "desc3d_embed_tier",
        "desc3d_force_field",
        "desc3d_compute_time_s",
        "desc3d_stereo_preserved",
    ]


def _stereo_preserved(mol_with_conf: Chem.Mol, input_chirality: frozenset) -> bool:
    """Return True if the embedded 3D geometry reproduces the input stereo.

    Compares the input's assigned chiral centers (captured before embedding)
    against the stereo re-derived from the 3D coordinates of the first
    conformer. Compounds with no assigned stereo return True (trivially
    preserved — nothing to drift from).

    Args:
        mol_with_conf: Mol with at least one conformer, post-embedding.
        input_chirality: frozenset of (atom_idx, CIP_code) tuples captured
            from the input mol before Chem.AddHs.

    Returns:
        True if stereo matches or there was no stereo to preserve, False if
        any center drifted during embedding.
    """
    if not input_chirality:
        return True
    try:
        mol_copy = Chem.Mol(mol_with_conf)
        Chem.AssignStereochemistryFrom3D(mol_copy, confId=0, replaceExistingTags=True)
        post_chirality = frozenset(
            Chem.FindMolChiralCenters(mol_copy, includeUnassigned=False, useLegacyImplementation=False)
        )
        return post_chirality == input_chirality
    except Exception as e:
        logger.debug(f"Stereo preservation check failed: {e}")
        return False


def _compute_descriptors_boltzmann(mol: Chem.Mol) -> Tuple[Dict[str, float], int]:
    """Compute Boltzmann-weighted ensemble descriptors.

    Computes energies for all conformers, selects those within the energy
    window, then returns Boltzmann-weighted averages of RDKit shape, Mordred
    3D, and pharmacophore descriptors. Conformer ensemble statistics are
    computed over the *full* generated ensemble (not just the window).

    This is the single descriptor-aggregation path used by both fast and
    Boltzmann modes. With few conformers (fast mode, n=10) the Boltzmann
    window typically includes all of them and the weighting naturally
    emphasizes the lowest-energy geometry. With many conformers (Boltzmann
    mode, n=50-300) it gives a proper ensemble average.

    Args:
        mol: RDKit molecule with conformer(s) and explicit Hs

    Returns:
        Tuple of (features_dict, confs_in_window) where confs_in_window is
        the number of conformers that contributed to the Boltzmann average.
    """
    energies = get_conformer_energies(mol)
    conf_ids, weights = boltzmann_weights(energies)

    # Collect per-conformer descriptors for all conformers in the window
    descriptor_keys = None
    per_conf_values: List[Dict[str, float]] = []

    for cid in conf_ids:
        d: Dict[str, float] = {}
        d.update(compute_rdkit_3d_descriptors(mol, cid))
        d.update(compute_mordred_3d_descriptors(mol, cid))
        d.update(compute_pharmacophore_3d_descriptors(mol, cid))
        per_conf_values.append(d)
        if descriptor_keys is None:
            descriptor_keys = list(d.keys())

    # Boltzmann-weighted average
    features: Dict[str, float] = {}
    for key in descriptor_keys:
        values = np.array([d[key] for d in per_conf_values], dtype=float)
        if np.all(np.isnan(values)):
            features[key] = np.nan
        else:
            nan_mask = np.isnan(values)
            w = weights.copy()
            w[nan_mask] = 0.0
            w_sum = w.sum()
            if w_sum > 0:
                values_clean = np.where(nan_mask, 0.0, values)
                features[key] = float(np.dot(w / w_sum, values_clean))
            else:
                features[key] = np.nan

    # Conformer ensemble stats are computed over the full generated ensemble,
    # not just the Boltzmann window.
    features.update(compute_conformer_statistics(mol))
    return features, len(conf_ids)


def compute_descriptors_3d(
    df: pd.DataFrame,
    mode: str = "fast",
    n_conformers: int = 10,
    optimize: bool = True,
    random_seed: int = 42,
    complexity_check: bool = True,
) -> pd.DataFrame:
    """
    Compute 3D molecular descriptors for ADMET modeling.

    Two modes:
        - ``"fast"`` (default): Fixed n_conformers (default 10), Boltzmann-
          weighted descriptors across the generated ensemble. Designed for
          realtime SageMaker endpoints.
        - ``"full"``: Adaptive n_conformers (50-300 based on rotatable
          bonds), same Boltzmann-weighted descriptors. Designed for overnight
          batch processing where higher conformer counts improve reproducibility.

    Both modes use the same descriptor aggregation (Boltzmann-weighted
    ensemble average over a 5 kcal/mol energy window) and produce the same
    74 output features, so downstream models can consume either pipeline's
    output interchangeably.

    Args:
        df: Input DataFrame with SMILES column
        mode: ``"fast"`` or ``"full"`` (default ``"fast"``)
        n_conformers: Number of conformers to generate (default 10, fast mode only;
                      Boltzmann mode uses adaptive counts and ignores this)
        optimize: Whether to run MMFF optimization (default True)
        random_seed: Random seed for conformer generation (default 42)
        complexity_check: Whether to skip molecules that exceed complexity thresholds
                         (default True). Set False for local analysis of complex molecules.

    Returns:
        DataFrame with 74 additional 3D descriptor columns:
        - 10 RDKit 3D shape descriptors
        - 52 Mordred 3D descriptors (CPSA, Geometrical, Gravitational, PBF)
        - 8 Pharmacophore 3D descriptors
        - 4 Conformer ensemble statistics

    Example:
        df = compute_descriptors_3d(df)                       # Fast (default)
        df = compute_descriptors_3d(df, mode="full")     # Boltzmann ensemble
    """
    if mode not in ("fast", "full"):
        raise ValueError(f"mode must be 'fast' or 'full', got '{mode}'")
    is_full = mode == "full"

    # Find SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    result = df.copy()
    n_molecules = len(df)

    n_confs_desc = "adaptive (50/200/300)" if is_full else f"{n_conformers}"
    logger.info(f"Computing 3D descriptors for {n_molecules} molecules (mode={mode})...")
    logger.info(f"Parameters: n_conformers={n_confs_desc}, optimize={optimize}, " f"force_tol={BOLTZMANN_FORCE_TOL}")

    # Initialize feature + diagnostic columns. Numeric diagnostics get np.nan;
    # string and bool diagnostics get object dtype so per-row assignment of
    # non-float values doesn't raise pandas FutureWarnings.
    feature_names = get_3d_feature_names()
    diag_names = get_3d_diagnostic_names()
    object_dtype_diagnostics = {
        "desc3d_status",
        "desc3d_mode",
        "desc3d_force_field",
        "desc3d_stereo_preserved",
    }
    for col in feature_names:
        result[col] = np.nan
    for col in diag_names:
        result[col] = pd.Series([pd.NA] * len(result), dtype=object) if col in object_dtype_diagnostics else np.nan
    result["desc3d_status"] = "skipped"
    result["desc3d_mode"] = mode

    start_time = time.time()

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

            if complexity_check:
                complexity_status = check_complexity(mol)
                if complexity_status is not None:
                    result.at[idx, "desc3d_status"] = complexity_status
                    continue

            # Snapshot the defined chiral centers so we can verify the 3D
            # embedding reproduces them. Stored as a set of (atom_idx, code)
            # tuples, restricted to CENTERS WITH ASSIGNED STEREO — undefined
            # centers are reported separately by mol_standardize.
            input_chirality = frozenset(
                Chem.FindMolChiralCenters(mol, includeUnassigned=False, useLegacyImplementation=False)
            )

            mol = Chem.AddHs(mol)

            # Conformer generation — mode only affects count
            n_confs = adaptive_n_conformers(mol) if is_full else n_conformers
            result.at[idx, "desc3d_confs_requested"] = n_confs

            mol, gen_info = generate_conformers(
                mol,
                n_conformers=n_confs,
                random_seed=random_seed,
                optimize=optimize,
                force_tol=BOLTZMANN_FORCE_TOL,
            )
            result.at[idx, "desc3d_embed_tier"] = gen_info["embed_tier"]
            result.at[idx, "desc3d_force_field"] = gen_info["force_field"]
            result.at[idx, "desc3d_embed_failures"] = gen_info["embed_failures"]
            result.at[idx, "desc3d_timeout_failures"] = gen_info["timeout_failures"]

            if mol is None or mol.GetNumConformers() == 0:
                result.at[idx, "desc3d_status"] = "skip:embed"
                continue

            result.at[idx, "desc3d_conf_count"] = mol.GetNumConformers()

            # Verify the embedded 3D geometry reproduces the input stereo. Only
            # meaningful when the input actually had assigned stereo; compounds
            # with no chiral centers are trivially preserved (True).
            result.at[idx, "desc3d_stereo_preserved"] = _stereo_preserved(mol, input_chirality)

            # Both modes: Boltzmann-weighted ensemble descriptors
            features, confs_in_window = _compute_descriptors_boltzmann(mol)

            for name, value in features.items():
                if name in result.columns:
                    result.at[idx, name] = value

            result.at[idx, "desc3d_confs_in_window"] = confs_in_window
            result.at[idx, "desc3d_status"] = "ok"
            result.at[idx, "desc3d_compute_time_s"] = round(time.time() - mol_start, 3)

        except Exception as e:
            logger.debug(f"3D descriptor calculation failed for index {idx}: {e}")
            result.at[idx, "desc3d_status"] = f"error:{type(e).__name__}"
            continue

    elapsed = time.time() - start_time
    valid_count = (result["desc3d_status"] == "ok").sum()
    throughput = n_molecules / elapsed if elapsed > 0 else 0

    logger.info(f"Computed 3D descriptors for {valid_count}/{n_molecules} molecules")
    logger.info(f"Time: {elapsed:.2f}s ({throughput:.1f} mol/s)")

    return result


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    # Configure pandas display
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1200)
    pd.set_option("display.max_colwidth", 50)

    print("=" * 80)
    print("mol_descriptors_3d.py - 3D Molecular Descriptor Tests")
    print("=" * 80)

    # Test molecules with diverse properties
    test_data = pd.DataFrame(
        {
            "smiles": [
                "CCO",  # Ethanol - small, simple
                "c1ccccc1",  # Benzene - flat aromatic
                "CC(C)NCC(O)c1ccc(O)c(O)c1",  # Isoproterenol - drug-like
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine - rigid
                "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
                "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone - steroid
                "C[C@H](N)C(=O)O",  # L-Alanine
                "CCCCCCCCCC",  # Decane - flexible chain
                "",  # Empty - should handle gracefully
                "INVALID_SMILES",  # Invalid - should handle gracefully
            ],
            "name": [
                "Ethanol",
                "Benzene",
                "Isoproterenol",
                "Caffeine",
                "Aspirin",
                "Testosterone",
                "L-Alanine",
                "Decane",
                "Empty",
                "Invalid",
            ],
        }
    )

    # Test 1: Basic functionality and feature counts
    print("\n" + "-" * 40)
    print("TEST 1: Basic Functionality")
    print("-" * 40)

    feature_names = get_3d_feature_names()
    print(f"\nTotal feature count: {len(feature_names)}")
    print("  - RDKit 3D shape: 10")
    print(f"  - Mordred 3D: {len(get_mordred_3d_feature_names())}")
    print(f"  - Pharmacophore 3D: {len(get_pharmacophore_3d_feature_names())}")
    print("  - Conformer stats: 5")

    result = compute_descriptors_3d(test_data, n_conformers=5, optimize=True)

    # Test 2: RDKit Shape Descriptors
    print("\n" + "-" * 40)
    print("TEST 2: RDKit 3D Shape Descriptors")
    print("-" * 40)

    shape_cols = ["name", "pmi1", "pmi2", "pmi3", "npr1", "npr2", "asphericity"]
    print(result[shape_cols].head(8).to_string(index=False))

    # Test 3: Shape classification using NPR plot
    print("\n" + "-" * 40)
    print("TEST 3: Shape Classification (NPR1 vs NPR2)")
    print("-" * 40)
    print("""
    NPR Plot interpretation:
    - NPR1~0, NPR2~0: Rod-like
    - NPR1~0.5, NPR2~0.5: Disc-like
    - NPR1~1, NPR2~1: Sphere-like
    """)

    for _, row in result[result["name"].isin(["Benzene", "Decane", "Testosterone", "Caffeine"])].iterrows():
        npr1, npr2 = row["npr1"], row["npr2"]
        if pd.notna(npr1) and pd.notna(npr2):
            if npr1 < 0.3 and npr2 > 0.7:
                shape = "rod-like"
            elif 0.4 < npr1 < 0.6 and 0.4 < npr2 < 0.6:
                shape = "disc-like"
            elif npr1 > 0.7 and npr2 > 0.7:
                shape = "sphere-like"
            else:
                shape = "intermediate"
            print(f"  {row['name']:15} NPR1={npr1:.3f}, NPR2={npr2:.3f} -> {shape}")

    # Test 4: Mordred CPSA Descriptors (sample)
    print("\n" + "-" * 40)
    print("TEST 4: Mordred CPSA Descriptors (sample)")
    print("-" * 40)

    # Show a few CPSA descriptors
    cpsa_cols = [c for c in result.columns if "psa" in c.lower() or "asa" in c.lower()][:5]
    if cpsa_cols:
        print(result[["name"] + cpsa_cols].head(6).to_string(index=False))

    # Test 5: Pharmacophore 3D Descriptors
    print("\n" + "-" * 40)
    print("TEST 5: Pharmacophore 3D Descriptors")
    print("-" * 40)

    pharm_cols = [
        "name",
        "pharm3d_molecular_axis",
        "pharm3d_amphiphilic_moment",
        "pharm3d_nitrogen_span",
        "pharm3d_imhb_potential",
        "pharm3d_elongation",
    ]
    print(result[pharm_cols].head(8).to_string(index=False))

    # Test 6: Conformer Statistics
    print("\n" + "-" * 40)
    print("TEST 6: Conformer Statistics")
    print("-" * 40)

    conf_cols = ["name", "desc3d_conf_count", "conf_energy_min", "conf_energy_range", "conformational_flexibility"]
    print(result[conf_cols].head(8).to_string(index=False))

    # Test 7: Performance benchmark
    print("\n" + "-" * 40)
    print("TEST 7: Performance Benchmark")
    print("-" * 40)

    configs = [
        ("Fast mode (n=1, no opt)", {"n_conformers": 1, "optimize": False}),
        ("Standard mode (n=10, opt)", {"n_conformers": 10, "optimize": True}),
    ]

    bench_data = test_data[test_data["name"].isin(["Ethanol", "Caffeine", "Aspirin", "Testosterone"])].copy()

    for name, params in configs:
        start = time.time()
        _ = compute_descriptors_3d(bench_data, **params)
        elapsed = time.time() - start
        throughput = len(bench_data) / elapsed
        print(f"  {name:30} {elapsed:6.2f}s ({throughput:5.1f} mol/s)")

    # Test 8: Feature name consistency check
    print("\n" + "-" * 40)
    print("TEST 8: Feature Name Consistency")
    print("-" * 40)

    expected_features = set(get_3d_feature_names())
    computed_features = set(result.columns) - set(test_data.columns)

    if expected_features == computed_features:
        print("PASS: All expected features computed correctly")
    else:
        missing = expected_features - computed_features
        extra = computed_features - expected_features
        if missing:
            print(f"WARNING: Missing features: {missing}")
        if extra:
            print(f"WARNING: Extra features: {extra}")

    print("\n" + "=" * 80)
    print("All tests completed!")
    print(f"Total 3D features: {len(get_3d_feature_names())}")
    print("=" * 80)
