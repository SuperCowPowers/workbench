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
from typing import Optional, List, Dict

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors3D
from mordred import Calculator as MordredCalculator
from mordred import CPSA, GeometricalIndex, GravitationalIndex, PBF
from scipy.spatial.distance import pdist

logger = logging.getLogger("workbench")


# =============================================================================
# Conformer Generation
# =============================================================================


def generate_conformers(
    mol: Chem.Mol,
    n_conformers: int = 10,
    random_seed: int = 42,
    optimize: bool = True,
) -> Optional[Chem.Mol]:
    """
    Generate 3D conformers using ETKDGv3.

    Args:
        mol: RDKit molecule object (will be modified in place)
        n_conformers: Number of conformers to generate
        random_seed: Random seed for reproducibility
        optimize: Whether to run MMFF optimization

    Returns:
        Molecule with conformers, or None if generation failed
    """
    if mol is None:
        return None

    try:
        # Add hydrogens (required for proper 3D geometry)
        mol = Chem.AddHs(mol)

        # Configure ETKDGv3 parameters
        # ETKDGv3 handles macrocycles by default (useMacrocycleTorsions=True)
        params = AllChem.ETKDGv3()
        params.randomSeed = random_seed
        params.useSmallRingTorsions = True  # Also handle small rings (3-6 membered)
        params.numThreads = 1  # Single thread to avoid issues in serverless

        # Suppress noisy RDKit warnings (UFFTYPER, etc.) during embedding
        rdkit_logger = RDLogger.logger()
        rdkit_logger.setLevel(RDLogger.ERROR)

        # Generate conformers
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=params)
        except RuntimeError as e:
            logger.debug(f"ETKDGv3 embedding raised {e}, trying fallback")
            conf_ids = []

        if len(conf_ids) == 0:
            # Fallback: random coordinates for difficult molecules
            fallback_params = AllChem.ETKDGv3()
            fallback_params.randomSeed = random_seed
            fallback_params.useSmallRingTorsions = True
            fallback_params.useRandomCoords = True
            fallback_params.numThreads = 1
            try:
                conf_ids = AllChem.EmbedMultipleConfs(mol, numConfs=n_conformers, params=fallback_params)
            except RuntimeError as e:
                logger.warning(f"Fallback embedding raised {e}")
                conf_ids = []

        if len(conf_ids) == 0:
            rdkit_logger.setLevel(RDLogger.WARNING)
            logger.warning("Failed to generate conformers for molecule")
            return None

        # Optimize conformers with MMFF94 if requested
        if optimize:
            try:
                AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=100, numThreads=1)
            except Exception as e:
                logger.debug(f"MMFF optimization failed: {e}")

        # Restore RDKit logging
        rdkit_logger.setLevel(RDLogger.WARNING)
        # Continue without optimization

        # Remove explicit Hs (3D coords on heavy atoms are preserved)
        mol = Chem.RemoveHs(mol)

        return mol

    except Exception as e:
        logger.warning(f"Conformer generation failed: {e}")
        return None


def get_conformer_energies(mol: Chem.Mol) -> List[float]:
    """
    Calculate MMFF94 energies for all conformers.

    Args:
        mol: RDKit molecule with conformers

    Returns:
        List of energies (kcal/mol), NaN for failed calculations
    """
    if mol is None or mol.GetNumConformers() == 0:
        return []

    energies = []
    mmff_props = AllChem.MMFFGetMoleculeProperties(mol)

    if mmff_props is None:
        return [np.nan] * mol.GetNumConformers()

    for conf_id in range(mol.GetNumConformers()):
        try:
            ff = AllChem.MMFFGetMoleculeForceField(mol, mmff_props, confId=conf_id)
            if ff is not None:
                energies.append(ff.CalcEnergy())
            else:
                energies.append(np.nan)
        except Exception:
            energies.append(np.nan)

    return energies


def get_lowest_energy_conformer_id(mol: Chem.Mol) -> int:
    """Get the conformer ID with the lowest MMFF energy."""
    energies = get_conformer_energies(mol)
    if not energies or all(np.isnan(e) for e in energies):
        return 0  # Default to first conformer

    valid_energies = [(i, e) for i, e in enumerate(energies) if not np.isnan(e)]
    if not valid_energies:
        return 0

    return min(valid_energies, key=lambda x: x[1])[0]


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


def compute_mordred_3d_descriptors(mol: Chem.Mol) -> Dict[str, float]:
    """
    Compute Mordred 3D descriptors.

    Args:
        mol: RDKit molecule with conformer

    Returns:
        Dictionary of descriptor name -> value
    """
    calc = get_mordred_3d_calculator()

    if mol is None or mol.GetNumConformers() == 0:
        # Return NaN for all descriptors
        return {str(desc): np.nan for desc in calc.descriptors}

    try:
        # Mordred CPSA needs explicit Hs for partial charge calculations
        mol_with_hs = Chem.AddHs(mol, addCoords=True)
        result_df = calc.pandas([mol_with_hs], nproc=1, quiet=True)

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


def _get_atom_positions_and_masses(mol: Chem.Mol, conf_id: int = 0) -> tuple:
    """
    Get atom positions and masses as numpy arrays.

    Note: Assumes explicit Hs have already been removed (via Chem.RemoveHs)
    during conformer generation, so all atoms are heavy atoms.

    Returns:
        Tuple of (positions array [N, 3], masses array [N]), or (None, None) if failed
    """
    if mol is None or mol.GetNumConformers() == 0:
        return None, None

    conf = mol.GetConformer(conf_id)
    positions = []
    masses = []

    for atom in mol.GetAtoms():
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
    Calculate molecular volume using convex hull of heavy atoms.

    Relevant for:
    - Binding site fit
    - Transporter size constraints
    - Solubility prediction
    """
    positions, _ = _get_atom_positions_and_masses(mol, conf_id)
    if positions is None or len(positions) < 4:
        return np.nan

    try:
        from scipy.spatial import ConvexHull

        hull = ConvexHull(positions)
        return float(hull.volume)
    except Exception:
        return np.nan


def compute_amphiphilic_moment(mol: Chem.Mol, conf_id: int = 0) -> float:
    """
    Calculate distance between centroids of polar and nonpolar atom regions.

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
    Calculate distance from molecular center of mass to centroid of basic nitrogens.

    Captures whether ionizable centers are peripheral or central, which affects:
    - Transporter recognition
    - Binding orientation
    - Membrane interaction
    """
    positions, masses = _get_atom_positions_and_masses(mol, conf_id)
    if positions is None:
        return np.nan

    conf = mol.GetConformer(conf_id)
    com = np.average(positions, axis=0, weights=masses)

    # Find basic nitrogens (protonatable)
    n_positions = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "N":
            # Include aromatic N or N with H (potential protonation sites)
            if atom.GetTotalNumHs() > 0 or atom.GetIsAromatic():
                pos = conf.GetAtomPosition(atom.GetIdx())
                n_positions.append([pos.x, pos.y, pos.z])

    if len(n_positions) == 0:
        return 0.0

    n_centroid = np.mean(n_positions, axis=0)
    return float(np.linalg.norm(com - n_centroid))


def compute_nitrogen_span(mol: Chem.Mol, conf_id: int = 0) -> float:
    """
    Calculate maximum distance between any two nitrogen atoms.

    Captures spatial distribution of basic centers, relevant for:
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
    Calculate distance from molecular center of mass to centroid of H-bond acceptors.

    Captures spatial distribution of H-bond acceptors, relevant for:
    - Binding interactions
    - Solubility
    - Permeability
    """
    positions, masses = _get_atom_positions_and_masses(mol, conf_id)
    if positions is None:
        return np.nan

    conf = mol.GetConformer(conf_id)
    com = np.average(positions, axis=0, weights=masses)

    # Find H-bond acceptors (O and N with lone pairs)
    hba_positions = []
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol == "O":
            pos = conf.GetAtomPosition(atom.GetIdx())
            hba_positions.append([pos.x, pos.y, pos.z])
        elif symbol == "N" and atom.GetTotalNumHs() == 0:
            pos = conf.GetAtomPosition(atom.GetIdx())
            hba_positions.append([pos.x, pos.y, pos.z])

    if len(hba_positions) == 0:
        return 0.0

    hba_centroid = np.mean(hba_positions, axis=0)
    return float(np.linalg.norm(com - hba_centroid))


def compute_intramolecular_hbond_potential(mol: Chem.Mol, conf_id: int = 0) -> int:
    """
    Estimate potential for intramolecular hydrogen bonds.

    Counts donor-acceptor pairs within favorable distance (2.5-3.5 Angstroms) and
    topological separation (4-6 bonds apart).

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

    # HBA atoms (O, N with lone pairs)
    hba_smarts = Chem.MolFromSmarts("[#7X2,#7X3,#8X1,#8X2]")
    hba_matches = mol.GetSubstructMatches(hba_smarts) if hba_smarts else []
    hba_indices = [m[0] for m in hba_matches]

    if not hbd_indices or not hba_indices:
        return 0

    imhb_count = 0

    for d_idx in hbd_indices:
        d_pos = conf.GetAtomPosition(d_idx)
        for a_idx in hba_indices:
            # Skip same atom or directly bonded
            if d_idx == a_idx:
                continue

            bond = mol.GetBondBetweenAtoms(d_idx, a_idx)
            if bond is not None:
                continue

            a_pos = conf.GetAtomPosition(a_idx)
            dist = d_pos.Distance(a_pos)

            # Typical IMHB distance range
            if 2.5 <= dist <= 3.5:
                # Check topological separation (4-6 bonds favors IMHB geometry)
                path = Chem.GetShortestPath(mol, d_idx, a_idx)
                if path and 4 <= len(path) <= 7:
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
    return {
        "pharm3d_molecular_axis": compute_molecular_axis_length(mol, conf_id),
        "pharm3d_molecular_volume": compute_molecular_volume_3d(mol, conf_id),
        "pharm3d_amphiphilic_moment": compute_amphiphilic_moment(mol, conf_id),
        "pharm3d_charge_centroid_dist": compute_charge_centroid_distance(mol, conf_id),
        "pharm3d_nitrogen_span": compute_nitrogen_span(mol, conf_id),
        "pharm3d_hba_centroid_dist": compute_hba_centroid_distance(mol, conf_id),
        "pharm3d_imhb_potential": compute_intramolecular_hbond_potential(mol, conf_id),
        # Derived: ratio of axis to volume^(1/3) - measures elongation
        "pharm3d_elongation": _compute_elongation(mol, conf_id),
    }


def _compute_elongation(mol: Chem.Mol, conf_id: int = 0) -> float:
    """Compute elongation as axis_length / volume^(1/3)."""
    axis = compute_molecular_axis_length(mol, conf_id)
    volume = compute_molecular_volume_3d(mol, conf_id)

    if np.isnan(axis) or np.isnan(volume) or volume <= 0:
        return np.nan

    return axis / (volume ** (1 / 3))


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
            "conf_count": 0,
            "conformational_flexibility": np.nan,
        }

    energies = get_conformer_energies(mol)
    valid_energies = [e for e in energies if not np.isnan(e)]

    if not valid_energies:
        return {
            "conf_energy_min": np.nan,
            "conf_energy_range": np.nan,
            "conf_energy_std": np.nan,
            "conf_count": mol.GetNumConformers(),
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
        "conf_count": mol.GetNumConformers(),
        "conformational_flexibility": flexibility,
    }


# =============================================================================
# Main Descriptor Computation Function
# =============================================================================


def get_3d_feature_names() -> List[str]:
    """
    Return list of all 3D feature names computed by this module.

    Returns:
        List of feature column names (75 total):
        - 10 RDKit 3D shape
        - 52 Mordred 3D (CPSA, Geometrical, Gravitational, PBF)
        - 8 Pharmacophore 3D
        - 5 Conformer statistics
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
        "conf_count",
        "conformational_flexibility",
    ]

    return rdkit_names + mordred_names + pharmacophore_names + conformer_names


def compute_descriptors_3d(
    df: pd.DataFrame,
    n_conformers: int = 10,
    optimize: bool = True,
    random_seed: int = 42,
    use_lowest_energy: bool = True,
) -> pd.DataFrame:
    """
    Compute 3D molecular descriptors for ADMET modeling.

    Args:
        df: Input DataFrame with SMILES column
        n_conformers: Number of conformers to generate (default 10)
        optimize: Whether to run MMFF optimization (default True)
        random_seed: Random seed for conformer generation (default 42)
        use_lowest_energy: Use lowest energy conformer for descriptors (default True)
                          If False, uses first conformer

    Returns:
        DataFrame with 75 additional 3D descriptor columns:
        - 10 RDKit 3D shape descriptors
        - 52 Mordred 3D descriptors (CPSA, Geometrical, Gravitational, PBF)
        - 8 Pharmacophore 3D descriptors
        - 5 Conformer ensemble statistics

    Example:
        df = compute_descriptors_3d(df)  # Standard usage
        df = compute_descriptors_3d(df, n_conformers=1, optimize=False)  # Fast mode
    """
    # Find SMILES column (case-insensitive)
    smiles_column = next((col for col in df.columns if col.lower() == "smiles"), None)
    if smiles_column is None:
        raise ValueError("Input DataFrame must have a 'smiles' column")

    result = df.copy()
    n_molecules = len(df)

    logger.info(f"Computing 3D descriptors for {n_molecules} molecules...")
    logger.info(f"Parameters: n_conformers={n_conformers}, optimize={optimize}")

    # Initialize feature columns
    feature_names = get_3d_feature_names()
    for col in feature_names:
        result[col] = np.nan

    # Process each molecule
    start_time = time.time()

    for idx, row in result.iterrows():
        smiles = row[smiles_column]

        if pd.isna(smiles) or smiles == "":
            continue

        try:
            # Create molecule
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue

            # Generate conformers
            mol = generate_conformers(
                mol,
                n_conformers=n_conformers,
                random_seed=random_seed,
                optimize=optimize,
            )

            if mol is None or mol.GetNumConformers() == 0:
                continue

            # Select conformer for descriptor calculation
            if use_lowest_energy and mol.GetNumConformers() > 1:
                conf_id = get_lowest_energy_conformer_id(mol)
            else:
                conf_id = 0

            # Compute RDKit 3D descriptors
            rdkit_3d = compute_rdkit_3d_descriptors(mol, conf_id)
            for name, value in rdkit_3d.items():
                result.at[idx, name] = value

            # Compute Mordred 3D descriptors
            mordred_3d = compute_mordred_3d_descriptors(mol)
            for name, value in mordred_3d.items():
                if name in result.columns:
                    result.at[idx, name] = value

            # Compute Pharmacophore 3D descriptors
            pharm_3d = compute_pharmacophore_3d_descriptors(mol, conf_id)
            for name, value in pharm_3d.items():
                result.at[idx, name] = value

            # Compute conformer ensemble statistics
            conf_stats = compute_conformer_statistics(mol)
            for name, value in conf_stats.items():
                result.at[idx, name] = value

        except Exception as e:
            logger.debug(f"3D descriptor calculation failed for index {idx}: {e}")
            continue

    elapsed = time.time() - start_time
    valid_count = result[feature_names[0]].notna().sum()
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

    conf_cols = ["name", "conf_count", "conf_energy_min", "conf_energy_range", "conformational_flexibility"]
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
