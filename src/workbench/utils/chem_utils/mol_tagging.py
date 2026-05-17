"""
mol_tagging.py — Molecular property tagging for ADMET data curation

Purpose
-------
Annotates a DataFrame with namespaced categorical tags that support filtering
and training-set curation decisions for ADMET modeling. Tags are *categorical
metadata only* — no numeric scores. For numeric features (QED, LogP, TPSA,
descriptors), see mol_descriptors.py and the smiles-to-2d feature endpoint.

Pipeline position
-----------------
Expects input that has already been processed by mol_standardize.standardize:
salts/mixtures resolved, charges neutralized, canonical tautomer chosen, and
(optionally) an ``undefined_chiral_centers`` column populated. This module
does NOT modify structures — it only annotates them.

Tag namespaces (separator: ``:``)
---------------------------------
- ``composition:*``  atom-level content (inorganic, halogenated, heavy_metal, …)
- ``structure:*``    topology and shape (aromatic, chiral, macrocycle, peptide, …)
- ``physchem:*``     physicochemical filter outcomes (ro5_compliant, veber_compliant, …)
                     **These are property-range filters, not drug-likeness verdicts.**
- ``liabilities:*``  published assay-interference / reactive structural alerts
                     (PAINS A/B/C, BRENK, NIH) via RDKit's FilterCatalog
- ``curation:*``     derived ADMET training-set decisions:
                       * ``curation:exclude:*``  → recommend dropping from training
                       * ``curation:caution:*``  → keep, but model-domain edge case

Important caveats
-----------------
- Lipinski's Rule of Five is an absorption heuristic for *oral* drugs, NOT a
  drug-likeness verdict. Many approved drugs (oncology, peptides, macrocycles,
  some antibiotics) violate Ro5. Use ``physchem:ro5_compliant`` as **one**
  signal, not **the** signal. See Oprea (2000).
- Curation tags reflect a default policy for *general* ADMET training. Specific
  endpoints (e.g. permeability vs aqueous solubility) may want different
  policies — use curation tags as a starting point, not the last word.

References
----------
- Lipinski et al. (1997)   Ro5.                       doi:10.1016/S0169-409X(96)00423-1
- Oprea (2000)             Property distributions of drug-related chemical databases.
                                                       doi:10.1023/A:1008130001697
- Veber et al. (2002)      Properties for oral bioavailability.
                                                       doi:10.1021/jm020017n
- Congreve et al. (2003)   Rule of Three for fragments.
                                                       doi:10.1016/S1359-6446(03)02831-9
- Baell & Holloway (2010)  PAINS filters.             doi:10.1021/jm901137j
- Brenk et al. (2008)      BRENK structural alerts.   doi:10.1002/cmdc.200700139
- Lovering et al. (2009)   Escape from flatland (Fsp3).
                                                       doi:10.1021/jm901241e
- Bento et al. (2020)      ChEMBL structure standardization pipeline.
                                                       doi:10.1186/s13321-020-00456-1
"""

import logging
from typing import List, Optional, Set

import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, Mol, rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams

from workbench.utils.chem_utils.toxicity import (
    contains_heavy_metals,
    halogen_toxicity_score,
)

logger = logging.getLogger("workbench")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Elements that routinely appear in approved drugs. Atoms outside this set
# get flagged as ``composition:unusual_element``.
DRUG_ELEMENTS_CORE = {"H", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
DRUG_ELEMENTS_EXTENDED = DRUG_ELEMENTS_CORE | {"B", "Si", "Se"}

# IUPAC: macrocycle = ring of ≥12 atoms.
MACROCYCLE_RING_SIZE = 12

# Heuristic peptide threshold: ≥3 secondary amide bonds suggests a tri-peptide
# or larger. Drugs with 2 amide bonds are common (e.g. lopinavir-class); 3+
# is a strong peptide signal.
PEPTIDE_BOND_THRESHOLD = 3
_PEPTIDE_BOND_SMARTS = Chem.MolFromSmarts("[NX3;H1][CX3](=O)[#6]")

# MW bounds for default ADMET training-set inclusion. Below ~100 Da public
# ADMET assays rarely have meaningful signal; above ~900 Da most public ADMET
# models are out of domain (peptides/PROTACs/macrocycles handled via caution
# tags rather than hard exclude).
ADMET_MW_MIN = 100.0
ADMET_MW_MAX = 900.0

# Lovering "escape from flatland": Fsp3 ≥ 0.5 marks a 3D-rich scaffold.
FSP3_HIGH_THRESHOLD = 0.5


# ---------------------------------------------------------------------------
# FilterCatalog cache (module-level — initialization is ~hundreds of ms)
# ---------------------------------------------------------------------------

_FILTER_CATALOGS: dict[str, FilterCatalog] = {}
_FILTER_CATALOG_MAP = {
    "pains_a": FilterCatalogParams.FilterCatalogs.PAINS_A,
    "pains_b": FilterCatalogParams.FilterCatalogs.PAINS_B,
    "pains_c": FilterCatalogParams.FilterCatalogs.PAINS_C,
    "brenk": FilterCatalogParams.FilterCatalogs.BRENK,
    "nih": FilterCatalogParams.FilterCatalogs.NIH,
}


def _get_filter_catalog(name: str) -> FilterCatalog:
    if name not in _FILTER_CATALOGS:
        params = FilterCatalogParams()
        params.AddCatalog(_FILTER_CATALOG_MAP[name])
        _FILTER_CATALOGS[name] = FilterCatalog(params)
    return _FILTER_CATALOGS[name]


# ---------------------------------------------------------------------------
# Category tag builders
# ---------------------------------------------------------------------------


def _composition_tags(mol: Mol) -> Set[str]:
    """Atom-content tags. Heavy-metal / halogen logic delegated to toxicity.py."""
    tags: Set[str] = set()

    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    has_carbon = "C" in symbols

    if not has_carbon:
        tags.add("composition:inorganic")

    if contains_heavy_metals(mol):
        tags.add("composition:heavy_metal")
        if has_carbon:
            tags.add("composition:organometallic")

    halogen_count, halogen_threshold = halogen_toxicity_score(mol)
    if halogen_count > 0:
        tags.add("composition:halogenated")
    if halogen_count > halogen_threshold:
        tags.add("composition:highly_halogenated")

    if any(s not in DRUG_ELEMENTS_EXTENDED for s in symbols):
        tags.add("composition:unusual_element")

    if any(a.GetIsotope() != 0 for a in mol.GetAtoms()):
        tags.add("composition:isotope_labeled")

    return tags


def _structure_tags(mol: Mol) -> Set[str]:
    """Topology / shape tags."""
    tags: Set[str] = set()

    ring_info = mol.GetRingInfo()
    atom_rings = ring_info.AtomRings()

    if not atom_rings:
        tags.add("structure:acyclic")
    else:
        if any(any(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring) for ring in atom_rings):
            tags.add("structure:aromatic")
        if any(len(ring) >= MACROCYCLE_RING_SIZE for ring in atom_rings):
            tags.add("structure:macrocycle")

    if any(a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED for a in mol.GetAtoms()):
        tags.add("structure:chiral")

    if rdMolDescriptors.CalcNumBridgeheadAtoms(mol) > 0:
        tags.add("structure:bridged")
    if rdMolDescriptors.CalcNumSpiroAtoms(mol) > 0:
        tags.add("structure:spiro")

    if _PEPTIDE_BOND_SMARTS is not None:
        n_amides = len(mol.GetSubstructMatches(_PEPTIDE_BOND_SMARTS))
        if n_amides >= PEPTIDE_BOND_THRESHOLD:
            tags.add("structure:peptide")

    if rdMolDescriptors.CalcFractionCSP3(mol) >= FSP3_HIGH_THRESHOLD:
        tags.add("structure:high_fsp3")

    return tags


def _physchem_tags(mol: Mol) -> Set[str]:
    """
    Physicochemical filter outcomes.

    These are *property-range filters*, not drug-likeness verdicts.
    Ro5 (Lipinski 1997) is an absorption heuristic for oral drugs;
    see Oprea (2000) for why Ro5 ≠ drug-likeness.
    """
    tags: Set[str] = set()

    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    rotbonds = Descriptors.NumRotatableBonds(mol)
    tpsa = Descriptors.TPSA(mol)

    ro5_violations = int(mw > 500) + int(logp > 5) + int(hbd > 5) + int(hba > 10)
    if ro5_violations <= 1:
        tags.add("physchem:ro5_compliant")
    if ro5_violations == 0:
        tags.add("physchem:ro5_strict")
    if mw > 500 or logp > 5:
        tags.add("physchem:beyond_ro5")

    if rotbonds <= 10 and tpsa <= 140:
        tags.add("physchem:veber_compliant")

    if 150 <= mw <= 350 and -3 <= logp <= 3.5:
        tags.add("physchem:lead_like")

    if mw <= 300 and logp <= 3 and hbd <= 3 and hba <= 3 and rotbonds <= 3:
        tags.add("physchem:fragment_like")

    return tags


def _liability_tags(mol: Mol) -> Set[str]:
    """Published assay-interference / reactive structural alerts."""
    tags: Set[str] = set()
    for name in _FILTER_CATALOG_MAP:
        if _get_filter_catalog(name).HasMatch(mol):
            tags.add(f"liabilities:{name}")
    return tags


def _curation_tags(
    mol: Mol,
    underlying: Set[str],
    undefined_chiral_centers: Optional[int],
) -> Set[str]:
    """
    ADMET training-set curation decisions, derived from the other categories.

    Reads from the already-computed ``underlying`` tag set to avoid recomputing
    anything; ``undefined_chiral_centers`` comes from mol_standardize when
    available.
    """
    tags: Set[str] = set()

    # --- hard excludes ---
    if "composition:inorganic" in underlying:
        tags.add("curation:exclude:inorganic")
    if "composition:organometallic" in underlying:
        tags.add("curation:exclude:organometallic")

    # Multi-fragment after standardization means a true mixture (salts should
    # already have been removed upstream); not modelable as a single compound.
    if len(Chem.GetMolFrags(mol)) > 1:
        tags.add("curation:exclude:mixture")

    mw = Descriptors.MolWt(mol)
    if mw < ADMET_MW_MIN:
        tags.add("curation:exclude:mw_too_low")
    if mw > ADMET_MW_MAX:
        tags.add("curation:exclude:mw_too_high")

    # --- cautions (kept, but model-domain edge cases) ---
    if undefined_chiral_centers and undefined_chiral_centers > 0:
        tags.add("curation:caution:stereo_undefined")
    if "composition:isotope_labeled" in underlying:
        tags.add("curation:caution:isotope_labeled")
    if "structure:peptide" in underlying:
        tags.add("curation:caution:peptide")
    if "structure:macrocycle" in underlying:
        tags.add("curation:caution:macrocycle")
    if "composition:heavy_metal" in underlying:
        tags.add("curation:caution:heavy_metal")
    if "composition:highly_halogenated" in underlying:
        tags.add("curation:caution:highly_halogenated")
    if "composition:unusual_element" in underlying:
        tags.add("curation:caution:unusual_element")
    if any(t.startswith("liabilities:pains_") for t in underlying):
        tags.add("curation:caution:pains")

    return tags


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


_CATEGORY_BUILDERS = {
    "composition": _composition_tags,
    "structure": _structure_tags,
    "physchem": _physchem_tags,
    "liabilities": _liability_tags,
}
ALL_CATEGORIES: List[str] = list(_CATEGORY_BUILDERS.keys()) + ["curation"]


def tag_molecules(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    tag_column: str = "tags",
    categories: Optional[List[str]] = None,
    chiral_column: str = "undefined_chiral_centers",
) -> pd.DataFrame:
    """
    Annotate a DataFrame with namespaced molecular tags.

    Expects standardized SMILES (output of mol_standardize.standardize).

    Parameters
    ----------
    df            : input DataFrame
    smiles_column : SMILES column (default 'smiles')
    tag_column    : output tag column (default 'tags')
    categories    : categories to emit. Default: all of ALL_CATEGORIES.
                    Valid values: 'composition', 'structure', 'physchem',
                    'liabilities', 'curation'. If 'curation' is requested,
                    the underlying categories are computed internally for
                    derivation even if not emitted.
    chiral_column : if present in df, integer count of undefined chiral
                    centers from mol_standardize. Used to derive
                    'curation:caution:stereo_undefined'. (default
                    'undefined_chiral_centers')

    Returns
    -------
    DataFrame with ``tag_column`` populated. Each cell is a sorted list of
    namespaced tag strings, or ``['invalid_smiles']`` for unparseable input.
    """
    if categories is None:
        categories = list(ALL_CATEGORIES)

    requested = set(categories)
    unknown = requested - set(ALL_CATEGORIES)
    if unknown:
        raise ValueError(f"Unknown tag categories: {sorted(unknown)}")

    need_curation = "curation" in requested
    compute_set = (requested & set(_CATEGORY_BUILDERS)) | (set(_CATEGORY_BUILDERS) if need_curation else set())
    emit_set = requested

    out = df.copy()
    has_chiral_col = chiral_column in out.columns

    def _tag_row(row) -> List[str]:
        smiles = row[smiles_column]
        if pd.isna(smiles) or smiles == "":
            return ["invalid_smiles"]
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ["invalid_smiles"]

        per_category: dict[str, Set[str]] = {}
        for cat in compute_set:
            per_category[cat] = _CATEGORY_BUILDERS[cat](mol)

        if need_curation:
            underlying = set().union(*per_category.values()) if per_category else set()
            undef = row[chiral_column] if has_chiral_col else None
            per_category["curation"] = _curation_tags(mol, underlying, undef)

        emitted: Set[str] = set()
        for cat in emit_set:
            emitted.update(per_category.get(cat, set()))
        return sorted(emitted)

    out[tag_column] = out.apply(_tag_row, axis=1)

    total = len(out)
    valid = int((out[tag_column].apply(lambda t: t != ["invalid_smiles"])).sum())
    excludes = int(out[tag_column].apply(lambda t: any(x.startswith("curation:exclude:") for x in t)).sum())
    cautions = int(out[tag_column].apply(lambda t: any(x.startswith("curation:caution:") for x in t)).sum())
    logger.info(
        f"Tagged {total} molecules: {valid} valid, " f"{excludes} curation:exclude, {cautions} curation:caution"
    )
    return out


def filter_by_tags(
    df: pd.DataFrame,
    require: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None,
    require_prefix: Optional[List[str]] = None,
    exclude_prefix: Optional[List[str]] = None,
    tag_column: str = "tags",
) -> pd.DataFrame:
    """
    Filter rows by tag membership.

    Parameters
    ----------
    require        : exact tags that must all be present (AND)
    exclude        : exact tags that must not be present
    require_prefix : row must have at least one tag matching any of these
                     prefixes (e.g. ['physchem:ro5_'] requires any ro5_* tag)
    exclude_prefix : row must have no tag matching any of these prefixes
                     (e.g. ['curation:exclude:'] drops everything flagged
                     for ADMET exclusion)

    Examples
    --------
    >>> # Drop everything ADMET curation flags for exclusion, keep cautions:
    >>> clean = filter_by_tags(df, exclude_prefix=["curation:exclude:"])

    >>> # Keep only orally-bioavailable, Ro5+Veber compliant candidates:
    >>> oral = filter_by_tags(df, require=["physchem:ro5_compliant",
    ...                                    "physchem:veber_compliant"])
    """
    tags_series = df[tag_column]
    mask = pd.Series(True, index=df.index)

    if require:
        for t in require:
            mask &= tags_series.apply(lambda tags, _t=t: _t in tags)
    if exclude:
        for t in exclude:
            mask &= tags_series.apply(lambda tags, _t=t: _t not in tags)
    if require_prefix:
        mask &= tags_series.apply(lambda tags: any(t.startswith(p) for t in tags for p in require_prefix))
    if exclude_prefix:
        mask &= tags_series.apply(lambda tags: not any(t.startswith(p) for t in tags for p in exclude_prefix))

    result = df[mask]
    logger.info(f"Filtered {len(df)} → {len(result)} molecules")
    return result


def get_tag_summary(df: pd.DataFrame, tag_column: str = "tags") -> pd.Series:
    """Return a descending count of every tag across the DataFrame."""
    all_tags = [t for row in df[tag_column] for t in row]
    return pd.Series(all_tags).value_counts()


def admet_training_set(
    df: pd.DataFrame,
    drop_cautions: bool = False,
    tag_column: str = "tags",
) -> pd.DataFrame:
    """
    One-shot curation filter for ADMET training data.

    Always drops:
        - invalid SMILES (``invalid_smiles``)
        - anything flagged ``curation:exclude:*``
          (inorganic, organometallic, mixture, mw_too_low, mw_too_high)

    Optionally drops:
        - anything flagged ``curation:caution:*`` (peptide, macrocycle,
          stereo_undefined, isotope_labeled, heavy_metal, highly_halogenated,
          unusual_element, pains) when ``drop_cautions=True``.

    Cautions are kept by default because the right policy is endpoint-specific
    (e.g. peptides matter for permeability, but heavy halogenation matters for
    lipophilicity assays). Use ``drop_cautions=True`` if you want a strict
    "small-molecule drug-like" cut.

    Logs a per-reason breakdown of dropped rows.
    """
    excludes = ["curation:exclude:"]
    if drop_cautions:
        excludes.append("curation:caution:")

    # Per-reason drop counts for the log
    reasons: dict[str, int] = {}
    for tags in df[tag_column]:
        if tags == ["invalid_smiles"]:
            reasons["invalid_smiles"] = reasons.get("invalid_smiles", 0) + 1
            continue
        for t in tags:
            if any(t.startswith(p) for p in excludes):
                reasons[t] = reasons.get(t, 0) + 1

    kept = filter_by_tags(
        df,
        exclude=["invalid_smiles"],
        exclude_prefix=excludes,
        tag_column=tag_column,
    )

    if reasons:
        breakdown = ", ".join(f"{k}={v}" for k, v in sorted(reasons.items()))
        logger.info(f"ADMET training set: {len(df)} → {len(kept)} (dropped: {breakdown})")
    else:
        logger.info(f"ADMET training set: {len(df)} → {len(kept)} (no drops)")
    return kept


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pd.set_option("display.max_colwidth", 200)

    test_data = pd.DataFrame(
        {
            "compound_id": [f"C{i:03d}" for i in range(1, 13)],
            "smiles": [
                "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
                "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
                "C" * 50,  # Long alkane
                "C(Cl)(Cl)(Cl)Cl",  # CCl4
                "[Zn+2].[Cl-].[Cl-]",  # ZnCl2 (inorganic)
                "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Ibuprofen
                "[Pb+2].[O-]C(=O)C",  # Lead acetate
                "N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)N[C@@H](C)C(=O)O",  # Tetra-Ala
                "C1CCCCCCCCCCC1",  # Cyclododecane
                "[2H]C([2H])([2H])C(=O)O",  # d3-acetate
                "",
                "INVALID_SMILES",
            ],
        }
    )

    print("Tagging test data (all categories)...")
    tagged = tag_molecules(test_data)
    for _, row in tagged.iterrows():
        print(f"{row['compound_id']}: {row['tags']}")

    print("\nTag summary:")
    print(get_tag_summary(tagged))

    print("\nfilter_by_tags(exclude_prefix=['curation:exclude:']):")
    kept = filter_by_tags(tagged, exclude_prefix=["curation:exclude:"])
    print(list(kept["compound_id"]))

    print("\nfilter_by_tags(require=['physchem:ro5_compliant','physchem:veber_compliant']):")
    oral = filter_by_tags(tagged, require=["physchem:ro5_compliant", "physchem:veber_compliant"])
    print(list(oral["compound_id"]))

    print("\nadmet_training_set(drop_cautions=False):")
    train = admet_training_set(tagged)
    print(list(train["compound_id"]))

    print("\nadmet_training_set(drop_cautions=True):")
    strict = admet_training_set(tagged, drop_cautions=True)
    print(list(strict["compound_id"]))
