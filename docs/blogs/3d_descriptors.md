# 3D Molecular Descriptors: Shape, Surface, and Pharmacophore Features
!!! tip inline end "Combine 2D + 3D"
    Run both the [2D descriptor endpoint](molecular_standardization.md) and the 3D endpoint, then concatenate the results for a ~390-feature set covering topological, electronic, and geometric properties.

2D molecular descriptors capture a lot about a molecule's chemistry from its connectivity graph alone -- molecular weight, hydrogen bond donors, topological polar surface area, and hundreds of other properties. But some of the most important ADMET properties depend on the molecule's *shape* in three dimensions: how it fits into a transporter binding site, whether it can fold to mask polar groups for membrane permeation, or how its charge distribution maps onto its surface.

Workbench's 3D descriptor endpoint computes **75 conformer-based features** from SMILES strings, covering molecular shape, charged partial surface area, pharmacophore spatial distribution, and conformational flexibility. In this blog we'll walk through the computational pipeline, the descriptor categories, and the production guardrails that make this work reliably as a deployed [feature endpoint](feature_endpoints.md).

## Why 3D Descriptors?

2D descriptors treat molecules as graphs -- atoms are nodes, bonds are edges. This misses geometry-dependent properties that matter for ADMET:

- **Membrane permeability** depends on molecular shape and the spatial separation of polar and nonpolar regions (amphiphilic moment)
- **Transporter interactions** (P-gp, BCRP) correlate with molecular elongation, nitrogen spatial distribution, and overall size
- **Protein-ligand binding** depends on 3D shape complementarity, not just functional group counts
- **Intramolecular hydrogen bonds** enable "chameleonic" behavior where molecules mask polar groups in nonpolar environments -- a purely 3D phenomenon

These properties can't be captured from the molecular graph. You need 3D coordinates.

## The Computation Pipeline

The 3D descriptor endpoint runs a multi-step pipeline for each molecule:

<figure style="margin: 20px auto; text-align: center;">
<img src="../../images/3d_descriptor_pipeline.svg" alt="3D descriptor pipeline: SMILES to Standardize to Conformers to 75 Descriptors" style="width: 100%; min-height: 300px;">
<figcaption><em>The 3D descriptor pipeline: standardization, tiered conformer generation with MMFF94s optimization, and four categories of 3D descriptors computed on the lowest-energy conformer.</em></figcaption>
</figure>

### Step 1: Standardization
The same [standardization pipeline](molecular_standardization.md) used by the 2D endpoints runs first -- salt extraction, charge neutralization, and tautomer canonicalization. This ensures the 3D descriptors are computed on the same canonical structure as the 2D descriptors.

### Step 2: Conformer Generation

Generating realistic 3D coordinates from a SMILES string is the most computationally intensive step. Workbench uses RDKit's **ETKDGv3** (Experimental Torsion Knowledge Distance Geometry v3), which biases conformer sampling toward torsion angles observed in crystal structures -- appropriate for the condensed-phase geometries relevant to ADMET.

The algorithm uses a three-tier embedding strategy to maximize success rates across diverse chemical structures:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Tier</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Strategy</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">When It's Needed</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">1. Standard ETKDGv3</td><td style="padding: 8px 16px;">Experimental torsion preferences + small ring handling</td><td style="padding: 8px 16px;">Works for ~95% of drug-like molecules</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2. Random Coordinates</td><td style="padding: 8px 16px;">Random initial positions instead of distance matrix eigenvalues</td><td style="padding: 8px 16px;">Molecules where distance bounds are hard to satisfy</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">3. Relaxed Constraints</td><td style="padding: 8px 16px;">Random coordinates + relaxed flat-ring enforcement</td><td style="padding: 8px 16px;">Strained bridged polycyclics, unusual ring topologies</td></tr>
  </tbody>
</table>

Each tier generates **10 conformers** with RMSD-based pruning (`pruneRmsThresh=0.5`) to ensure conformational diversity while removing redundant geometries. All conformers are optimized with the **MMFF94s** force field (preferred over MMFF94 for its improved handling of planar nitrogen centers common in drug molecules). For molecules with unsupported MMFF atom types, the pipeline automatically falls back to **UFF** (Universal Force Field).

The lowest-energy conformer from the ensemble is selected for descriptor calculation. While Boltzmann-weighted ensemble averages are theoretically more rigorous, the energy errors inherent in force-field methods make the weighting unreliable in practice -- the lowest-energy conformer provides a robust single-geometry approximation.

### Step 3: Descriptor Calculation

All 75 descriptors are computed on the molecule with **explicit hydrogens preserved** throughout. This is important -- MMFF94s energy calculations, Mordred CPSA partial charges, and mass-weighted shape descriptors all require explicit Hs for correct results.

## Descriptor Categories

### RDKit 3D Shape Descriptors (10 features)

These capture the overall molecular shape using the inertia tensor:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Descriptor</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">What It Captures</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">PMI1, PMI2, PMI3</td><td style="padding: 8px 16px;">Principal moments of inertia -- raw shape information</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">NPR1, NPR2</td><td style="padding: 8px 16px;">Normalized PMI ratios -- classify molecules as rod-like, disc-like, or spherical on the PMI triangle plot</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Asphericity</td><td style="padding: 8px 16px;">How far from spherical (0 = sphere, higher = elongated)</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Eccentricity</td><td style="padding: 8px 16px;">Shape elongation (0 = sphere, 1 = linear)</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Inertial Shape Factor</td><td style="padding: 8px 16px;">Ratio of smallest to largest PMI -- flat vs compact</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Radius of Gyration</td><td style="padding: 8px 16px;">Overall molecular size (mass-weighted spread from center)</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Spherocity Index</td><td style="padding: 8px 16px;">How spherical the molecule is (1 = perfect sphere)</td></tr>
  </tbody>
</table>

The NPR1/NPR2 triangle plot is a widely used visualization for molecular shape classification: rod-shaped molecules cluster near (0, 1), disc-shaped near (0.5, 0.5), and spherical near (1, 1). This is particularly useful for understanding transporter substrate preferences -- P-gp tends to prefer elongated, amphipathic molecules.

### Mordred 3D Descriptors (52 features)

Mordred's 3D modules compute surface-area-based descriptors that capture how charge, polarity, and hydrophobicity distribute across the molecular surface:

- **CPSA (43 descriptors)**: Charged Partial Surface Area -- the 3D extension of topological polar surface area. Maps partial charges onto the solvent-accessible surface to capture electrostatic features that govern solvation, permeability, and protein binding.
- **Geometrical Index (4)**: Petitjean shape indices measuring molecular topology in 3D space.
- **Gravitational Index (4)**: Mass-weighted distance descriptors.
- **PBF (1)**: Plane of Best Fit -- measures molecular planarity, relevant for membrane intercalation and crystal packing.

### Pharmacophore 3D Descriptors (8 features)

Custom descriptors capturing the spatial distribution of pharmacophoric features:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Descriptor</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">ADMET Relevance</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Molecular Axis Length</td><td style="padding: 8px 16px;">Maximum heavy-atom distance -- P-gp substrates are typically 25-30 &#8491; long</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Molecular Volume</td><td style="padding: 8px 16px;">Convex hull volume -- binding site fit, transporter size constraints</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Amphiphilic Moment</td><td style="padding: 8px 16px;">Polar/nonpolar centroid separation -- membrane orientation, transporter recognition</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Charge Centroid Distance</td><td style="padding: 8px 16px;">Distance from center of mass to basic nitrogen centroid -- binding orientation</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Nitrogen Span</td><td style="padding: 8px 16px;">Max distance between any two nitrogens -- multi-point binding</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">HBA Centroid Distance</td><td style="padding: 8px 16px;">H-bond acceptor spatial distribution -- solubility, permeability</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">IMHB Potential</td><td style="padding: 8px 16px;">Intramolecular H-bond potential -- chameleonic permeability (polar group masking)</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Elongation</td><td style="padding: 8px 16px;">Axis length / volume^(1/3) -- shape anisotropy</td></tr>
  </tbody>
</table>

The **intramolecular hydrogen bond potential** (IMHB) deserves special mention. Molecules that can form intramolecular H-bonds can "mask" their polar groups in nonpolar membrane environments, dramatically increasing permeability despite high polar surface area. This chameleonic behavior is a key design strategy in modern medicinal chemistry and is invisible to 2D descriptors.

### Conformer Ensemble Statistics (5 features)

Rather than discarding the conformer ensemble after selecting the lowest-energy geometry, we extract statistics that capture conformational flexibility:

- **Energy minimum**: The lowest force field energy -- a proxy for strain
- **Energy range / standard deviation**: How spread out the conformer energies are
- **Conformational flexibility index**: Normalized energy range -- higher values indicate more conformational freedom
- **Conformer count**: How many unique conformers were generated (after RMSD pruning)

Highly flexible molecules tend to have larger energy ranges, more conformers, and higher flexibility indices. These features correlate with permeability (flexible molecules pay higher entropic penalties for binding) and metabolic stability.

## Production Guardrails

The 3D endpoint is significantly more compute-intensive than 2D (~1-2 molecules/second vs. near-instant for 2D). Several safeguards keep it reliable as a deployed service:

### Molecular Complexity Check
Before attempting conformer generation, molecules are screened against complexity thresholds:

| Property | Threshold | Rationale |
|----------|-----------|-----------|
| Heavy atoms | > 100 | Embedding time scales roughly O(n^2) |
| Rotatable bonds | > 30 | Combinatorial explosion of conformer space |
| Ring systems | > 10 | Complex ring topologies cause embedding failures |

Molecules exceeding any threshold receive NaN values for all 75 features -- the same behavior as a failed conformer generation. These thresholds can be disabled for local analysis (`complexity_check=False`).

### Per-Molecule Timeout
Conformer generation for certain bridged polycyclic systems can hang for minutes even on small molecules. A per-molecule timeout (default 10 seconds) ensures one difficult molecule doesn't consume the entire endpoint response budget. The timeout can be disabled for local debugging (`timeout=None`).

### Batch Size Management
The endpoint uses a smaller inference batch size (10 molecules) compared to the 2D endpoint, keeping total batch processing time safely under the serverless timeout limit.

### Serverless Configuration
The 3D endpoint runs with the maximum serverless memory tier (6144 MB), which also provides the most vCPUs -- important since conformer generation and force field optimization are CPU-bound.

## Deploying the Endpoint

The 3D descriptor endpoint is deployed like any other Workbench feature endpoint:

```python
python feature_endpoints/rdkit_3d_v1.py
```

This creates a `smiles-to-3d-descriptors-v1` endpoint that accepts DataFrames with a SMILES column and returns the same DataFrame with 75 additional descriptor columns. See the [Feature Endpoints](feature_endpoints.md) blog for the full architecture discussion.

### Using the Endpoint

```python
from workbench.api import Endpoint

# Compute 3D descriptors
end_3d = Endpoint("smiles-to-3d-descriptors-v1")
df_3d = end_3d.inference(df)

# Combine with 2D descriptors for a full feature set
end_2d = Endpoint("smiles-to-taut-md-stereo-v1")
df_2d = end_2d.inference(df)
```

## References

- **ETKDGv3**: Riniker, S. & Landrum, G.A. *"Better Informed Distance Geometry."* J. Chem. Inf. Model. 55, 2562-2574 (2015). [DOI: 10.1021/acs.jcim.5b00654](https://doi.org/10.1021/acs.jcim.5b00654)
- **MMFF94s Force Field**: Tosco, P., Stiefl, N. & Landrum, G. *"Bringing the MMFF force field to the RDKit."* J. Cheminform. 6, 37 (2014). [DOI: 10.1186/s13321-014-0037-3](https://doi.org/10.1186/s13321-014-0037-3)
- **RDKit 3D Descriptors**: [https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html)
- **Mordred Community**: [https://github.com/JacksonBurns/mordred-community](https://github.com/JacksonBurns/mordred-community)
- **CPSA Descriptors**: Stanton, D.T. & Jurs, P.C. *"Development and Use of Charged Partial Surface Area Structural Descriptors."* Anal. Chem. 62, 2323-2329 (1990). [DOI: 10.1021/ac00220a013](https://doi.org/10.1021/ac00220a013)
- **Chameleonic Molecules**: Whitty, A., et al. *"Quantifying the chameleonic properties of macrocycles and other high-molecular-weight drugs."* Drug Discov. Today 21, 712-717 (2016). [DOI: 10.1016/j.drudis.2016.02.005](https://doi.org/10.1016/j.drudis.2016.02.005)
- **Conformer Generation Best Practices**: [RDKit Blog - Understanding Conformer Generation Errors](https://greglandrum.github.io/rdkit-blog/posts/2023-05-17-understanding-confgen-errors.html)
- **PMI Descriptor Variability**: [RDKit Blog - Variability of PMI Descriptors](https://greglandrum.github.io/rdkit-blog/posts/2022-06-22-variability-of-pmi-descriptors.html)

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
