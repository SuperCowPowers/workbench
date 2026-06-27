# 3D Molecular Descriptors: Shape, Surface, and Pharmacophore Features
!!! tip inline end "When to reach for 3D"
    On most TDC ADMET endpoints, careful 2D fingerprints + learned graph representations are highly competitive on their own. Use the 3D endpoint when you need shape-aware features for geometry-sensitive endpoints (passive permeability, P-gp / BCRP interactions, conformer-dependent solubility), typically alongside the 2D feature set, not as a replacement for it.

2D molecular descriptors capture a lot about a molecule's chemistry from its connectivity graph alone -- molecular weight, hydrogen bond donors, topological polar surface area, and hundreds of other properties. Some ADMET properties have geometric components that 2D descriptors capture only indirectly through their correlations with the molecular graph: how a molecule fits into a transporter binding site, whether it can fold to mask polar groups for membrane permeation, or how its charge distribution maps onto its surface. The 3D endpoint exposes these directly as engineered features -- whether they help on a given task is an empirical question, not a foregone conclusion.

Workbench's 3D descriptor endpoints compute **74 conformer-based features** from SMILES strings, covering molecular shape, charged partial surface area, pharmacophore spatial distribution, and conformational flexibility. Like all Workbench endpoints, the contract is simple: **send a DataFrame, get a DataFrame back** -- the input DataFrame comes back with 74 descriptor columns appended. The pipeline runs as the **`smiles-to-3d-full-v1`** async endpoint, using an adaptive, Boltzmann-weighted conformer ensemble for high-quality features.

## When 3D Descriptors Help (and When They Don't)

2D descriptors treat molecules as graphs -- atoms are nodes, bonds are edges. There are ADMET-relevant properties whose geometric components 2D captures only indirectly:

- **Membrane permeability** depends on molecular shape and the spatial separation of polar and nonpolar regions (amphiphilic moment)
- **Transporter interactions** (P-gp, BCRP) correlate with molecular elongation, nitrogen spatial distribution, and overall size
- **Protein-ligand binding** depends on 3D shape complementarity, not just functional group counts
- **Intramolecular hydrogen bonds** enable "chameleonic" behavior where molecules mask polar groups in nonpolar environments -- a 3D phenomenon

3D descriptors expose these directly as engineered features. Whether they actually move the needle on a given downstream model is a separate, empirical question -- and the answer is more nuanced than the bullets above suggest.

### When 2D Alone Is Enough

There is a real and well-supported skeptical position in the cheminformatics community: for most ADMET endpoints, well-engineered 2D fingerprints + learned graph representations are competitive with -- or better than -- 2D + 3D combined.

The evidence:

- On the [TDC ADMET leaderboards](https://tdcommons.ai/benchmark/overview/) through 2024-2026, top reproducible models (MapLight, MapLight+GNN, CaliciBoost, NovoExpert-2) use **ECFP/Avalon/ErG + 200 RDKit 2D physchem + GIN embeddings**, with **no explicit 3D features**. The Koleiev et al. 2026 critical assessment of TDC leaderboard reproducibility makes this concrete.
- **PharmaBench** (Niu et al., *Sci. Data* 2024) finds no statistically significant 2D-vs-3D difference on most ADMET endpoints across thousands of compounds.
- **Bahia et al.** (*Mol. Inform.* 2023) report a 2D + 3D advantage over 2D alone -- but the delta is low single-digit AUC / R², not transformative.

So treat the 3D feature stream honestly: a *complementary* set that may give modest, endpoint-dependent gains (most plausibly on passive permeability, P-gp / BCRP recognition, conformer-dependent solubility) on top of a strong 2D + learned-representation baseline. It is not a foregone conclusion that 3D features improve any specific ADMET model -- run an ablation. See the [Limitations & Future Work](#limitations-future-work) section for forward-looking upgrades and a fuller honest accounting.

## The 3D Descriptor Pipeline

Workbench provides the `smiles-to-3d-full-v1` endpoint for 3D descriptors:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;"></th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">smiles-to-3d-full-v1</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Conformers</td><td style="padding: 8px 16px;">50-200 (adaptive by rotatable bonds)</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Aggregation</td><td style="padding: 8px 16px;">Boltzmann-weighted ensemble (GFN2-xTB energies)</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Deployment</td><td style="padding: 8px 16px;">Async SageMaker endpoint (scale-to-zero)</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Use case</td><td style="padding: 8px 16px;">Training pipelines and overnight batch processing (10k-100k compounds)</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Output</td><td style="padding: 8px 16px;">74 features + 11 diagnostic columns</td></tr>
  </tbody>
</table>

The pipeline uses **Boltzmann-weighted ensemble averaging** -- descriptors are computed on every conformer within a 5 kcal/mol energy window of the lowest-energy conformer, then combined using normalized Boltzmann weights:

$$\Large w_i = \frac{e^{-\Delta E_i \,/\, k_BT}}{\displaystyle\sum_j e^{-\Delta E_j \,/\, k_BT}}, \qquad \langle d \rangle = \sum_i w_i \, d_i$$

where $\Delta E_i = E_i - E_{\min}$ is the energy above the minimum conformer, $k_BT$ is the thermal energy at 298 K (0.592 kcal/mol), and $d_i$ is the descriptor value for conformer $i$. This is more reproducible than single-conformer descriptors, which can vary significantly with random seed, especially for flexible molecules. The MARCEL benchmark and Nikonenko et al. have shown that ensemble approaches produce more stable QSAR models.

The energies $E_i$ that drive these weights come from **GFN2-xTB** (a fast semi-empirical quantum method), not the MMFF94s force field used to build the geometries. MMFF94s energy rankings are known to be unreliable for flexible and polar molecules -- the conformers it ranks lowest are frequently not the ones a quantum method favors, which biases the weighted average toward the wrong geometries. Decoupling the two -- MMFF94s for *geometry*, GFN2-xTB for the *energy ranking* that sets the weights -- is the single highest-leverage accuracy lever for the ensemble, since every one of the 74 features is a Boltzmann average over these weights. See [Step 3](#step-3-boltzmann-weighted-descriptor-calculation) for the mechanics and fallback behavior.

### Adaptive Conformer Counts (Boltzmann Mode)

The full endpoint scales conformer count to molecular flexibility:

| Rotatable Bonds | Conformers |
|-----------------|------------|
| < 8 | 50 |
| ≥ 8 | 200 |

The count is capped at 200 because GFN2-xTB scores every conformer (cost ≈ heavy atoms × conformers), while for heavy/flexible molecules only a handful of conformers fall inside the 5 kcal/mol Boltzmann window regardless of how many are generated — so beyond ~200 the extra compute buys little usable ensemble. For molecules whose features are genuinely seed-sensitive, the effective lever is seed-diversity or a wider energy window rather than raw conformer count (see [Limitations](#limitations-future-work)).

## The Computation Pipeline

The 3D descriptor endpoint runs a multi-step pipeline for each molecule:

<figure style="margin: 20px auto; text-align: center;">
<img src="../../images/3d_descriptor_pipeline.svg" alt="3D descriptor pipeline: SMILES to Standardize to Conformers to 74 Descriptors" style="width: 100%; min-height: 300px;">
<figcaption><em>The 3D descriptor pipeline: standardization, tiered conformer generation with MMFF94s geometry optimization, GFN2-xTB energy ranking, and Boltzmann-weighted ensemble descriptors across four categories.</em></figcaption>
</figure>

### Step 1: Standardization
The same [standardization pipeline](molecular_standardization.md) used by the 2D endpoints runs first -- salt extraction, charge neutralization, and tautomer canonicalization. Stereochemistry is preserved through tautomer canonicalization (we override RDKit's default `tautomerRemoveSp3Stereo=True` which would otherwise silently strip `@` markers). This ensures the 3D descriptors are computed on the same canonical, stereo-faithful structure as the 2D descriptors.

Standardize also emits an `undefined_chiral_centers` diagnostic column counting any chiral centers in the input SMILES that lacked a stereo flag. Nonzero values mean the downstream 3D features reflect an arbitrary enantiomer — users should address ambiguous input upstream.

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

Conformer **geometries** are optimized with the **MMFF94s** force field (preferred over MMFF94 for its improved handling of planar nitrogen centers common in drug molecules), using `optimizerForceTol=0.0135` which provides a ~20% speedup with negligible geometry loss. For molecules with unsupported MMFF atom types, the pipeline automatically falls back to **UFF** (Universal Force Field). Note that MMFF94s is used here only to *build* the geometries -- the energies that rank those geometries for Boltzmann weighting come from GFN2-xTB (see Step 3).

RMSD-based pruning (`pruneRmsThresh=0.5`) removes redundant geometries -- rigid molecules like benzene naturally collapse to 1-2 unique conformers, while flexible chains retain more diversity.

### Step 3: Boltzmann-Weighted Descriptor Calculation

All 74 descriptors are computed on the molecule with **explicit hydrogens preserved** throughout — GFN2-xTB energy calculations, Mordred CPSA partial charges, and RDKit's mass-weighted shape descriptors (PMI, radius of gyration) all require explicit Hs for correct results.

**Conformer energy ranking (GFN2-xTB).** The energies that set the Boltzmann weights come from single-point **GFN2-xTB** calculations (via the [`tblite`](https://github.com/tblite/tblite) library) on the MMFF-optimized geometries. xTB only *scores* the conformers — it does not move atoms, so the geometries the descriptors see are still the MMFF94s ones. The molecule's total formal charge is passed through so charged and zwitterionic species are ranked correctly. If `tblite` is unavailable or a molecule fails to converge, the pipeline transparently falls back to MMFF94s/UFF energies; the `desc3d_energy_method` diagnostic column records which model actually produced the weights (`GFN2-xTB`, `MMFF94s`, or `UFF`) so a fallback is never silent. GFN2-xTB is deterministic and adds roughly 0.1–0.5 s per conformer, which is why xTB ranking runs on the async **full** endpoint where the per-molecule time budget accommodates it.

The custom pharmacophore descriptors, however, follow the cheminformatics convention of heavy-atom-only geometry for distance and centroid calculations (molecular axis, nitrogen span, charge/HBA centroids). The one exception is molecular volume, which uses RDKit's grid-based van der Waals volume and does include Hs — this gives a physically meaningful volume even for small molecules where a heavy-atom-only convex hull would be degenerate.

For each conformer within the 5 kcal/mol energy window, shape, surface, and pharmacophore descriptors are computed independently and then combined via Boltzmann-weighted averaging. Conformer ensemble statistics (energy range, flexibility index) are computed over the full generated ensemble, not just the window.

After embedding, the pipeline also verifies that the 3D geometry reproduces the input stereochemistry and reports the result in the `desc3d_stereo_preserved` diagnostic column — a True/False gate that catches the rare case where ETKDGv3 silently drops a stereo specification on strained scaffolds.

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

The NPR1/NPR2 triangle plot is a widely used visualization for molecular shape classification: rod-shaped molecules cluster near (0, 1), disc-shaped near (0.5, 0.5), and spherical near (1, 1). Landrum's RDKit blog has shown that these PMI-derived descriptors are among the most conformer-sensitive, which is precisely why Boltzmann-weighted averaging improves their reproducibility.

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
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Molecular Volume</td><td style="padding: 8px 16px;">Van der Waals volume via RDKit grid (0.2 &#8491; spacing) -- binding site fit, transporter size constraints</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Amphiphilic Moment</td><td style="padding: 8px 16px;">Polar/nonpolar centroid separation (polar = N/O/S/P + halogens; carbons adjacent to N/O/S/P are neutral) -- membrane orientation, transporter recognition</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Charge Centroid Distance</td><td style="padding: 8px 16px;">Distance from center of mass to centroid of charge-site nitrogens (quaternary/aromatic/N-H) -- captures peripheral vs central ionizable groups</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Nitrogen Span</td><td style="padding: 8px 16px;">Max distance between any two nitrogens (no filter) -- multi-point binding, overall N distribution</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">HBA Centroid Distance</td><td style="padding: 8px 16px;">Distance from COM to centroid of pure H-bond acceptors (all O + N with no H and no + charge; nitro groups excluded) -- solubility, permeability</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">IMHB Potential</td><td style="padding: 8px 16px;">Intramolecular H-bond count: D...A distance 2.5-3.5 &#8491; + 4-6 bond separation + D-H...A angle &#8805; 120&deg; -- chameleonic permeability</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">Elongation</td><td style="padding: 8px 16px;">Axis length / volume^(1/3) -- shape anisotropy</td></tr>
  </tbody>
</table>

The **intramolecular hydrogen bond potential** (IMHB) deserves special mention. Molecules that can form intramolecular H-bonds can "mask" their polar groups in nonpolar membrane environments, dramatically increasing permeability despite high polar surface area. This chameleonic behavior is a key design strategy in modern medicinal chemistry and is invisible to 2D descriptors.

### Conformer Ensemble Statistics (4 features)

Statistics computed over the full generated conformer ensemble that capture conformational flexibility:

- **Energy minimum**: The lowest GFN2-xTB energy (or MMFF94s/UFF on fallback) -- a proxy for strain
- **Energy range / standard deviation**: How spread out the conformer energies are
- **Conformational flexibility index**: Normalized energy range -- higher values indicate more conformational freedom

Highly flexible molecules tend to have larger energy ranges and higher flexibility indices. These features correlate with permeability (flexible molecules pay higher entropic penalties for binding) and metabolic stability.

### Diagnostic Columns

In addition to the 74 model features, the endpoint produces 11 `desc3d_*` diagnostic columns that track pipeline status, conformer generation quality, energy model, stereochemistry preservation, and compute time. These are prefixed to distinguish them from model inputs:

| Column | Description |
|--------|-------------|
| `desc3d_status` | `ok`, `skip:parse`, `skip:heavy_atoms`, `skip:rot_bonds`, `skip:rings`, `skip:ring_complexity`, `skip:cost`, `skip:embed`, `skip:empty` |
| `desc3d_conf_count` | Conformers after RMSD pruning |
| `desc3d_confs_requested` | Target conformer count |
| `desc3d_confs_in_window` | Conformers in the Boltzmann energy window |
| `desc3d_embed_failures` | Distance geometry retry count |
| `desc3d_timeout_failures` | Per-conformer RDKit timeout count |
| `desc3d_embed_tier` | Which embedding tier succeeded (1/2/3) |
| `desc3d_force_field` | Geometry optimizer: MMFF94s, UFF, or none |
| `desc3d_energy_method` | Energy model used for Boltzmann weights: GFN2-xTB, MMFF94s, or UFF |
| `desc3d_stereo_preserved` | True if the 3D geometry reproduces the input's assigned stereo (always True for achiral inputs) |
| `desc3d_compute_time_s` | Per-molecule wall clock |

Endpoint output also includes the `undefined_chiral_centers` column emitted by the upstream `standardize()` step — count of chiral centers in the original input SMILES that lacked a stereo flag, so users can see when features reflect an arbitrary enantiomer.

## Production Guardrails

The 3D endpoints are significantly more compute-intensive than 2D. Several safeguards keep them reliable:

### Molecular Complexity Check
Before attempting conformer generation, molecules are screened against size and topology thresholds that catch molecules too large or complex for reliable conformer generation. These are sized for the async endpoint's 60-minute invocation budget in Boltzmann mode, which comfortably accommodates larger drug-like molecules (PROTACs, small peptides, natural products):

| Property | Threshold | Rationale |
|----------|-----------|-----------|
| Heavy atoms | > 150 | Embedding time scales roughly O(n^2) |
| Rotatable bonds | > 50 | Combinatorial explosion of conformer space |
| Ring systems | > 10 | Extreme ring counts indicate cage structures |
| Ring complexity score | > 15 | Backstop for highly constrained polycyclic cages |
| xTB cost (heavy atoms × conformers) | > 14000 | Backstop for molecules too expensive to score with GFN2-xTB |

The **ring complexity score** (rings + bridgehead atoms + spiro atoms) is a permissive backstop -- common drug scaffolds score well under 15. The **xTB cost** backstop catches molecules that pass the size guards but are pathologically expensive for the quantum energy step: GFN2-xTB scores every conformer, so cost scales as `heavy atoms × conformers`, and a large, very flexible molecule (e.g. Irganox 1010 at 85 heavy atoms × 200 conformers = 17000) would otherwise spend many minutes of xTB to keep only a handful of in-window conformers. It only bites molecules with ≥8 rotatable bonds and more than ~70 heavy atoms — the large-and-very-flexible corner — leaving normal drug-likes (including heavy fused-ring natural products at the 50-conformer tier) untouched. Molecules that exceed any threshold get a specific `desc3d_status` (e.g. `skip:heavy_atoms`, `skip:cost`) instead of feature values, so downstream pipelines can detect and route them appropriately. Upstream, `standardize()` independently rejects molecules over 500 atoms as a sanity cap — its 500-atom limit is intentionally larger than the 3D pipeline's 150-heavy-atom limit so the 3D pipeline's own guards are always the binding constraint.

Molecules exceeding any threshold receive NaN features and a specific `desc3d_status` explaining the skip reason. These guards can be disabled for local analysis (`complexity_check=False`).

## Deploying the Endpoint

```bash
python feature_endpoints/smiles_to_3d_full_v1.py
```

The full endpoint deploys as an [async endpoint](../api_classes/async_endpoint.md) with scale-to-zero -- the instance spins down when idle and cold-starts on the next request. This is ideal for overnight batch runs where you don't want to pay for idle compute during the day.

### Using the Endpoint

```python
from workbench.api import Endpoint

# Async deployment, standard Endpoint API (auto-routes through async core)
end_full = Endpoint("smiles-to-3d-full-v1")
df_3d_full = end_full.inference(df)

# Works with InferenceCache for persistent S3-backed caching
from workbench.api.inference_cache import InferenceCache
cached_endpoint = InferenceCache(end_full, cache_key_column="smiles")
df_cached = cached_endpoint.inference(big_df)  # Only computes uncached rows
```

## Limitations & Future Work

The pipeline is conservative by design — production ADMET targets stable, deterministic features over the latest research methods. A few areas worth flagging for downstream users and future iterations:

**3D vs 2D in ADMET reality.** As noted in the introduction, top reproducible TDC ADMET models lean on 2D fingerprints + learned graph representations. The published evidence (PharmaBench *Sci. Data* 2024; Bahia *Mol. Inform.* 2023) is that 3D descriptors give marginal-but-real gains on geometry-sensitive endpoints and roughly neutral effects on most others. The 3D feature stream complements rather than replaces a strong 2D + learned-representation baseline.

**Cross-seed variance on highly flexible molecules.** For heavy/flexible molecules only a handful of conformers land inside the 5 kcal/mol Boltzmann window regardless of how many are generated, so raw conformer count is a weak lever there — the 200-conformer cap reflects that. Different random seeds still produce slightly different Boltzmann averages on highly flexible molecules; for most ADMET endpoints this residual is below downstream model noise, but for tasks that genuinely depend on a single conformer geometry it is not. The more effective levers for that regime — seed-diversity ensembles or a wider energy window — are candidate future upgrades.

**Recently shipped:** **single-point GFN2-xTB re-ranking** of MMFF-optimized conformers before Boltzmann weighting (via the `tblite` library, deterministic). Kong et al. (*ChemPhysChem* 2025) show GFN2-xTB is currently the most suitable energy filter for drug-like conformer ranking, and we measured near-zero rank correlation between MMFF94s and GFN2-xTB orderings on flexible/polar molecules — so this materially shifts the Boltzmann weights and, with them, all 74 ensemble-averaged features. See [Step 3](#step-3-boltzmann-weighted-descriptor-calculation).

**Forward-looking upgrades** (evidence-backed; not yet implemented):

1. **CONFORGE as alternative embedder** for macrocycles and very-flexible scaffolds. Seidel et al. (*JCIM* 2023, CDPKit) — open source, slightly outperforms RDKit on small molecules and matches it on macrocycles where ETKDGv3 sampling plateaus.
2. **Replace Gasteiger partial charges in CPSA** with AM1-BCC or an ML charge model (DASH; Mahmoud et al. 2023). Gasteiger is documented as the least accurate common partial-charge method, and CPSA accounts for 43 of our 52 Mordred 3D features — the highest-leverage upgrade for the existing feature set.

Deliberately *not* on this list: ML conformer generators (ETFlow, GeoMol, Lyrebird) — research-stage with no proven ADMET benefit; MACE-OFF / ANI-2x routine optimization — too heavy for production throughput; tautomer/protomer ensemble enumeration — mature in research, niche in production. We may revisit any of these as the surrounding tooling matures.

## References

**Conformer Ensemble Methods**

- Zhu, Y., Hwang, J., Adams, K., et al. *"Learning Over Molecular Conformer Ensembles: Datasets and Benchmarks."* ICLR 2024. [arXiv: 2310.00115](https://arxiv.org/abs/2310.00115)
- Nikonenko, A., Zankov, D., Baskin, I., et al. *"Multiple Conformer Descriptors for QSAR Modeling."* Mol. Inform. 40, 2060030 (2021). [DOI: 10.1002/minf.202060030](https://doi.org/10.1002/minf.202060030)
- Hamakawa, Y. & Miyao, T. *"Understanding Conformation Importance in Data-Driven Property Prediction Models."* J. Chem. Inf. Model. 65, 3388-3404 (2025). [DOI: 10.1021/acs.jcim.5c00018](https://doi.org/10.1021/acs.jcim.5c00018)
- Adams, K. & Coley, C.W. *"The Impact of Conformer Quality on Learned Representations of Molecular Conformer Ensembles."* arXiv (2025). [arXiv: 2502.13220](https://arxiv.org/abs/2502.13220)

**Conformer Generation**

- Riniker, S. & Landrum, G.A. *"Better Informed Distance Geometry: Using What We Know To Improve Conformation Generation."* J. Chem. Inf. Model. 55, 2562-2574 (2015). [DOI: 10.1021/acs.jcim.5b00654](https://doi.org/10.1021/acs.jcim.5b00654)
- Wang, S., Witek, J., Landrum, G.A. & Riniker, S. *"Improving Conformer Generation for Small Rings and Macrocycles Based on Distance Geometry and Experimental Torsional-Angle Preferences."* J. Chem. Inf. Model. 60, 2044-2058 (2020). [DOI: 10.1021/acs.jcim.0c00025](https://doi.org/10.1021/acs.jcim.0c00025)
- Landrum, G. *"Optimizing conformer generation parameters."* RDKit Blog (2022). [Blog post](https://greglandrum.github.io/rdkit-blog/posts/2022-09-29-optimizing-conformer-generation-parameters.html)
- Landrum, G. *"Variability of PMI Descriptors."* RDKit Blog (2022). [Blog post](https://greglandrum.github.io/rdkit-blog/posts/2022-06-22-variability-of-pmi-descriptors.html)
- Landrum, G. *"Understanding conformer generation failures."* RDKit Blog (2023). [Blog post](https://greglandrum.github.io/rdkit-blog/posts/2023-05-17-understanding-confgen-errors.html)
- Landrum, G. *"Scaling conformer generation."* RDKit Blog (2025). [Blog post](https://greglandrum.github.io/rdkit-blog/posts/2025-08-30-confgen-scaling.html)
- Datamol conformer generation with adaptive tiering. [Documentation](https://docs.datamol.io/stable/tutorials/Conformers.html)
- Seidel, T., Permann, C., Wieder, O., Kohlbacher, S. & Langer, T. *"High-Quality Conformer Generation with CONFORGE: Algorithm and Performance Assessment."* J. Chem. Inf. Model. 63, 5549-5570 (2023). [DOI: 10.1021/acs.jcim.3c00563](https://doi.org/10.1021/acs.jcim.3c00563)

**Force Fields**

- Tosco, P., Stiefl, N. & Landrum, G. *"Bringing the MMFF force field to the RDKit: implementation and validation."* J. Cheminform. 6, 37 (2014). [DOI: 10.1186/s13321-014-0037-3](https://doi.org/10.1186/s13321-014-0037-3)

**Conformer Energy Ranking (GFN2-xTB)**

- Bannwarth, C., Ehlert, S. & Grimme, S. *"GFN2-xTB—An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method."* J. Chem. Theory Comput. 15, 1652-1671 (2019). [DOI: 10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- `tblite` — light-weight tight-binding framework providing the GFN2-xTB Python bindings. [GitHub](https://github.com/tblite/tblite)
- Kong, Z., et al. *"Discriminating High from Low Energy Conformers of Druglike Molecules."* ChemPhysChem (2025). [DOI: 10.1002/cphc.202400992](https://doi.org/10.1002/cphc.202400992)

**Descriptors**

- RDKit 3D Descriptors: [Documentation](https://www.rdkit.org/docs/source/rdkit.Chem.Descriptors3D.html)
- Mordred Community: [GitHub](https://github.com/JacksonBurns/mordred-community)
- Stanton, D.T. & Jurs, P.C. *"Development and Use of Charged Partial Surface Area Structural Descriptors in Computer-Assisted Quantitative Structure-Property Relationship Studies."* Anal. Chem. 62, 2323-2329 (1990). [DOI: 10.1021/ac00220a013](https://doi.org/10.1021/ac00220a013)
- Bleiziffer, P., Schaller, K. & Riniker, S. *"Machine Learning of Partial Charges Derived from High-Quality Quantum-Mechanical Calculations."* J. Chem. Inf. Model. 58, 579-589 (2018). [DOI: 10.1021/acs.jcim.7b00663](https://doi.org/10.1021/acs.jcim.7b00663)
- Lehner, M.T., Katzberger, P., Maeder, N., et al. *"DASH: Dynamic Attention-Based Substructure Hierarchy for Partial Charge Assignment."* J. Chem. Inf. Model. 63, 6014-6028 (2023). [DOI: 10.1021/acs.jcim.3c00800](https://doi.org/10.1021/acs.jcim.3c00800)

**ADMET Benchmarks and 2D vs 3D Evidence**

- Huang, K., et al. *"Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development."* [TDC ADMET Leaderboards](https://tdcommons.ai/benchmark/overview/)
- Koleiev, I., Stratiichuk, R., Shevchuk, N., et al. *"Critical Assessment of ML Models for ADMET Prediction in TDC Leaderboards."* bioRxiv (2026). [DOI: 10.64898/2026.02.26.708193](https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1)
- Niu, Z., et al. *"PharmaBench: Enhancing ADMET Benchmarks with Large Language Models."* Sci. Data 11, 985 (2024). [DOI: 10.1038/s41597-024-03793-0](https://doi.org/10.1038/s41597-024-03793-0)
- Bahia, M.S., et al. *"Comparison Between 2D and 3D Descriptors in QSAR Modeling Based on Bio-Activities."* Mol. Inform. 42, 2200186 (2023). [DOI: 10.1002/minf.202200186](https://doi.org/10.1002/minf.202200186)

**ADMET and Chameleonic Molecules**

- Whitty, A., et al. *"Quantifying the chameleonic properties of macrocycles and other high-molecular-weight drugs."* Drug Discov. Today 21, 712-717 (2016). [DOI: 10.1016/j.drudis.2016.02.005](https://doi.org/10.1016/j.drudis.2016.02.005)

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
