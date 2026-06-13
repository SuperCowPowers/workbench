# Do 3D Descriptors Help on PXR Induction? A Single-Target Case Study

!!! tip inline end "The one-line takeaway"
    **On this one target** (OpenADMET PXR induction) and **this one held-out analog series**, the 3D conformer descriptors we tested earned **high SHAP importance** inside a combined model yet **did not generalize** out-of-distribution — adding them to our models hurt held-out accuracy, and a 3D-only model scored *negative* R². High attribution is not the same as transferable signal. Read this as a single, carefully-measured data point — not a verdict on 3D descriptors in general.

There is a long-running, well-supported skeptical position in cheminformatics: for many ADMET endpoints, well-engineered 2D descriptors plus a learned graph representation are competitive with — or better than — what you get by adding 3D conformer features. Our [3D descriptor deep-dive](3d_descriptors.md) lays out that literature and the mechanics of the 3D pipeline (including the cases where 3D *does* help). This post is a narrow empirical companion: one clean, reproducible ablation on a single **blind challenge** target where the answers were revealed after the fact. We are deliberately not generalizing beyond it.

We use the [OpenADMET PXR Induction Blind Challenge](https://openadmet.org/blindchallenges/) — predicting human PXR induction (pEC50). It is a good stress test for two reasons. First, the held-out set (**Analog Set 1**, 253 compounds) is a *new chemotype*, not a random split of the training pool — so it measures genuine out-of-distribution transfer, which is where descriptor choices actually get exposed. Second, PXR is a promiscuous, lipophilicity-driven xenobiotic sensor, so we have a strong mechanistic prior about which features *should* matter.

## The Setup

| | |
|---|---|
| **Training set** | 4,139 compounds, pEC50 1.61–7.55 (mean 4.32, σ 1.12) |
| **Held-out set** | Analog Set 1 — 253 compounds, revealed pEC50 (mean 4.66, σ 1.03) |
| **2D features** | 313 RDKit + Mordred 2D descriptors (`smiles-to-2d-v1`) |
| **3D features** | 74 Boltzmann-ensemble conformer descriptors (`smiles-to-3d-full-v1`) |
| **Models** | XGBoost UQ and PyTorch Tabular UQ on each feature block, plus a SMILES-only Chemprop D-MPNN |

Every model reports two numbers that matter:

- **Cross-fold R²** — 5-fold CV on the training pool. This is what you see *before* the blind answers come in.
- **Held-out R²** — on Analog Set 1, the new analog series. This is the honest generalization number.

The whole point of the exercise is the *gap* between those two columns.

## The Result

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Feature Set</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Model</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Cross-fold R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out Spearman</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D</td><td style="padding: 8px 16px;">XGBoost</td><td style="padding: 8px 16px;">0.566</td><td style="padding: 8px 16px;">0.350</td><td style="padding: 8px 16px;">0.67</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">3D</td><td style="padding: 8px 16px;">XGBoost</td><td style="padding: 8px 16px;">0.519</td><td style="padding: 8px 16px; color: #c0392b; font-weight: bold;">&minus;0.101</td><td style="padding: 8px 16px;">&mdash;</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D + 3D</td><td style="padding: 8px 16px;">XGBoost</td><td style="padding: 8px 16px;">0.587</td><td style="padding: 8px 16px;">0.381</td><td style="padding: 8px 16px;">0.70</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D</td><td style="padding: 8px 16px;">PyTorch (313)</td><td style="padding: 8px 16px;">0.531</td><td style="padding: 8px 16px;">0.437</td><td style="padding: 8px 16px;">0.73</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">3D</td><td style="padding: 8px 16px;">PyTorch (74)</td><td style="padding: 8px 16px;">0.472</td><td style="padding: 8px 16px; color: #c0392b; font-weight: bold;">&minus;0.118</td><td style="padding: 8px 16px;">&mdash;</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D + 3D</td><td style="padding: 8px 16px;">PyTorch (387)</td><td style="padding: 8px 16px;">0.549</td><td style="padding: 8px 16px; font-weight: bold;">0.458</td><td style="padding: 8px 16px;">0.73</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">3D</td><td style="padding: 8px 16px;">PyTorch (100-SHAP)</td><td style="padding: 8px 16px;">0.462</td><td style="padding: 8px 16px; color: #c0392b; font-weight: bold;">&minus;0.224</td><td style="padding: 8px 16px;">0.08</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">SMILES</td><td style="padding: 8px 16px;"><b>Chemprop D-MPNN</b></td><td style="padding: 8px 16px; font-weight: bold;">0.609</td><td style="padding: 8px 16px; font-weight: bold;">0.559</td><td style="padding: 8px 16px;">0.82</td></tr>
  </tbody>
</table>

Three things jump out **for this target and this held-out series** (we keep repeating that scope on purpose — none of these are claims about 3D descriptors in general):

**Here, 3D-only doesn't just underperform — it collapses.** A cross-fold R² of ~0.47–0.52 looks like a working model. On this new analog series the same model scores *negative* R² (worse than predicting the training mean) with essentially **zero rank correlation**. The CV→held-out drop is ~0.62 in R² for 3D-only, versus ~0.22 for 2D-only. The conformer descriptors are fitting something real *within the training chemotypes* that, on this particular series, does not carry over.

**2D transfers reasonably well on this target.** 2D-only holds R² ≈ 0.35–0.44 out-of-distribution and a Spearman of 0.67–0.73. It degrades from CV — every model does — but it degrades gracefully instead of inverting.

**Adding 3D to 2D, on this target, is at best a wash.** In cross-fold, 2D+3D edges out 2D by a consistent but tiny ~0.02 R². On the held-out set that margin evaporates into noise: +0.03 for XGBoost, +0.02 for the full PyTorch model — and **−0.10 for the feature-selected PyTorch-50**, where adding 3D features actively made generalization worse. If you only looked at cross-fold here, you would conclude 3D reliably helps. The blind answers, for this target, say otherwise.

And the model that comes out ahead on both columns is **Chemprop** — a learned representation straight off the SMILES graph, no hand-engineered 3D anywhere. More on that below.

## The SHAP Paradox

Here is the part that trips people up. Inside the combined **2D+3D** model, the 3D features look *important*. Ranking the 387 features by mean absolute SHAP value:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Rank</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Feature</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">SHAP</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Type</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 6px 16px;">1</td><td style="padding: 6px 16px;"><b>mollogp</b></td><td style="padding: 6px 16px;">0.250</td><td style="padding: 6px 16px;">2D</td></tr>
    <tr><td style="padding: 6px 16px;">2</td><td style="padding: 6px 16px;">heavyatomcount</td><td style="padding: 6px 16px;">0.046</td><td style="padding: 6px 16px;">2D</td></tr>
    <tr><td style="padding: 6px 16px;">3</td><td style="padding: 6px 16px;">smr_vsa5</td><td style="padding: 6px 16px;">0.033</td><td style="padding: 6px 16px;">2D</td></tr>
    <tr><td style="padding: 6px 16px;">4</td><td style="padding: 6px 16px;">xpc_6d</td><td style="padding: 6px 16px;">0.032</td><td style="padding: 6px 16px;">2D</td></tr>
    <tr><td style="padding: 6px 16px;">5</td><td style="padding: 6px 16px;"><b>pharm3d_nitrogen_span</b></td><td style="padding: 6px 16px;">0.032</td><td style="padding: 6px 16px; color: #c0392b;">3D</td></tr>
    <tr><td style="padding: 6px 16px;">6</td><td style="padding: 6px 16px;">labuteasa</td><td style="padding: 6px 16px;">0.027</td><td style="padding: 6px 16px;">2D</td></tr>
    <tr><td style="padding: 6px 16px;">7</td><td style="padding: 6px 16px;"><b>m3d_fpsa3</b></td><td style="padding: 6px 16px;">0.026</td><td style="padding: 6px 16px; color: #c0392b;">3D</td></tr>
    <tr><td style="padding: 6px 16px;">8</td><td style="padding: 6px 16px;"><b>eccentricity</b></td><td style="padding: 6px 16px;">0.025</td><td style="padding: 6px 16px; color: #c0392b;">3D</td></tr>
    <tr><td style="padding: 6px 16px;">9</td><td style="padding: 6px 16px;">numvalenceelectrons</td><td style="padding: 6px 16px;">0.024</td><td style="padding: 6px 16px;">2D</td></tr>
    <tr><td style="padding: 6px 16px;">10</td><td style="padding: 6px 16px;"><b>pharm3d_elongation</b></td><td style="padding: 6px 16px;">0.023</td><td style="padding: 6px 16px; color: #c0392b;">3D</td></tr>
  </tbody>
</table>

**Four of the top ten features are 3D.** Across the model, 3D descriptors are 32% of the top-50 SHAP features and 25% of all non-zero contributors — despite being only 19% of the feature pool. By the most popular interpretability metric in the field, the conformer features are pulling real weight.

And yet a model built *only* from those same features scores negative R² out-of-distribution. How are both true at once?

Because **SHAP measures a feature's contribution to the model's fit on the data it was scored against — not whether that contribution generalizes.** The 3D descriptors are genuine handles on the *training* chemotypes. `pharm3d_nitrogen_span`, `eccentricity`, and `pharm3d_elongation` correlate with pEC50 *in-sample*, the gradient-booster happily carves decision boundaries along them, and SHAP faithfully reports that those splits move predictions. None of that tells you the relationship survives a shift to a new analog series — and here, it doesn't. The conformer features encode geometry-specific quirks of the compounds the model trained on, and a new series doesn't share them.

Contrast that with the feature SHAP ranks #1, by a factor of five: **`mollogp`** — calculated lipophilicity, a plain 2D descriptor. That is exactly the feature mechanism predicts should dominate. PXR is a famously promiscuous nuclear receptor with a large, hydrophobic ligand-binding pocket; lipophilic compounds are its canonical activators. The model's single most important feature is both the most generalizable one *and* the one a pharmacologist would have named first. The high-SHAP 3D features are riding behind it, contributing to fit without contributing to transfer.

!!! warning "The lesson"
    Feature importance — SHAP, gain, permutation, any of them — is computed against a fixed dataset. It ranks what the model *leaned on*, not what will *hold up* on new chemistry. A descriptor block can be simultaneously high-importance and non-generalizing. The only way to tell the difference is to ablate the block and measure on a genuinely held-out distribution. Importance plots are a hypothesis; the held-out ablation is the test.

## Chemprop Comes Out Ahead Here

On this target, the SMILES-only D-MPNN tops both columns — cross-fold R² 0.609 and held-out R² 0.559, a Spearman of 0.82 on the new series. It never sees a hand-engineered descriptor, 2D or 3D. It learns a representation end-to-end from the molecular graph.

That is at least consistent with the broader ADMET picture: many strong reproducible TDC leaderboard models pair 2D fingerprints with learned graph embeddings, and the learned component does a lot of the work. For PXR — a target whose dominant driver (lipophilicity / size) is well captured by the graph — a learned representation did better than the fixed descriptor catalogs we tried, and the fixed 3D catalog was the part that transferred worst. We would not assume that ordering holds on a geometry-sensitive endpoint.

## A Controlled Test: Adding 3D to the Best Model

The ablations above vary the whole feature set at once. To isolate the 3D contribution more cleanly, we held the model fixed at our strongest one (Chemprop) and *appended* 3D features as extra molecular descriptors (Chemprop's hybrid mode), varying only **how many** 3D features we added: the 6 that land in the combined model's top-20 SHAP, the 16 in its top-50, and — as a control against any feature-selection artifact — all 74.

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Chemprop variant</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">3D features added</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Cross-fold R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out RAE</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out Spearman</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">plain (SMILES only)</td><td style="padding: 8px 16px;">0</td><td style="padding: 8px 16px;">0.609</td><td style="padding: 8px 16px; font-weight: bold;">0.569</td><td style="padding: 8px 16px; font-weight: bold;">0.559</td><td style="padding: 8px 16px;">0.82</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">+ top-20 SHAP 3D</td><td style="padding: 8px 16px;">6</td><td style="padding: 8px 16px;">0.620</td><td style="padding: 8px 16px;">0.637</td><td style="padding: 8px 16px;">0.480</td><td style="padding: 8px 16px;">0.79</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">+ top-50 SHAP 3D</td><td style="padding: 8px 16px;">16</td><td style="padding: 8px 16px;">0.630</td><td style="padding: 8px 16px;">0.704</td><td style="padding: 8px 16px;">0.343</td><td style="padding: 8px 16px;">0.74</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">+ all 3D (control)</td><td style="padding: 8px 16px;">74</td><td style="padding: 8px 16px;">0.631</td><td style="padding: 8px 16px; color: #c0392b; font-weight: bold;">0.772</td><td style="padding: 8px 16px; color: #c0392b; font-weight: bold;">0.268</td><td style="padding: 8px 16px;">0.65</td></tr>
  </tbody>
</table>

*(RAE — relative absolute error — is the challenge's headline metric: total absolute error divided by that of a mean-only baseline, so **lower is better** and 1.0 means "no better than guessing the mean.")*

The two evaluation columns move in **opposite directions**, monotonically:

- **Cross-fold R² goes up** as we add more 3D: 0.609 → 0.620 → 0.630 → 0.631.
- **Held-out accuracy goes down** as we add more 3D: RAE 0.569 → 0.637 → 0.704 → 0.772 (and R² 0.559 → 0.268).

So on this target, the in-distribution metric says "3D helps, add more" at every step, while the out-of-distribution metric says the opposite at every step. Because only the 3D feature count changes, this is a cleaner read than the whole-feature-set ablation — and the **all-74 control is the worst of the four**, so it isn't an artifact of which 3D features we picked. For this endpoint and this held-out series, appending these hand-engineered 3D descriptors to an already-strong learned representation was counterproductive, in proportion to how many we added. We would not assume the same on a target where geometry is the dominant signal.

## So Is 3D Useless? No.

We want to be clear about what this post does and doesn't show. It is **one endpoint, one held-out series, one set of 3D descriptors** — not evidence that 3D features are unhelpful in general. A few honest caveats:

- **PXR is an unusually 2D-friendly target.** Its biology is dominated by lipophilicity and size, which 2D physchem captures well. The [literature](3d_descriptors.md#when-3d-descriptors-help-and-when-they-dont) is clear that 3D features give *real* (if often modest) gains on geometry-sensitive endpoints — passive permeability, P-gp / BCRP recognition, conformer-dependent solubility. PXR isn't one of those, so it's close to a worst case for 3D. On a geometry-driven target we would expect a different result, and we'd run the same ablation to find out.
- **The transferable takeaway is the diagnostic, not the verdict on 3D.** The reusable result is the *method*: a large CV→held-out gap concentrated in one feature block, combined with high in-sample importance for that block, is a signature of features that fit but don't transfer. That's worth checking on any model, with any descriptor family — including ours.
- **3D is best treated as a complement, not a replacement.** Where it helps, it helps *on top of* a strong 2D + learned-representation baseline — and the only way to know for your target is exactly this kind of held-out ablation, never the cross-fold number or the SHAP plot alone.

## Reproducing This

The whole comparison is a small, self-contained DAG: a feature-set producer that builds the 2D / 3D / 2D+3D blocks, then one model script per block plus a Chemprop script. Each model captures `cross_fold`, `full`, and a `pxr_phase1_test` capture on the revealed Analog Set 1, so the held-out R² lands on every endpoint.

```python
from workbench.api import Model

# Pull the two numbers that matter for any of the deployed models
m = Model("pxr-2d-3d-reg-xgb")
cv      = m.get_inference_metrics("full_cross_fold")   # what you see before the answers
heldout = m.get_inference_metrics("pxr_phase1_test")   # the honest generalization number

# The SHAP ranking that looks so convincing — and isn't the whole story
ranked = m.shap_importance()   # [(feature, mean_abs_shap), ...] descending
```

Swap the model name across `pxr-2d-reg-xgb`, `pxr-3d-reg-xgb`, `pxr-2d-3d-reg-xgb`, the PyTorch variants, `pxr-reg-chemprop`, and the hybrids (`pxr-chemprop-hybrid-3d-top20`, `-top50`, `-all`) to rebuild both tables above.

## References

- Huang, K., et al. *"Therapeutics Data Commons: Machine Learning Datasets and Tasks for Drug Discovery and Development."* [TDC ADMET Leaderboards](https://tdcommons.ai/benchmark/overview/)
- Koleiev, I., Stratiichuk, R., Shevchuk, N., et al. *"Critical Assessment of ML Models for ADMET Prediction in TDC Leaderboards."* bioRxiv (2026). [DOI: 10.64898/2026.02.26.708193](https://www.biorxiv.org/content/10.64898/2026.02.26.708193v1)
- Niu, Z., et al. *"PharmaBench: Enhancing ADMET Benchmarks with Large Language Models."* Sci. Data 11, 985 (2024). [DOI: 10.1038/s41597-024-03793-0](https://doi.org/10.1038/s41597-024-03793-0)
- Bahia, M.S., et al. *"Comparison Between 2D and 3D Descriptors in QSAR Modeling Based on Bio-Activities."* Mol. Inform. 42, 2200186 (2023). [DOI: 10.1002/minf.202200186](https://doi.org/10.1002/minf.202200186)
- Lundberg, S.M. & Lee, S.-I. *"A Unified Approach to Interpreting Model Predictions."* NeurIPS 2017. [arXiv: 1705.07874](https://arxiv.org/abs/1705.07874)
- OpenADMET Blind Challenges. [openadmet.org/blindchallenges](https://openadmet.org/blindchallenges/)

## Questions?
<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
</content>
</invoke>
