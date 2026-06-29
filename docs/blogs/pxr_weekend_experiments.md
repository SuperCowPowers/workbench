# A Weekend on the OpenADMET PXR Challenge

!!! tip inline end "Takeaway"
    Almost everything fancy lost to a plain Chemprop D-MPNN. The one win came from the move aligned with the biology — multi-task learning on lipophilicity.

This isn't a "new SOTA" post. We entered the [OpenADMET PXR Induction Blind Challenge](https://openadmet.org/blindchallenges/) (predict human PXR induction, pEC50), submitted late, and spent a weekend seeing what we could learn. The honest summary: a vanilla learned graph model — Chemprop, no hand-engineering — beat almost everything else we built, and the more *unmotivated* machinery we bolted on, the worse out-of-distribution (OOD) generalization got. The lone exception that moved the needle was a multi-task model supervised by logD and logP.

It's worth writing up precisely *because* most of it is the un-exciting result. "Strong baseline plus honest held-out evaluation" is advice everyone gives and few pressure-test in public.

## The Setup

We submitted after Phase 1 closed, so the Phase 2 leaderboard is blind to us — but the challenge revealed the Phase 1 answers (**Analog Set 1**, 253 compounds) afterward. That set is our yardstick, and a good one: it's a **new chemotype**, not a random split, so it measures genuine OOD transfer.

| | |
|---|---|
| **Training set** | 4,139 compounds, pEC50 1.61–7.55 (mean 4.32, σ 1.12) |
| **Held-out set** | Analog Set 1 — 253 compounds, revealed pEC50 (mean 4.66, σ 1.03) |
| **2D features** | 313 RDKit + Mordred 2D descriptors |
| **3D features** | 26 curated GFN2-xTB descriptors (`smiles-to-3d-v2`, built this weekend) |
| **Metric** | **RAE** — relative absolute error, the challenge's headline number |

!!! note "Reading the numbers"
    **RAE** = total absolute error ÷ a mean-only predictor (lower is better; 1.0 = no better than the mean). Every held-out number is on the same 253-compound Analog Set 1, with those rows zero-weighted out of training.

## The Baseline That Wouldn't Break

A from-scratch Chemprop D-MPNN — a learned representation straight off the SMILES graph, no hand-engineered descriptors:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Model</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out RAE ↓</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out ρ</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">n</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 8px 16px; font-weight: bold;">Chemprop D-MPNN (SMILES only)</td><td style="padding: 8px 16px; font-weight: bold;">0.569</td><td style="padding: 8px 16px;">0.559</td><td style="padding: 8px 16px;">0.82</td><td style="padding: 8px 16px;">253</td></tr>
  </tbody>
</table>

A held-out RAE of **0.569** and Spearman **0.82** on a brand-new analog series is genuinely strong — the line every experiment below is measured against, and almost never beats. Why so hard? PXR is a lipophilicity-driven sensor whose dominant signal a learned graph captures well, and Chemprop's defaults already do the things that matter (5-fold ensemble, MAE loss = the RAE objective, scaffold split). Not much obvious headroom — which is what made the weekend interesting.

## Experiment 1 — Do 3D Descriptors Help? (xTB v2)

The obvious lever is 3D conformer geometry. Our first-generation 74-descriptor set ("v1") had never clearly beaten 2D on any ADMET assay, so we rebuilt the 3D layer around a real quantum-chemical engine: ETKDGv3 conformers → MMFF → **GFN2-xTB** (via `tblite`), Boltzmann-averaged into **26 curated descriptors** (electronic, surface, shape, pharmacophore, flexibility). The bet: fewer, physics-grounded features transfer better than 74 noisy ones.

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Model</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Feature Set</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out RAE ↓</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">n</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 8px 16px; font-weight: bold;">Chemprop (baseline)</td><td style="padding: 8px 16px;">SMILES</td><td style="padding: 8px 16px; font-weight: bold;">0.569</td><td style="padding: 8px 16px;">0.559</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">PyTorch (339)</td><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D + 3D&nbsp;v2</td><td style="padding: 8px 16px; font-weight: bold;">0.671</td><td style="padding: 8px 16px;">0.443</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">PyTorch (313)</td><td class="text-teal" style="padding: 8px 16px;">2D only</td><td style="padding: 8px 16px;">0.680</td><td style="padding: 8px 16px;">0.436</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">PyTorch (387)</td><td class="text-teal" style="padding: 8px 16px;">2D + 3D&nbsp;v1 (prior)</td><td style="padding: 8px 16px;">0.685</td><td style="padding: 8px 16px;">0.458</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">XGBoost</td><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D + 3D&nbsp;v2</td><td style="padding: 8px 16px;">0.746</td><td style="padding: 8px 16px;">0.380</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">XGBoost</td><td class="text-teal" style="padding: 8px 16px;">2D only</td><td style="padding: 8px 16px;">0.766</td><td style="padding: 8px 16px;">0.350</td><td style="padding: 8px 16px;">253</td></tr>
  </tbody>
</table>

It's a **partial win**. With 26 features instead of 74, v2 nudges the right way: PyTorch 2D+3D v2 (0.671) beats 2D-only (0.680), where the old v1 block had *hurt* (0.685); XGBoost shows the same ordering. So the rebuild worked — a smaller, grounded 3D block went from slightly-harmful to slightly-helpful. But the best descriptor model of the weekend (0.671) is still **~0.10 RAE behind plain Chemprop** (0.569). A real improvement that doesn't close the gap to the learned representation.

## Experiment 2 — Ensemble Everything (No)

If no single descriptor model beats Chemprop, maybe a blend does. We combined the natural trio — XGBoost, PyTorch, Chemprop:

- **Learned-weight blend:** 0.578 — indistinguishable from Chemprop alone (0.577*); the optimizer just put nearly all weight on Chemprop.
- **Equal-weight blend:** 0.596 — *worse*; averaging in weaker, correlated members drags the strong one down.
- **De-shrinkage:** the held-out predictions are slightly regressed to the mean (slope ≈ 0.94), but correcting for it on held-out data doesn't transfer (0.586).

The standard ensemble lesson, the hard way: a blend only helps with members that are both **strong and diverse**. Here one member dominated and the rest were correlated descriptor models, so the best the ensemble could do was *match* Chemprop. The real lever would be a genuinely different strong member — not more flavors of the same descriptors. (Hold that thought.)

## Experiment 3 — Foundation-Model Warm-Start (CheMeleon)

The trendy idea: warm-start the Chemprop MPNN from a pretrained foundation model (**CheMeleon**, via `from_foundation`). Its validated recipe is a full fine-tune; the LP-FT literature (Kumar et al. 2022) argues a short freeze can help small/OOD regimes. We swept the freeze length:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Model</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Warm-start</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out RAE ↓</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">n</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 8px 16px; font-weight: bold;">Chemprop (baseline)</td><td style="padding: 8px 16px;">none — from scratch</td><td style="padding: 8px 16px; font-weight: bold;">0.577*</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">Chemprop + CheMeleon</td><td style="padding: 8px 16px;">freeze 0 (full fine-tune)</td><td style="padding: 8px 16px; color: #c0392b;">0.696</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">Chemprop + CheMeleon</td><td style="padding: 8px 16px;">freeze 10 (LP-FT)</td><td style="padding: 8px 16px; color: #c0392b;">0.704</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">Chemprop + CheMeleon</td><td style="padding: 8px 16px;">freeze 20 (LP-FT)</td><td style="padding: 8px 16px; color: #c0392b;">0.706</td><td style="padding: 8px 16px;">253</td></tr>
  </tbody>
</table>

Every variant is **~0.12 RAE worse** than from-scratch, and freeze length barely matters. Even freeze-0 — CheMeleon's *own* protocol — loses, so it's not a tuning miss. The likely cause: `from_foundation` pins the MPNN to CheMeleon's pretrained dimensions, replacing the tuned `depth=6 / hidden_dim=700`. On a small, 2D-friendly assay, the purpose-tuned network just fits better. Foundation warm-starts earn their keep on large, geometry-rich tasks; this isn't one.

## Experiment 4 — Multi-Task on Lipophilicity (the one that worked)

Every experiment so far added capacity without a mechanistic reason to expect transfer. This one started from the biology. PXR is a lipophilicity sensor — calculated logP dominates every descriptor model we built. Feeding lipophilicity *in* as a descriptor hurt (Experiment 1), so instead we made the model *learn* it: a multi-task Chemprop with pEC50 as the primary task and public **logP** (52k compounds) and **logD** (4.2k) as auxiliaries supervising the shared encoder. Only the pEC50 head is scored; task weights keep it dominant (`[1.0, 0.2, 0.3]`).

The bet: anchoring the encoder to the property that *drives* the target — with ~13× more lipophilicity data than PXR labels — should transfer better. It did:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Model</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Auxiliary task(s)</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out RAE ↓</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out ρ</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">n</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 8px 16px; font-weight: bold;">Chemprop (baseline)</td><td style="padding: 8px 16px;">none (single-task)</td><td style="padding: 8px 16px;">0.577</td><td style="padding: 8px 16px;">0.530</td><td style="padding: 8px 16px;">0.82</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">Multi-task</td><td class="text-teal" style="padding: 8px 16px;">+ logP</td><td style="padding: 8px 16px; color: #c0392b;">0.586</td><td style="padding: 8px 16px;">0.500</td><td style="padding: 8px 16px;">0.80</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">Multi-task</td><td class="text-teal" style="padding: 8px 16px;">+ logD</td><td style="padding: 8px 16px;">0.561</td><td style="padding: 8px 16px;">0.538</td><td style="padding: 8px 16px;">0.80</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px; font-weight: bold;">Multi-task</td><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">+ logP + logD</td><td style="padding: 8px 16px; font-weight: bold;">0.556</td><td style="padding: 8px 16px; font-weight: bold;">0.568</td><td style="padding: 8px 16px; font-weight: bold;">0.83</td><td style="padding: 8px 16px;">253</td></tr>
  </tbody>
</table>

The combined model lands at **0.556 RAE**, beating the baseline by ~0.021 — small but real (R² and ρ move the same way), and the ordering tells a coherent story:

- **logP alone hurts** (0.586): its 52k rows pull the encoder toward plain lipophilicity, which ignores ionization.
- **logD alone helps** (0.561): logD is lipophilicity at physiological pH — the mechanistically right driver of PXR exposure.
- **Both is best** (0.556): data-rich logP regularizes while logD anchors. logP earns its keep as a *companion*, not alone.

This mirrors the descriptor experiments: there, *injecting* hand-engineered features hurt; here, giving the model more *supervised practice at learning lipophilicity itself* improved transfer. Same signal, opposite verdict — inject vs. teach.

A caveat: single run, ~0.021 RAE (a few × the noise floor). We trust it because three metrics and three variants agree, not because it's large. A real lever — a gentle one.

## What the Weekend Taught Us

The weekend tells a sharper story than "nothing worked" — it's *what kind* of complexity pays off:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Experiment</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Best held-out RAE ↓</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">vs. baseline</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 8px 16px; font-weight: bold;">Plain Chemprop D-MPNN (baseline)</td><td style="padding: 8px 16px; font-weight: bold;">0.569</td><td style="padding: 8px 16px;">&mdash;</td></tr>
    <tr><td style="padding: 8px 16px;">1 — xTB 3D&nbsp;v2 rebuild (best descriptor model)</td><td style="padding: 8px 16px;">0.671</td><td style="padding: 8px 16px; color: #c0392b;">worse</td></tr>
    <tr><td style="padding: 8px 16px;">2 — XGB + PyTorch + Chemprop ensemble</td><td style="padding: 8px 16px;">0.578</td><td style="padding: 8px 16px; color: #c0392b;">~tie (no gain)</td></tr>
    <tr><td style="padding: 8px 16px;">3 — CheMeleon foundation warm-start</td><td style="padding: 8px 16px;">0.696</td><td style="padding: 8px 16px; color: #c0392b;">worse</td></tr>
    <tr><td style="padding: 8px 16px; font-weight: bold;">4 — Multi-task on logD + logP</td><td style="padding: 8px 16px; font-weight: bold;">0.556</td><td style="padding: 8px 16px; color: #2e7d32; font-weight: bold;">better (&minus;0.021)</td></tr>
  </tbody>
</table>

Two takeaways we'll carry forward:

1. **The strong simple baseline *is* the experiment.** A from-scratch D-MPNN with sensible defaults set a bar that better descriptors, ensembling, and a foundation model all failed to clear. Most of the value was in establishing how good the simple thing already was — so we believed the small win when it finally came.
2. **Complexity pays off when it's aligned with the mechanism, not when it's just more.** The losers added capacity for its own sake; the winner added supervised practice at the property that drives the target. Injecting lipophilicity as a feature hurt; teaching the encoder to predict it helped.

None of this is a verdict on 3D descriptors, ensembles, or foundation models in general — it's one endpoint, one held-out series, one weekend. PXR is unusually 2D-friendly and lipophilicity-driven, which is exactly why a lipophilicity-anchored multi-task model is what moved it. The transferable result is the method: strong baseline, genuinely shifted held-out set, and make every addition beat it there before you believe it.

## Reproducing This

The whole thing is a small Workbench DAG: a feature-set producer, one model script per experiment, and a Chemprop baseline. Each model captures a `pxr_phase1_test` inference on the revealed Analog Set 1, so the held-out numbers land on every endpoint.

```python
import numpy as np
from workbench.api import Model

def held_out_rae(model_name):
    """RAE on the revealed Analog Set 1 (lower is better; 1.0 = mean-only)."""
    df = Model(model_name).get_inference_predictions("pxr_phase1_test")
    y, p = df["pec50"].to_numpy(float), df["prediction"].to_numpy(float)
    return np.abs(y - p).mean() / np.abs(y - y.mean()).mean()

print(held_out_rae("pxr-reg-chemprop"))             # single-task baseline, ~0.569
print(held_out_rae("pxr-2d-3dv2-reg-pytorch-339"))  # best descriptor model, ~0.671
print(held_out_rae("pxr-reg-chemprop-mt-both"))     # multi-task winner, ~0.556
```

*\* Two from-scratch Chemprop numbers appear above (0.569 and 0.577): same recipe, where 0.569 is the deployed full-pool model and 0.577 zero-weights Analog Set 1 out of training (the matched control for the later experiments). The ~0.008 gap is training stochasticity, not signal.*

## References

- Heid, E., et al. *"Chemprop: A Machine Learning Package for Chemical Property Prediction."* J. Chem. Inf. Model. 64, 9–17 (2024). [DOI: 10.1021/acs.jcim.3c01250](https://doi.org/10.1021/acs.jcim.3c01250)
- Kumar, A., et al. *"Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution."* ICLR 2022. [arXiv: 2202.10054](https://arxiv.org/abs/2202.10054)
- Bannwarth, C., et al. *"GFN2-xTB — An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method."* J. Chem. Theory Comput. 15, 1652–1671 (2019). [DOI: 10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- Bahia, M.S., et al. *"Comparison Between 2D and 3D Descriptors in QSAR Modeling Based on Bio-Activities."* Mol. Inform. 42, 2200186 (2023). [DOI: 10.1002/minf.202200186](https://doi.org/10.1002/minf.202200186)
- OpenADMET Blind Challenges. [openadmet.org/blindchallenges](https://openadmet.org/blindchallenges/)

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
