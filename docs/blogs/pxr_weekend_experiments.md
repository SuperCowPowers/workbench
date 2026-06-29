# A Weekend on the OpenADMET PXR Challenge

!!! tip inline end "The one-line takeaway"
    We spent a weekend trying to beat a plain, from-scratch Chemprop D-MPNN on the OpenADMET PXR induction challenge. Better 3D descriptors, a multi-model ensemble, a foundation-model warm-start — **everything fancy scored worse** on the revealed held-out analog series. The *one* thing that helped was the un-flashy, mechanistically-motivated move: multi-task learning on lipophilicity (logD/logP), PXR's dominant driver. Complexity-for-its-own-sake lost; complexity aligned with the biology won — modestly.

This isn't a "new SOTA" post. We entered the [OpenADMET PXR Induction Blind Challenge](https://openadmet.org/blindchallenges/) (predict human PXR induction, pEC50), submitted late, and then spent a weekend poking at the problem to see what we could learn. The honest summary: a vanilla learned graph model — Chemprop, no hand-engineering — beat almost everything else we built, and the more *unmotivated* machinery we bolted on, the worse out-of-distribution generalization got. The exception that finally moved the needle was a multi-task model that supervises the shared encoder with logD and logP.

That's worth writing down precisely *because* most of it is the un-exciting result. Strong simple baselines plus honest held-out evaluation is the boring advice everyone gives and few people pressure-test on themselves in public. So here's us doing it — including the one place added complexity actually paid off.

## The Setup

We submitted after Phase 1 closed, so the Phase 2 leaderboard is blinded to us. Fortunately the challenge revealed the Phase 1 answers — **Analog Set 1**, 253 compounds — after that phase concluded. That revealed set is our honest yardstick for everything below.

It's a *good* yardstick: Analog Set 1 is a **new chemotype**, not a random split of the training pool, so it measures genuine out-of-distribution (OOD) transfer — exactly where modeling choices get exposed.

| | |
|---|---|
| **Training set** | 4,139 compounds, pEC50 1.61–7.55 (mean 4.32, σ 1.12) |
| **Held-out set** | Analog Set 1 — 253 compounds, revealed pEC50 (mean 4.66, σ 1.03) |
| **2D features** | 313 RDKit + Mordred 2D descriptors (`smiles-to-2d-v1`) |
| **3D features (v1)** | 74 Boltzmann-ensemble conformer descriptors (`smiles-to-3d-full-v1`) |
| **3D features (v2)** | 26 curated GFN2-xTB descriptors (`smiles-to-3d-v2`, built this weekend) |
| **Metric** | **RAE** — relative absolute error, the challenge's headline number |

!!! note "Reading the numbers"
    **RAE** (relative absolute error) is total absolute error divided by that of a mean-only predictor, so **lower is better** and 1.0 means "no better than guessing the mean." Every held-out number below is on the same 253-compound Analog Set 1 (so **n = 253** throughout), and for the descriptor models the Analog Set 1 rows are zero-weighted out of training, so the held-out score is honest. We also show held-out R² and Spearman ρ where useful.

## The Baseline That Wouldn't Break

Before any experiments, here's the wall we kept running into. A from-scratch Chemprop D-MPNN — a learned representation straight off the SMILES graph, no hand-engineered descriptors of any kind:

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

A held-out RAE of **0.569** and Spearman **0.82** on a brand-new analog series is a genuinely strong number. Hold it in your head; it's the line every experiment below is measured against — and never beats.

Why is a plain D-MPNN this hard to beat here? PXR is a promiscuous, lipophilicity-driven xenobiotic sensor with a large hydrophobic ligand pocket. Its dominant signal (size + lipophilicity) is well captured by a learned 2D graph representation, and Chemprop's defaults already do the things that usually matter: a 5-fold ensemble, MAE loss (which *is* the RAE objective), and a scaffold split. There isn't much obvious headroom — which is the whole reason the weekend was interesting.

## Experiment 1 — Do 3D Descriptors Help? (No, and the SHAP plot lies about it)

The first idea was the obvious one: add 3D conformer geometry. We ran clean 2D / 3D / 2D+3D ablations with both XGBoost and PyTorch Tabular, using our original 74-feature 3D block (v1).

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Feature Set</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Model</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Cross-fold R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out R²</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Held-out ρ</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D</td><td style="padding: 8px 16px;">XGBoost</td><td style="padding: 8px 16px;">0.566</td><td style="padding: 8px 16px;">0.350</td><td style="padding: 8px 16px;">0.67</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">3D</td><td style="padding: 8px 16px;">XGBoost</td><td style="padding: 8px 16px;">0.519</td><td style="padding: 8px 16px; color: #c0392b; font-weight: bold;">&minus;0.101</td><td style="padding: 8px 16px;">&mdash;</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D + 3D</td><td style="padding: 8px 16px;">XGBoost</td><td style="padding: 8px 16px;">0.587</td><td style="padding: 8px 16px;">0.381</td><td style="padding: 8px 16px;">0.70</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D</td><td style="padding: 8px 16px;">PyTorch (313)</td><td style="padding: 8px 16px;">0.531</td><td style="padding: 8px 16px;">0.437</td><td style="padding: 8px 16px;">0.73</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">3D</td><td style="padding: 8px 16px;">PyTorch (74)</td><td style="padding: 8px 16px;">0.472</td><td style="padding: 8px 16px; color: #c0392b; font-weight: bold;">&minus;0.118</td><td style="padding: 8px 16px;">&mdash;</td></tr>
    <tr><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D + 3D</td><td style="padding: 8px 16px;">PyTorch (387)</td><td style="padding: 8px 16px;">0.549</td><td style="padding: 8px 16px; font-weight: bold;">0.458</td><td style="padding: 8px 16px;">0.73</td></tr>
  </tbody>
</table>

Two things to notice. **3D-only collapses** — a respectable cross-fold R² (~0.47–0.52) inverts to *negative* held-out R² on the new series, with essentially zero rank correlation. And **adding 3D to 2D is, at best, a wash** — a tiny consistent edge in cross-fold that evaporates into noise out-of-distribution. Neither 2D+3D model comes anywhere near the Chemprop baseline's 0.559 held-out R².

### The SHAP paradox

Here's the part that trips people up. Inside the combined **2D+3D** model, the 3D features look *important*. Ranked by mean absolute SHAP, four of the top ten features are 3D (`pharm3d_nitrogen_span`, `m3d_fpsa3`, `eccentricity`, `pharm3d_elongation`), and 3D descriptors make up 32% of the top-50 SHAP features despite being only 19% of the pool.

And yet a model built *only* from those same features scores negative R² out-of-distribution. Both are true at once because:

!!! warning "High importance ≠ transferable signal"
    **SHAP measures a feature's contribution to the model's fit on the data it was scored against — not whether that contribution generalizes.** The 3D descriptors are real handles on the *training* chemotypes; the booster happily splits on them and SHAP faithfully reports it. None of that survives a shift to a new analog series. Importance plots are a hypothesis; the held-out ablation is the test.

The feature that ranks #1 by a factor of five is **`mollogp`** — plain 2D calculated lipophilicity, exactly what PXR pharmacology says should dominate. The high-SHAP 3D features ride behind it, padding the fit without adding transfer.

*(For the longer treatment of the 3D pipeline and where 3D descriptors genuinely do help — permeability, P-gp recognition, conformer-dependent solubility — see the [3D descriptor deep-dive](3d_descriptors.md). PXR, dominated by lipophilicity, is close to a worst case for 3D.)*

## Experiment 2 — Rebuild the 3D Layer From Scratch (xTB v2)

Our 74-feature v1 3D block is a first-generation guess, and it had never clearly beaten 2D on *any* ADMET assay we'd tried. So rather than keep defending it, we rebuilt the 3D layer from the ground up around a real quantum-chemical engine: ETKDGv3 conformers → MMFF optimization → **GFN2-xTB** (via `tblite`) ranking and Boltzmann-weighted averaging, producing **26 curated descriptors** — electronic (xTB dipole, quadrupole, HOMO/LUMO gap, hardness, electrophilicity, partial-charge stats), surface (charge-weighted PSA, polar/apolar SASA), shape, pharmacophore geometry, and flexibility.

The bet: fewer, more physically-grounded features should transfer better than 74 noisy ones. At a matched model, on the held-out set:

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
    <tr><td style="padding: 8px 16px;">PyTorch (387)</td><td class="text-teal" style="padding: 8px 16px;">2D + 3D&nbsp;v1</td><td style="padding: 8px 16px;">0.685</td><td style="padding: 8px 16px;">0.458</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">XGBoost</td><td class="text-teal" style="padding: 8px 16px; font-weight: bold;">2D + 3D&nbsp;v2</td><td style="padding: 8px 16px;">0.746</td><td style="padding: 8px 16px;">0.380</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">XGBoost</td><td class="text-teal" style="padding: 8px 16px;">2D + 3D&nbsp;v1</td><td style="padding: 8px 16px;">0.762</td><td style="padding: 8px 16px;">&mdash;</td><td style="padding: 8px 16px;">253</td></tr>
    <tr><td style="padding: 8px 16px;">XGBoost</td><td class="text-teal" style="padding: 8px 16px;">2D only</td><td style="padding: 8px 16px;">0.766</td><td style="padding: 8px 16px;">0.350</td><td style="padding: 8px 16px;">253</td></tr>
  </tbody>
</table>

This one is a **partial win** — the only one of the weekend. With **26 features instead of 74**, v2 actually moves the needle the right way: for PyTorch, 2D+3D v2 (0.671) beats 2D-only (0.680), whereas the old v1 block had *hurt* (0.685). For XGBoost, v2 (0.746) edges out v1 (0.762) and 2D-only (0.766). So the rebuild did its job: a smaller, physics-grounded 3D block turned a slightly-harmful descriptor set into a slightly-helpful one.

But look at the baseline row. Our best descriptor model of the entire weekend — xTB v2, PyTorch-339 at **0.671** — is still **~0.10 RAE behind plain Chemprop (0.569)**. A real improvement over v1 that doesn't come close to closing the gap to the learned representation.

## Experiment 3 — Ensemble Everything (No)

If no single descriptor model beats Chemprop, maybe a *blend* of diverse models does. We took the natural trio — XGBoost, PyTorch Tabular, and Chemprop — and tried to combine them. It didn't work, in an instructive way:

- **Learned-weight blend:** 0.578 — statistically indistinguishable from Chemprop alone (0.577*). The optimizer simply put almost all the weight on Chemprop.
- **Equal-weight blend:** 0.596 — *worse* than Chemprop. Averaging in weaker, correlated members drags the strong one down.
- **De-shrinkage trick:** the held-out predictions are slightly regressed to the mean (slope ≈ 0.94), but correcting for it on held-out data doesn't transfer and pushed RAE to 0.586.

The lesson is the standard one about ensembles, learned the hard way: a blend only helps when members are both **strong and diverse**. Here one member dominated and the others were correlated descriptor models, so the best the ensemble could do was *recover* Chemprop's score, never exceed it. The real lever would be a genuinely different strong member (a second graph model, a learned embedding) — not more flavors of the same tabular descriptors.

## Experiment 4 — A Foundation-Model Warm-Start (CheMeleon)

The last idea was the trendy one: warm-start the Chemprop MPNN from a pretrained molecular foundation model. We used **CheMeleon** via Chemprop's `from_foundation` hook. CheMeleon's own validated recipe is a full fine-tune (freeze 0 epochs); the linear-probe-then-fine-tune literature (Kumar et al. 2022) argues a short freeze can help small + OOD regimes like ours. Rather than guess, we swept the freeze length and scored each on the held-out set:

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

Every variant is **~0.12 RAE worse** than the from-scratch baseline, and the freeze length barely matters (0.696 → 0.706, monotone but tiny next to the gap). Even freeze-0 — CheMeleon's *own* validated protocol — loses badly, so this isn't a freeze-tuning mistake.

The likely cause: `from_foundation` pins the MPNN to CheMeleon's pretrained architecture (its depth and hidden dimensions), replacing the tuned `depth=6 / hidden_dim=700` that Workbench's from-scratch Chemprop uses. On a small, 2D-friendly assay, the purpose-tuned from-scratch network simply fits better than the general-purpose pretrained one. Foundation warm-starts earn their keep on large, hard, geometry-rich tasks; this isn't one.

## Experiment 5 — Multi-Task on Lipophilicity (the one that worked)

Every experiment so far added machinery without a mechanistic reason to expect transfer: *maybe* geometry helps, *maybe* a blend helps, *maybe* a foundation model helps. This last one started from the biology instead. PXR is a lipophilicity sensor — calculated logP was the single most important feature in Experiment 1, by 5×. So instead of feeding lipophilicity in as a descriptor (which hurt), we made the model *learn* it: a multi-task Chemprop where pEC50 is the primary task and public **logP** (52k compounds) and **logD** (4.2k) are auxiliary tasks supervising the shared MPNN encoder. Only the pEC50 head is scored; the auxiliaries just shape the representation. Task weights keep pEC50 dominant in the gradient (`[1.0, 0.2, 0.3]` for pEC50/logP/logD).

The bet: anchoring the encoder to the property that *drives* the target — using ~13× more lipophilicity data than we have PXR labels — should transfer to a new chemotype better than a PXR-only model. It did:

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

The combined model lands at **0.556 RAE**, beating the single-task baseline by ~0.021 — small, but real: **R² and Spearman move the same way** (0.530 → 0.568, 0.82 → 0.83), and the ordering across the three variants tells a coherent mechanistic story rather than a random one:

- **logP alone *hurts*** (0.586). Its 52k rows dominate the encoder and pull it toward plain lipophilicity, which ignores ionization — not quite PXR's axis.
- **logD alone *helps*** (0.561). logD is lipophilicity at physiological pH (it accounts for charge state), which is the mechanistically correct driver of PXR exposure — exactly the auxiliary you'd predict should work.
- **Both together is best** (0.556). The data-rich logP regularizes the shared representation while logD keeps it anchored to the right property. logP earns its keep as a *companion*, not on its own.

This is the mirror image of the descriptor experiments. There, lipophilicity helped most *as a thing the graph model already learns*, and bolting on extra hand-engineered features hurt. Here, giving the model *more supervised practice at learning lipophilicity itself* — on a far larger, related dataset — is what finally improved transfer to the new series. Same underlying signal, opposite verdict, depending on whether you inject it or teach it.

A caveat for honesty: this is a single training run, and ~0.021 RAE is a modest gain (a few × the run-to-run noise floor). We trust it because three metrics and three variants agree, not because it's large. It's a real lever — just a gentle one.

## What the Weekend Actually Taught Us

Lined up, the weekend tells a sharper story than "nothing worked" — it's *what kind* of complexity pays off:

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
    <tr><td style="padding: 8px 16px;">1 — 2D / 3D&nbsp;v1 descriptor ablations</td><td style="padding: 8px 16px;">0.685</td><td style="padding: 8px 16px; color: #c0392b;">worse</td></tr>
    <tr><td style="padding: 8px 16px;">2 — xTB 3D&nbsp;v2 rebuild (best descriptor model)</td><td style="padding: 8px 16px;">0.671</td><td style="padding: 8px 16px; color: #c0392b;">worse</td></tr>
    <tr><td style="padding: 8px 16px;">3 — XGB + PyTorch + Chemprop ensemble</td><td style="padding: 8px 16px;">0.578</td><td style="padding: 8px 16px; color: #c0392b;">~tie (no gain)</td></tr>
    <tr><td style="padding: 8px 16px;">4 — CheMeleon foundation warm-start</td><td style="padding: 8px 16px;">0.696</td><td style="padding: 8px 16px; color: #c0392b;">worse</td></tr>
    <tr><td style="padding: 8px 16px; font-weight: bold;">5 — Multi-task on logD + logP</td><td style="padding: 8px 16px; font-weight: bold;">0.556</td><td style="padding: 8px 16px; color: #2e7d32; font-weight: bold;">better (&minus;0.021)</td></tr>
  </tbody>
</table>

Three takeaways we'll actually carry forward:

1. **A strong simple baseline is the experiment.** A from-scratch D-MPNN with sensible defaults (ensemble, MAE loss, scaffold split) set a bar that better descriptors, ensembling, and a foundation model all failed to clear. Most of the weekend's value was in *establishing how good the simple thing already was* — so that when something finally beat it, we believed the small margin.
2. **Cross-fold lies; held-out doesn't.** Almost every dead end (3D-only, 2D+3D, the hybrids) looked fine or even good in cross-validation and only revealed itself on the new analog series. The recurring failure signature — high cross-fold and high feature-importance, low held-out — is the thing to watch for, on any model with any feature family.
3. **Complexity pays off when it's aligned with the mechanism, not when it's just more.** The four things that lost added capacity for its own sake — more features, more models, more pretraining. The one that won added *supervised practice at the property that drives the target* (lipophilicity), using a far larger related dataset. Same lesson from two sides: injecting lipophilicity as a feature hurt; teaching the encoder to predict it helped.

None of this is a verdict on 3D descriptors, ensembles, or foundation models in general — it's one endpoint, one held-out series, one weekend. PXR is an unusually 2D-friendly, lipophilicity-driven target, which is exactly why a lipophilicity-anchored multi-task model is the thing that moved it. The transferable result is the *method*: pick a strong baseline, hold out a genuinely shifted set, make every addition beat it there before you believe it — and when you reach for complexity, reach for the kind your domain knowledge says should matter.

## Reproducing This

The whole thing is a small Workbench DAG: one feature-set producer (2D + both 3D blocks), one model script per experiment, and a Chemprop baseline. Each model captures a `pxr_phase1_test` inference on the revealed Analog Set 1, so the held-out numbers above land on every endpoint.

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

*\* Two from-scratch Chemprop numbers appear above (0.569 and 0.577). They're the same recipe — 0.569 is the deployed model trained on the full pool; 0.577 is the variant that zero-weights Analog Set 1 out of training, used as the matched control for the CheMeleon sweep. The ~0.008 gap is training stochasticity, not signal.*

## References

- Heid, E., et al. *"Chemprop: A Machine Learning Package for Chemical Property Prediction."* J. Chem. Inf. Model. 64, 9–17 (2024). [DOI: 10.1021/acs.jcim.3c01250](https://doi.org/10.1021/acs.jcim.3c01250)
- Kumar, A., et al. *"Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution."* ICLR 2022. [arXiv: 2202.10054](https://arxiv.org/abs/2202.10054)
- Bannwarth, C., et al. *"GFN2-xTB — An Accurate and Broadly Parametrized Self-Consistent Tight-Binding Quantum Chemical Method."* J. Chem. Theory Comput. 15, 1652–1671 (2019). [DOI: 10.1021/acs.jctc.8b01176](https://doi.org/10.1021/acs.jctc.8b01176)
- Lundberg, S.M. & Lee, S.-I. *"A Unified Approach to Interpreting Model Predictions."* NeurIPS 2017. [arXiv: 1705.07874](https://arxiv.org/abs/1705.07874)
- Bahia, M.S., et al. *"Comparison Between 2D and 3D Descriptors in QSAR Modeling Based on Bio-Activities."* Mol. Inform. 42, 2200186 (2023). [DOI: 10.1002/minf.202200186](https://doi.org/10.1002/minf.202200186)
- OpenADMET Blind Challenges. [openadmet.org/blindchallenges](https://openadmet.org/blindchallenges/)

## Questions?
<img align="right" src="../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
