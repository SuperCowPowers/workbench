# HPO: WIP
!!! tip inline end "HPO in Workbench"
    Hyperparameter search runs inside a single training job for ChemProp, XGBoost, and PyTorch. See the [Hyperparameter Optimization](../models/hpo.md) page for the API.

*This document is a work in progress: the results below are cross-validated, and we intend to rerun the comparison against true held-out sets before drawing firm conclusions.*

<figure style="margin: 20px 0; width: 100%; display: block;">
<img src="../../models/images/hpo_overview.svg" alt="HPO wrapping model creation: a search driver hands hyperparameters to the model templates, gets scores back, and only the winner is published" style="width: 100%; height: auto; display: block;">
<figcaption><em>HPO wraps normal model creation — same training code, many configurations, one published model.</em></figcaption>
</figure>

Hyperparameter search has an obvious implementation: try a few hundred configurations, keep the one with the best score. We built that, ran it on five real datasets, and found that the best-scoring configuration was **usually the wrong one to publish**.

This post is about why, what we do instead, and what the numbers actually look like.

## A search reports a biased number

Every trial's score is an estimate with noise in it. Fold assignment, weight initialization, batch ordering — run the same configuration twice and you get two different numbers.

Now take the minimum over 250 such estimates. You are not selecting for the best configuration; you are selecting for the best *combination* of configuration and luck. The reported minimum is therefore optimistically biased, and the bias grows with the number of trials. This is the same selection effect that makes cross-validated model selection scores untrustworthy as performance estimates — Cawley and Talbot's 2010 paper is the canonical treatment.

The practical consequence: the trial at the top of the leaderboard is disproportionately likely to be there because it drew a lucky evaluation, not because it is genuinely better.

## We measured it

Workbench treats the search as a **shortlist**. After the search finishes, a second stage re-scores the top finalists on fresh trainings and publishes whichever wins *there*. That gives us a natural experiment: how often does the search's own #1 survive re-scoring?

Across five searches on real datasets:

<table style="width: 100%;">
  <thead>
    <tr>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Search</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Trials</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Gain over baseline</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">Which finalist won</th>
      <th style="background-color: rgba(58, 134, 255, 0.5); color: white; padding: 10px 16px;">From trial #</th>
    </tr>
  </thead>
  <tbody>
    <tr><td style="padding: 8px 16px;">AqSol ChemProp</td><td style="padding: 8px 16px;">60</td><td class="text-orange" style="padding: 8px 16px; font-weight: bold;">+6.0%</td><td style="padding: 8px 16px;">rank 2</td><td style="padding: 8px 16px;">51</td></tr>
    <tr><td style="padding: 8px 16px;">AqSol XGBoost</td><td style="padding: 8px 16px;">250</td><td class="text-orange" style="padding: 8px 16px; font-weight: bold;">+1.7%</td><td style="padding: 8px 16px;">rank 2</td><td style="padding: 8px 16px;">35</td></tr>
    <tr><td style="padding: 8px 16px;">PXR XGBoost</td><td style="padding: 8px 16px;">250</td><td class="text-orange" style="padding: 8px 16px; font-weight: bold;">+2.7%</td><td style="padding: 8px 16px;">rank 2</td><td style="padding: 8px 16px;">232</td></tr>
    <tr><td style="padding: 8px 16px;">PXR PyTorch</td><td style="padding: 8px 16px;">100</td><td class="text-orange" style="padding: 8px 16px; font-weight: bold;">+6.3%</td><td style="padding: 8px 16px;">rank 1</td><td style="padding: 8px 16px;">21</td></tr>
    <tr><td style="padding: 8px 16px;">AqSol XGBoost (narrowed space)</td><td style="padding: 8px 16px;">100</td><td class="text-orange" style="padding: 8px 16px; font-weight: bold;">+0.4%</td><td style="padding: 8px 16px;">rank 3</td><td style="padding: 8px 16px;">6</td></tr>
  </tbody>
</table>

**The search's top trial won exactly once out of five.** Four times the configuration we published was one the leaderboard had ranked second or third. If we had shipped the search's answer directly, we would have shipped a luckier-but-not-better configuration in 80% of these runs.

Look at the last column too. The winning configurations came from trials 6, 21, 35, 51, and 232. Two searches ran 250 trials and published something found in the first 35. There is no relationship between how long the search ran and where in the run the eventual winner appeared — which is exactly what you would expect if late "improvements" are mostly noise.

## The baseline is the important part

The second stage re-scores the finalists, and it also re-scores one more candidate: **your own untuned hyperparameters**.

That single addition is what bounds the downside. A search that finds nothing real loses to your settings, and your settings get published unchanged. Ties go to you. The failure mode where an expensive search quietly makes your model worse — which is otherwise entirely possible, since the search's winner is biased and your baseline was never scored on the same footing — cannot happen.

It also makes "the search found nothing" a legitimate, reportable outcome rather than a broken run. You spent the compute and learned that your configuration was already good. That is worth knowing, and it is worth *saying* rather than papering over with a number that looks like an improvement.

## Which knobs actually mattered

Once you have the trials, you can ask which hyperparameters moved the objective at all. Workbench fits a random-forest surrogate to the search's own trials and reports two numbers:

```python
model.hpo_importance()
```

| knob | importance | effect | best |
|---|---|---|---|
| `max_lr` | 0.51 | 7.9% | 0.0043 |
| `batch_size` | 0.37 | 7.8% | 512 |
| `hidden_dim` | 0.07 | 2.4% | 100 |
| `depth` | 0.02 | 0.5% | 5 |

`importance` is a share and always sums to 1 — so in a search where nothing mattered, something still looks important. `effect` is the absolute read: how far the objective actually moves across that knob's range. **A knob earns tuning only when both are high.** `depth` above holds a real share of very little.

We pointed this at our own ChemProp search space and it cost a knob its place. `warmup_epochs` had been in the space since the beginning, on the reasonable theory that learning-rate warmup matters. Across three searches its measured effect was 0.06%, 0.02%, and 0.10% of the objective — and its rank correlation with the objective flipped sign between datasets, which is the signature of noise rather than signal. On one run the sampler spent 25 of 60 trials chasing it.

It is no longer in the search space. In a 60-trial budget, an inert knob is not free — it is a whole dimension the sampler explores instead of the ones that pay.

## What to expect

Two honest caveats, because hyperparameter search is oversold.

**Framework defaults are a strong baseline.** ChemProp's defaults in particular come from people who tuned them on molecular data; the literature finds ChemProp HPO to be roughly a coin flip against them on small datasets. Our gains above run from +0.4% to +6.3% on cross-validated MAE. Real, but not transformative.

**Cross-validated gains do not imply out-of-distribution gains.** On the OpenADMET PXR challenge, where the held-out set is a distinct analog series rather than a random split, our searched models *lost* to stock defaults — despite winning in-distribution. Tuning finds the configuration best suited to the chemistry you already have. If your evaluation is a new chemical series, that is not the same objective, and a search optimizing the first will not reliably improve the second.

This is why the objective defaults to out-of-fold cross-validation on training rows, and why rows designated through `validation_ids` are held out of the search entirely. They stay an honest benchmark, and a benchmark you tuned against is not a benchmark.

## Practical takeaways

1. **Never quote the search's best value as the model's improvement.** It is the biased number. Quote the re-ranked pair — `best_value` against `baseline_value` — which is what the publish decision actually turned on.
2. **Treat a baseline win as information, not failure.** It means your configuration was already competitive and you now have evidence for that.
3. **Check `hpo_importance()` before widening a search.** Knobs that don't move the objective are a tax on the budget, and you cannot tell which ones those are without looking.
4. **More trials is not obviously better.** Our winners came from trial 6 as often as trial 232. Budget buys shortlist quality, not a guaranteed better model.
5. **Evaluate out-of-distribution if that's how the model will be used.** In-distribution improvement is the easy part.

## References

**Selection bias in model selection** — why the best-of-N score is not an unbiased performance estimate:

- Cawley, G.C., Talbot, N.L.C. *"On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation."* Journal of Machine Learning Research 11, 2079–2107 (2010). [https://jmlr.org/papers/v11/cawley10a.html](https://jmlr.org/papers/v11/cawley10a.html)

**Search backends** — the samplers and schedulers Workbench builds on:

- Akiba, T., Sano, S., Yanase, T., et al. *"Optuna: A Next-generation Hyperparameter Optimization Framework."* KDD (2019). [arXiv:1907.10902](https://arxiv.org/abs/1907.10902)
- Li, L., Jamieson, K., Rostamizadeh, A., et al. *"A System for Massively Parallel Hyperparameter Tuning."* MLSys (2020). [arXiv:1810.05934](https://arxiv.org/abs/1810.05934)

**HPO for molecular property prediction** — what the field measures on this class of model:

- Yang, K., Swanson, K., Jin, W., et al. *"Analyzing Learned Molecular Representations for Property Prediction."* Journal of Chemical Information and Modeling 59(8), 3370–3388 (2019). [DOI: 10.1021/acs.jcim.9b00237](https://doi.org/10.1021/acs.jcim.9b00237)
- *"BOOM: Benchmarking Out-Of-distribution Molecular property predictions."* [arXiv:2505.01912](https://arxiv.org/abs/2505.01912)

## Questions?
<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS and Workbench. Please contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com) or on chat us up on [Discord](https://discord.gg/WHAJuz8sw8)
