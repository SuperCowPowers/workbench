# Blind Challenges

> what transfers to any blind prediction contest: ranking vs placement, splits, metrics, label mining

Rules distilled from published methodology reports by top finishers in blind ADMET challenges,
plus our own measurement record. Numbers quoted are from those contests and are illustrative —
the rule is what transfers, not the figure.

## The one rule: 90% of effort on ranking

**Ranking** is the model knowing which compound is more potent. **Placement** is making the
prediction distribution match the test distribution — affine rescaling, moment matching, bias
correction.

Once predictions are moment-matched, every error metric collapses to a function of the
correlation `rho` alone, so placement closes the gap to that ceiling once and only ranking
raises it. Placement is fast, closed-form and immediately measurable while ranking is slow and
uncertain — which is why it feels productive and is the trap. Half a day at the end, not a thread.

**Derive the transform instead of searching for it.** For squared error the optimum is
shrinkage, `k = rho`, giving `R2 = rho^2` — not `k = 1`, which gives a strictly worse
`2*rho - 1`. Moment matching only wins where the metric punishes under-dispersion directly, as
a hinge does. Work out which case the metric is in first.

**Grade mechanisms after placement, never before**, or a gain living in the mean or spread gets
double-counted. Report Spearman beside any error delta — placement cannot touch it, so it is
the honest read. Split a weak target's error into offset and ordering before accepting a noise
ceiling.

## Build the validation split to match the test set

Do this before any modelling. Analog-enriched test sets are the norm: a blind set is often
built by taking the strongest hits and pulling catalog neighbours of each, so train-test
similarity is high. Scaffold and Butina splits deliberately *separate* analogs, measuring a
harder and different regime than the one being scored.

Compute the nearest-training-neighbour similarity distribution of the test set, then pick a
holdout whose distribution matches it. The symptom of getting this wrong is a fixed offset
between out-of-fold and leaderboard scores.

- **Score it over the label range the test set occupies**, not the labelled subset. A CV
  scoring only hit-range rows cannot reward information about inactives, which are usually most
  of the test set. Mechanisms judged null there are not falsifiable.
- **Bootstrap in the unit the test set was drawn in** — resample analog clusters, not rows.
- **Expect a residual gap.** No split closes it entirely. Use CV to compare *methods* and the
  live board to choose among *candidates*.
- **Know what the ruler resolves.** If model-to-model differences run below the split's
  resolution, most comparisons return "cannot tell" — say that rather than reporting a sign.

## Read the metric for exploitable asymmetry

Ask four questions of any metric: is the penalty non-quadratic (hinge, clipped, interval)?
Does the tolerance vary with the true value? Is it asymmetric around the truth? Is it
normalized by something a submission can influence?

Any yes means the optimal prediction is **not** the conditional mean. Simulate expected metric
cost as a function of where the prediction is placed and submit the argmin — an afternoon, no
training. For an interval hinge with wide intervals at low activity, this shifts low
predictions up. Checking whether the metric *ranks models* like MAE does is a different
question and does not answer this one.

On a classification track the decision threshold is a tuned parameter — MCC is highly
sensitive to call rate at low positive rates, so pick thresholds out of fold.

## Mine the label-generating process, not just the labels

The largest single gain available, and it is free.

- **Diagnose selection bias in which compounds got the expensive measurement.** If a cheap
  primary screen gates an expensive dose-response curve, the measured labels skew active while
  the test set does not. Check per target; it differs.
- **Impute point labels from the cheap readout** — isotonic regression from screen readout to
  fitted label. Point labels move the distribution; bounds under a bounded loss do not, because
  the gradient is zero below the bound, so they can only remove bias that recalibration removes
  for free.
- **Train the cheap readout as an active/inactive classifier.** The cheapest probe on the
  board: gradient boosting on fingerprints over screen hit-versus-non-hit has beaten a full
  stack at band separation on the hardest target. Build it first.
- **Do not let a purity metric veto an imputation** — purity is the right question for a bound
  and the wrong one for a conditional mean.

## Ensemble across representations, not seeds

Learners sharing one representation share an information ceiling; averaging them buys almost
nothing. Different *input representation* is what decorrelates.

- **Get two genuinely different representations, one of them 3D.** A 3D-native model that
  consumes conformers is not the same as 3D descriptors appended to a 2D backbone — the
  appended form has measured null. Average predictions over several conformers per compound.
- **Fine-tune the foundation model multi-task.** Frozen embeddings lose to fine-tuning in every
  evaluation; multi-task on related public data beats challenge data alone.
- **Select members on decorrelation, not solo score.** A member that loses standalone can be
  the best thing in the pool. Caruana greedy selection over a candidate pool is the cheap way.
  Weak-and-similar members do hurt.
- **Cheap families rank behind the second representation**, not beside it. Once a 3D model is
  in, the extra families contribute less and per-target.
- **Let the assay decide whether targets share a model** — a target read on a different
  instrument has a different error structure and wants its own blend.

## Auxiliary tasks from unrelated endpoints

Aux heads from endpoints unrelated to the target assay shape the shared representation. Top
finishers routinely use them: LogD, pKa, liver microsome clearance, efflux ratio, solubility,
plasma protein binding — all available publicly. Cheap to try and a default, not an experiment.

## Spend the submission budget

Every published report used the live leaderboard as its model-selection instrument. If the
board permits overwrites, it is the most reliable signal available and a budget exists to be
spent — hoarding submissions is the placement rabbit hole wearing a different hat.

**But the board's scored half is not the test set.** Use it for ranking decisions, which are
model properties and cross intact; distrust it for placement, which overfits the easier half.
Check whether the final score uses hidden compounds only — where it does, the board is a pure
probe and nothing fitted to it is graded.

**Log every submission's per-target score at submit time.** A fixed scored subset makes that
history a clean A/B on identical compounds, and it is the only way a late regression is visible.

## Order of operations

1. Characterize the test set: size, how it was drawn, similarity to training, label range.
2. Build the matching validation split, scored over that range. Nothing is measured before this.
3. Read the metric for asymmetry; note the shift to apply at the end.
4. Audit the label-generating process; impute point labels and train the screen classifier.
5. Add unrelated-endpoint aux tasks.
6. Get two representations working, one 3D, fine-tuned multi-task.
7. Blend by decorrelation, weights fitted out of fold, Caruana greedy.
8. Spend submissions to confirm the blend on the real board.
9. Placement and the metric shift, last. Half a day.

Measured negative results, each cheap to re-derive and expensive to forget: bounded loss on
imputed bounds, MVE/NLL uncertainty heads, kNN residual correction, frozen foundation
embeddings, out-of-fold-fitted ensemble weights, confidence weighting, structure-based
co-folding, and 3D conformer features appended to a 2D model. Check this list before re-running
any of them.
