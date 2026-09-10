# PK Data Consistency

> physical-plausibility checks on predicted F, CL, and Vd — catching rows that are individually reasonable but impossible together

A PK panel is a meta-endpoint over three heads: bioavailability, clearance, and volume
of distribution. They arrive in one frame, so every check below is row-wise — no joins.

```python
from workbench.api import MetaEndpoint

ep = MetaEndpoint("admet-pk-panel")
ep.output_columns()                    # which heads this panel returns, before running it
predictions = ep.inference(df)
```

Head names are `<target>_pred` (see the `plotting` guide). Read them off
`output_columns()` rather than hardcoding — the panel's composition is what decides them.

**Units.** These checks assume the ADMET convention: **CL in mL/min/kg**, **Vd in L/kg**,
**F as a percentage**. Those two units do not agree with each other; anything deriving a
rate from both needs `CL * 0.06` to reach L/h/kg first.

## Reference values

Hepatic blood flow `Q_h`, mL/min/kg (Davies & Morris 1993):

| species | Q_h  |
|---------|------|
| mouse   | 90   |
| rat     | 55   |
| monkey  | 44   |
| dog     | 31   |
| human   | 21   |

## 1. F against CL — the coupling check

The one check that needs two heads at once, and the one that catches rows where each
value looks fine alone. A hepatically-cleared compound loses drug to first pass, so
extraction caps how much can reach systemic circulation:

```
F_max = 100 * (1 - CL / Q_h)
```

A row with `F_pred` above that ceiling is inconsistent: the panel is claiming high
clearance and high bioavailability at once.

```python
Q_H = 90  # mouse
ceiling = 100 * (1 - predictions["cl_pred"] / Q_H)
suspect = predictions[predictions["f_pred"] > ceiling]
```

Two things keep this a **flag for review, not a rejection**:

- It is a ceiling, not an estimate. Real `F = F_abs * F_gut * F_hep` sits below it, so
  rows *under* the ceiling are unremarkable rather than confirmed good.
- It assumes hepatic clearance dominates. A renally or biliarily cleared compound can
  legitimately pair high total CL with high F, and will flag here as a false positive.

## 2. Hard bounds

Violations here are physically impossible, not merely unlikely — treat them as defects.

- `F_pred` outside `(0, 100]`. Above 100% cannot happen; at 0 the row carries nothing.
- `CL_pred <= 0`, or above cardiac output. Above `Q_h` is not itself impossible — it
  means clearance is not purely hepatic — but it invalidates check 1 for that row.
- `Vd_pred` below plasma volume (~0.04 L/kg). Nothing distributes into less than plasma.
- `Vd_pred` above ~100 L/kg is suspicious but real for lipophilic bases, which bind
  tissue extensively. Review rather than drop.

## 3. Derived quantities — where CL and Vd interact

Neither head alone says anything about duration; together they fix it.

```python
half_life = 0.693 * predictions["vd_pred"] / (0.06 * predictions["cl_pred"])   # hours
```

A mid-range Vd and a mid-range CL can still imply a 30-second or a two-month half-life.
Flag outside roughly 5 minutes to 1 week and look at the pair — real drugs sit outside
that window (amiodarone's terminal half-life is weeks), so it is a review trigger.

`pk.bateman(volume=..., clearance=...)` is the visual form of this check: an implausible
pair produces a curve that is visibly flat or vertical. See the `plotting` guide.

If the source data carries measured exposure, `AUC = F * dose / CL` is a direct
reproduction test rather than a plausibility one — the strongest check available.

## 4. Modeling artifacts

These are bad *predictions* rather than bad data, and each has a specific cause:

- **Negative CL or Vd.** Both are log-normal; a model trained on linear targets predicts
  into negatives and is dominated by high-clearance outliers. The fix is upstream —
  train on `log10`.
- **A distribution in the wrong place.** CL centered near 20 is mL/min/kg; centered near
  1 it is L/h/kg. Vd centered near 1 is L/kg; near 70 it is total L for a human. A panel
  whose heads were trained on differently-scaled columns produces exactly this.
- **Constant or near-constant predictions** for a head, which means that model failed to
  learn and its column should not feed a downstream calculation at all.

## Working the findings

Report counts and the rows, not a verdict — a flagged row is a question for the user,
who knows whether the chemotype is renally cleared or the assay was run at a dose the
model never saw. Check 1's false-positive mode makes this especially true.

For target-column health on the training data behind any one head — censored values,
duplicates, activity cliffs — see the `data_cleanup` guide.
