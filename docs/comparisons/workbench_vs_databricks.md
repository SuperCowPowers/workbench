!!! tip inline end "Visibility and Control"
    Workbench provides AWS ML Pipeline visibility through the [Workbench Dashboard](../index.md) and control over the creation, modification, and deletion of artifacts through the Python API and [Workbench REPL](../repl/index.md).

# Workbench vs Databricks

Databricks is a mature, comprehensive platform covering data engineering, collaborative
notebooks, and the full ML lifecycle. It's the "Rolls-Royce" of ML platforms — robust,
feature-rich, built for enterprise scale.

Workbench is a go-kart. Lightweight, fast, AWS-native, and focused on one job: making ML
pipeline creation, deployment, and management straightforward.

This page is an honest comparison, including the places Databricks is clearly ahead. If
you're evaluating both, you should know where each one actually wins.

## The short version

| Dimension | Winner | Why |
|---|---|---|
| **Ease of creating a model** | Workbench | Four class abstractions, ~10 lines from raw data to live endpoint |
| **Freedom to build any model** | Databricks | Any framework, arbitrary Python. Workbench supports a curated set |
| **Fleet visibility** | Workbench | Whole-project artifact DAG and model contests in one view |
| **Experiment tracking** | Databricks | MLflow run history and comparison. Workbench has none |
| **Domain fit (ADMET / molecules)** | Workbench | ChemProp variants, uncertainty quantification, scaffold/Butina splits built in |
| **Breadth of platform** | Databricks | Spark, streaming, governance, multi-cloud, BI |

---

## 1. Ease of creating a model

Workbench models the pipeline as four nouns — `DataSource` → `FeatureSet` → `Model` →
`Endpoint` — and each stage produces the next. A complete pipeline from a CSV in S3 to a
deployed endpoint running inference:

```python
from workbench.api import DataSource, FeatureSet, Model, ModelType, Endpoint

ds = DataSource("s3://workbench-public-data/common/abalone.csv")
ds.to_features("abalone_features")

fs = FeatureSet("abalone_features")
fs.to_model(
    name="abalone-regression",
    model_type=ModelType.REGRESSOR,
    target_column="class_number_of_rings",
)

model = Model("abalone-regression")
model.to_endpoint(name="abalone-regression-end")

endpoint = Endpoint("abalone-regression-end")
print(endpoint.test_inference())
```

Note what the code does *not* contain: no estimator instantiation, no train/test split, no
model signature, no metric logging, no serving config. Behind `to_model()` you get a 5-fold
cross-validated ensemble with calibrated uncertainty, a generated training script, and
container selection.

The Databricks equivalent uses MLflow for training and registration, the Feature Engineering
client for feature tables, and the Deployments SDK for serving — three SDKs, plus you write
the modeling code itself (estimator, hyperparameters, split, signature, metrics).

**Fair counterpoint:** Databricks' unbundling is a genuine virtue if your work doesn't fit a
four-stage chain. Workbench's abstraction helps precisely because it is opinionated, and
that cuts both ways.

## 2. Freedom to build the model you want

**Databricks wins this outright.** It imposes essentially no constraint — any framework,
arbitrary Python, and MLflow's `pyfunc` wrapper serves whatever you register.

Workbench supports a curated set of frameworks — scikit-learn, XGBoost, PyTorch, ChemProp,
Transformers, and Meta (composite) models — plus a custom-script escape hatch.

Within that set the tuning surface is deep. ChemProp, for example, exposes architecture
(`hidden_dim`, `depth`, `dropout`, `ffn_hidden_dim` including tapered per-layer lists), the
full Noam learning-rate schedule (`warmup_epochs`, `init_lr`, `max_lr`, `final_lr`),
ensembling (`n_folds`), splitting strategy (`random`, `scaffold`, `butina`), foundation-model
fine-tuning (`from_foundation`, `freeze_mpnn_epochs`), and uncertainty version selection.

Workbench also ships four ChemProp model shapes — Single-Task, Multi-Task, Hybrid
(SMILES + descriptors), and Foundation (CheMeleon) — that compose freely. On Databricks,
ChemProp is available through community guides and the Genesis Workbench accelerator, but
you assemble and maintain the training and serving code yourself.

**Current gaps:** Workbench has no AutoML, and hyperparameter optimization is designed but
not yet shipped. Databricks integrates Optuna and Ray Tune today.

## 3. Visibility

Workbench provides dedicated dashboard pages with drill-downs for DataSources, FeatureSets,
Models, Endpoints, ML Pipelines, and Model Contests.

Two of these have no direct Databricks analog:

- **ML Pipelines** renders the entire project as an artifact DAG — public data, feature sets,
  models, and endpoints together, color-coded by type. Dozens of models and endpoints and
  every dependency between them, in one view.
- **Model Contests** treats every endpoint as an ongoing competition: the champion model,
  ranked challengers with their metrics, and a badge when a challenger is closing in.

Databricks' Unity Catalog lineage is finer-grained (column-level) but graph-shallow — a model
lineage view typically renders the source table and the model version. Its pipeline DAG view
shows *data tables*, not ML artifacts. Seeing an entire model fleet means navigating the
catalog one model at a time.

**Where Databricks wins:** MLflow experiment tracking. A searchable, filterable, chartable
history of every training run with parameters and metrics. Workbench has no equivalent —
metrics live per-model and per-contest, not as a queryable run history.

---

## Where Databricks is ahead

- **Any framework.** Workbench supports a curated set; Databricks supports whatever you write.
- **Experiment tracking.** MLflow run history and comparison. Workbench has none.
- **AutoML and HPO.** Optuna/Ray Tune integration today.
- **Model versioning.** Unity Catalog keeps version history with aliases; Workbench promotion
  is current-only.
- **Traffic splitting.** Percentage-based routing for A/B tests and gradual rollout;
  Workbench promotion is an atomic swap.
- **Governance.** Unity Catalog access control, auditing, and column-level lineage.
- **Scale and breadth.** Spark, Delta Lake, streaming, SQL analytics, BI integration,
  multi-cloud, and a large support organization.

## Where Workbench is ahead

- **Time to a working pipeline.** One import, ~10 statements, raw data to live endpoint.
- **Fleet visibility.** The whole ML artifact graph and every model contest in one view.
- **Dependency resolution by inference.** Workbench derives the full DAG from artifact
  references and propagates staleness forward automatically — change a DataSource and every
  downstream FeatureSet and Model rebuilds in dependency order. Databricks offers table-update
  triggers, but they are explicitly configured per job, capped at 10 watched tables, fire at
  job granularity, and do not infer retraining from catalog lineage.
- **Automatic champion/challenger.** Contests run and publish continuously. On Databricks,
  champion/challenger is a documented convention over registry aliases with comparison logic
  you write.
- **Domain-native modeling.** ChemProp variants, ensemble uncertainty with conformal
  calibration, and scaffold/Butina splits that address the leakage modes that actually matter
  in molecular property prediction. (Point-in-time correctness, a headline feature-store
  feature, is largely inapplicable here — molecular descriptors are deterministic functions of
  structure.)
- **No metering layer.** Your AWS bill plus a license, deployed in your own account.

## Choosing between them

Choose **Databricks** if you need a general-purpose data and ML platform, work across
multiple clouds, need Spark or streaming, have governance and audit requirements, or want
maximum freedom in how models are built.

Choose **Workbench** if you're on AWS, want ML pipelines running quickly with minimal
ceremony, need to see and manage many models at once, and are working in molecular property
prediction where the domain modeling is already built for you.

They are not the same class of vehicle, and that's the point.

## Additional Resources

- Using Workbench for ML Pipelines: [Workbench API Classes](../api_classes/overview.md)

<img align="right" src="../../images/scp.png" width="180">

- Workbench Core Classes: [Core Classes](../core_classes/overview.md)
- ChemProp Models: [ChemProp Model Types](../models/chemprop_models.md)
- Consulting Available: [SuperCowPowers LLC](https://www.supercowpowers.com)
