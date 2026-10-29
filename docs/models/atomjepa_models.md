# Atom-JEPA Models

[Atom-JEPA](https://arxiv.org/abs/2610.08400) is a 3D molecular encoder: an EquiformerV3 (39M parameters) pretrained by latent prediction on about 19 million molecules. Workbench fine-tunes it per dataset as a regression framework, with the same training, held-out scoring, and confidence modeling as the other frameworks.

Atom-JEPA reads 3D structure. Workbench generates the conformers from SMILES (RDKit ETKDGv3 + MMFF94), so a model needs only a `smiles` column.

## Creating an Atom-JEPA Model

```python
from workbench.api import FeatureSet, ModelType, ModelFramework

fs = FeatureSet("open_admet_mppb")

model = fs.to_model(
    name="mppb-reg-atomjepa",
    model_type=ModelType.UQ_REGRESSOR,
    model_framework=ModelFramework.ATOMJEPA,
    target_column="mppb",
    feature_list=["smiles"],
    description="Fine-tuned Atom-JEPA for mouse plasma protein binding",
    tags=["atomjepa", "mppb"],
)

# Out-of-fold predictions and metrics from training
model.get_inference_predictions("training")
```

Any other numeric features in `feature_list` (an assay readout, a descriptor) enter the readout alongside the encoder's output.

## When to Use Atom-JEPA

Atom-JEPA is strongest on physicochemical and binding endpoints, and it is not a universal upgrade. Measured in Workbench:

| Assay | Data | Metric | Atom-JEPA | Comparison |
|-------|------|--------|-----------|------------|
| MPPB (ExpansionRx) | 1,302 compounds, 5-fold scaffold out-of-fold | MAE | **0.165** | Chemprop 0.175 (Δ −0.010, 95% CI [−0.018, −0.002]) |
| LogD (ExpansionRx) | 5,039 compounds, 5-fold scaffold out-of-fold | MAE | **0.216** | Chemprop 0.239 (Δ −0.024, 95% CI [−0.030, −0.017]); HPO-tuned Chemprop 0.223 |
| PXR (OpenADMET) | 253 held-out compounds, one fold model, + assay readouts | RAE | 0.582 | Chemprop + readouts 0.569, **TabICL + Monroe + readouts 0.535** |
| CYP3A4 (OpenADMET) | 2,335 compounds, 5-fold out-of-fold, + assay readouts | Spearman | 0.800 | **TabICL + Monroe + readouts 0.821** |

An independent evaluation of the authors' own code ([Model Validation Central](https://patwalters.github.io/model-validation-central/reports/atom-jepa.html), multi-task, 5×5 cross-validation on ExpansionRx and Biogen ADME) finds the same profile:

- Fine-tuned Atom-JEPA beats fine-tuned CheMeleon on 38 of 45 endpoint–metric combinations.
- Against Monroe + TabPFN 3 it leads on ExpansionRx and trails on Biogen.
- Its best endpoints are LogD, LogS, HLM clearance, and plasma and brain protein binding; its weakest are Caco-2 permeability and MLM clearance.

In practice:

- **Try Atom-JEPA** for lipophilicity, solubility, microsomal clearance, and protein or tissue binding.
- **Prefer [TabICL](tabicl_models.md) over the [Monroe embedding](../blogs/monroe_tabicl.md)** for permeability and for target-activity assays like PXR and CYP inhibition.
- **Fine-tune it.** The frozen encoder used as an embedding is weak in both evaluations.
- **Budget GPU hours.** A Chemprop model trains in minutes; an Atom-JEPA model takes hours.

## How an Atom-JEPA Model Is Built

| Step | What happens |
|------|--------------|
| **Conformers** | Each molecule gets `n_conformers` seeded conformers. Training draws one at random per epoch; predictions average over all of them. Molecules RDKit cannot embed are dropped from training. |
| **Fine-tuning** | Per fold: the encoder is frozen for the first epochs while the readout trains, then the whole model trains with warmup and cosine decay. An exponential moving average of the weights is what gets validated, picked and saved. |
| **Ensemble** | The fold models are the served model: the prediction is their mean, `prediction_std` their spread. |
| **Uncertainty** | The out-of-fold spread feeds Workbench's confidence model, with neighbors found by structure (fingerprint proximity). |

Training runs on a single GPU. A fold of about 1,000 molecules takes roughly 20–30 minutes, so a 5-fold model on a few thousand molecules takes a few hours.

## Hyperparameters

| Hyperparameter | Default | What it does |
|----------------|---------|--------------|
| `n_folds` | 5 | Fold models in the ensemble (1 = one train/validation split, with no spread for uncertainty) |
| `epochs` | 50 | Training epochs per fold; the best epoch is kept |
| `batch_size` | 32 | Molecules per training step |
| `encoder_lr` / `head_lr` | 5e-5 / 5e-4 | Learning rates for the pretrained encoder and the new readout |
| `freeze_encoder_epochs` | 10 | Epochs the encoder stays frozen while the readout trains |
| `warmup_epochs` | 10 | Linear warmup length; the encoder re-warms over the same length after unfreezing |
| `ema_decay` | 0.99 | Weight averaging; validation and prediction use the averaged weights |
| `n_conformers` | 10 | Conformers per molecule |
| `selection_conformers` | 2 | Conformers per molecule behind each epoch's validation score, which picks the best epoch |
| `split_strategy` | `"scaffold"` | `"scaffold"`, `"butina"`, or `"random"` |
| `uq_version` | `"v1"` | Primary confidence model (`"v0"`, `"v1"`, `"v2"`) |

The defaults are the authors' ADMET recipe.

## Endpoints

Atom-JEPA endpoints run on a GPU. A prediction is the fold ensemble over every conformer of every molecule, which is past what a serverless endpoint can hold or answer within its 60-second limit, so `to_endpoint()` refuses serverless for these models.

```python
# Async: scales to zero when idle, so an idle endpoint costs nothing
endpoint = model.to_endpoint(async_endpoint=True)

# Real-time: always on, for low-latency scoring
endpoint = model.to_endpoint(serverless=False)
```

Both run Workbench's GPU inference image. GPU capacity is often short, so rather than one instance type they list instance pools, and SageMaker provisions from the first with capacity, at creation and on every scale-up:

| Priority | Instance | Hardware |
|---|---|---|
| 1 | `ml.g6.xlarge` | NVIDIA L4 |
| 2 | `ml.g5.xlarge` | NVIDIA A10G |
| 3 | `ml.g6.2xlarge` | NVIDIA L4 |
| 4 | `ml.g5.2xlarge` | NVIDIA A10G |
| 5 | `ml.c7i.4xlarge` | CPU, async endpoints only |

The CPU pool is a fallback: the same image serves there at roughly 30 seconds per molecule, against under a second on a GPU. Real-time endpoints stop at the GPU pools, since a real-time request must answer within 60 seconds. Pass `instance=` to deploy a single type instead. An async endpoint stays on one instance unless you pass `max_instances=`.

When every pool is out, SageMaker keeps trying for up to an hour. If it still has no instance, `to_endpoint()` deletes the failed endpoint and raises `EndpointCapacityError` (from `workbench.core.artifacts.endpoint_core`); the model is intact, so deploy it again later. An async endpoint scaling up from zero raises the same error after the same hour.

Endpoint instances are quota-limited per instance type and region; check `<instance type> for endpoint usage` for each pool type in the Service Quotas console. To run several Atom-JEPA endpoints at once, or scale one past a single instance, request an increase there.

A model's training captures need no endpoint: `get_inference_predictions("training")` gives the out-of-fold rows and `get_inference_predictions("validation")` the rows held out with `validation_ids`.

## Pretrained Weights

The encoder weights (Hugging Face `atom-jepa/atom-jepa`, revision `8f7514f`, CC BY 4.0) are published in Workbench's foundation-model registry, so training jobs fetch them from S3 and never depend on Hugging Face.

## Running Locally

Local training works through `workbench.local` with the `modeling` extra. It runs on CPU (Apple's MPS trains this model far slower than CPU), which suits small smoke tests rather than full models.

## Limits

- Regression only, one target per model.
- 3D conformers come from RDKit; a molecule it cannot embed is dropped from training and predicts `NaN`.

## Questions?

<img align="right" src="../../images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)

## References

- **Paper:** [Atom-JEPA: Joint-Embedding Predictive Architecture for 3D Atomistic Systems](https://arxiv.org/abs/2610.08400) (arXiv 2610.08400)
- **Code:** [github.com/khelverskovp/atom-jepa](https://github.com/khelverskovp/atom-jepa) (MIT); `pip install atom-jepa`
- **Weights:** [huggingface.co/atom-jepa/atom-jepa](https://huggingface.co/atom-jepa/atom-jepa) (CC BY 4.0)
- **Independent evaluation:** Pat Walters, [Model Validation Central: Fine-tuning in 3D (Atom-JEPA)](https://patwalters.github.io/model-validation-central/reports/atom-jepa.html), with the study code in [model-validation-central/studies/atom-jepa](https://github.com/PatWalters/model-validation-central/tree/main/studies/atom-jepa)
