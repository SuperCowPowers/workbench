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

## How an Atom-JEPA Model Is Built

| Step | What happens |
|------|--------------|
| **Conformers** | Each molecule gets `n_conformers` seeded conformers. Training draws one at random per epoch; predictions average over all of them. Molecules RDKit cannot embed are dropped from training. |
| **Fine-tuning** | Per fold: the encoder is frozen for the first epochs while the readout trains, then the whole model trains with warmup and cosine decay. An exponential moving average of the weights is what gets validated, picked and saved. |
| **Ensemble** | The fold models are the served model: the prediction is their mean, `prediction_std` their spread. |
| **Uncertainty** | The out-of-fold spread feeds Workbench's confidence model, with neighbors found by structure (fingerprint proximity). |

Training runs on a single GPU. A fold of about 1,000 molecules takes roughly 15–20 minutes, so a 5-fold model on a few thousand molecules takes a few hours.

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

Serving is not yet supported: the inference images don't carry the Atom-JEPA dependencies. Score a model from its training captures: `get_inference_predictions("training")` for the out-of-fold rows and `get_inference_predictions("validation")` for rows held out with `validation_ids`.

## Pretrained Weights

The encoder weights (Hugging Face `atom-jepa/atom-jepa`, revision `8f7514f`, CC BY 4.0) are published in Workbench's foundation-model registry, so training jobs fetch them from S3 and never depend on Hugging Face.

## Running Locally

Local training works through `workbench.local` with the `modeling` extra. It runs on CPU (Apple's MPS trains this model far slower than CPU), which suits small smoke tests rather than full models.

## Limits

- Regression only, one target per model.
- 3D conformers come from RDKit; a molecule it cannot embed is dropped from training and predicts `NaN`.

## References

- Atom-JEPA: [arXiv 2610.08400](https://arxiv.org/abs/2610.08400), [github.com/khelverskovp/atom-jepa](https://github.com/khelverskovp/atom-jepa) (MIT)
- Independent evaluation: [Model Validation Central, Atom-JEPA report](https://patwalters.github.io/model-validation-central/reports/atom-jepa.html)
