# OpenADMET Challenge
This challenge is a community-driven initiative to benchmark predictive models for ADMET properties in drug discovery, hosted by OpenADMET in collaboration with ExpansionRx.

- Huggingface Space: <https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge>

## ADMET Properties
The challenge covers 9 key ADMET properties:

| Property | Description | Units |
|----------|-------------|-------|
| LogD | Lipophilicity (octanol-water distribution) | log units |
| KSOL | Kinetic solubility | µM |
| HLM CLint | Human liver microsome intrinsic clearance | µL/min/mg |
| MLM CLint | Mouse liver microsome intrinsic clearance | µL/min/mg |
| Caco-2 Papp A>B | Caco-2 permeability (apical to basolateral) | cm/s |
| Caco-2 Efflux | Caco-2 efflux ratio | ratio |
| MPPB | Mouse plasma protein binding | % bound |
| MBPB | Mouse brain plasma binding | % bound |
| MGMB | Mouse gut microbiome binding | % bound |

## Our Approach
We train 5 different model types for each ADMET endpoint:

1. **XGBoost** - Gradient boosted trees on RDKit molecular descriptors
2. **PyTorch Tabular** - Neural network on RDKit molecular descriptors
3. **ChemProp** - Message Passing Neural Network (MPNN) on molecular graphs
4. **ChemProp Hybrid** - MPNN + RDKit descriptors combined
5. **ChemProp Multi-Task** - Single MPNN predicting all 9 endpoints simultaneously

### ChemProp Configuration
For our chemprop setup we use R-applet's GitHub as a reference:
[ADMET Challenge 2025 reference](https://github.com/R-applet/ADMET_Challenge_2025). So all credit for the good chemprop setup goes to R-applet (<https://github.com/R-applet>)

| Parameter | Value |
|-----------|-------|
| Message passing depth | 6 |
| Hidden dimension | 700 |
| FFN hidden dimension | 2000 |
| FFN layers | 2 |
| Dropout | 0.25 |
| Max epochs | 400 |
| Early stopping patience | 40 |
| Batch size | 16 |

For multi-task models, we use **dynamic task weights** computed as inverse sample counts:
```
weight[task] = (1 / sample_count[task]) / min(1 / sample_counts)
```
This gives higher weight to targets with fewer training samples.

## Meta-Model Ensemble

Our final submission uses **inverse-variance weighted averaging** across all 5 model types:

1. Each model produces predictions with uncertainty estimates (prediction_std)
2. For each molecule, we weight each model's prediction by `1 / variance` where `variance = std²`
3. The weighted average is computed in log-space (before inverse transform)
4. Final predictions are transformed back to original scale

This approach:
- Gives more weight to confident predictions
- Naturally handles model disagreement
- Produces robust predictions across diverse chemical space

### Running Inference

```python
from run_inference import run_meta_model_inference

# Generate submission with 5-model ensemble
run_meta_model_inference("submission_meta.csv")
```

## References

- **Chemprop**: <https://github.com/chemprop/chemprop>
- **R-applet ADMET Challenge**: <https://github.com/R-applet/ADMET_Challenge_2025>
- **Workbench**: <https://github.com/SuperCowPowers/workbench>

### Contributions
If you'd like to contribute to the Workbench project, you're more than welcome. All contributions will fall under the existing project [license](https://github.com/SuperCowPowers/workbench/blob/main/LICENSE). If you are interested in contributing or have questions please feel free to contact us at [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com).

<img align="right" src="../docs/images/scp.png" width="180">

® Amazon Web Services, AWS, the Powered by AWS logo, are trademarks of Amazon.com, Inc. or its affiliates
