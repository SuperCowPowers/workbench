# ChemProp and Workbench: WORLD DOMINATION!

<figure style="float: right; width: 280px;">
<img alt="chemprop_mascot" src="images/chemprop_robot.svg">
<figcaption>ChemBot says: "Your molecules will OBEY!"</figcaption>
</figure>

**FOOLISH MORTALS!** Did you think your puny XGBoost models could compete with the MAGNIFICENT POWER of neural networks? The [OpenADMET Leaderboard](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge) has spoken, and **ChemProp is Dominating!**

## The MIGHTY Neural Network Arsenal

Workbench doesn't just give you ONE way to crush your ADMET predictions—it gives you an entire DOOM ARMY of model options!

| Model Type | Description | Power Level |
|------------|-------------|-------------|
| **XGBoost** | Gradient boosted trees on RDKit molecular descriptors | STRONG! |
| **PyTorch** | Neural network on RDKit molecular descriptors | IMPRESSIVE! |
| **ChemProp** | Message Passing Neural Network (MPNN) on molecular graphs | DEVASTATINGLY POWERFUL! |
| **ChemProp Hybrid** | MPNN + Top RDKit descriptors combined | TERRIFYINGLY EFFECTIVE! |
| **ChemProp Multi-Task** | Single MPNN predicting ALL endpoints at once | ULTIMATE DESTRUCTION! |

## Why ChemProp? WHY?!

<figure style="float: left; width: 240px; margin-right: 20px;">
<img alt="pytorch_mascot" src="images/pytorch_brain.svg">
<figcaption>PyTorch Pete: "Brains? I've got LAYERS of them!"</figcaption>
</figure>

*Ahem.* Let us explain with SCIENCE:

Traditional models look at molecules like a list of boring numbers (descriptors). ChemProp looks at the ACTUAL molecular graph—atoms as nodes, bonds as edges—and uses Message Passing Neural Networks to learn from the molecular STRUCTURE itself!

The results from OpenADMET Challenge? **ChemProp Single-Task models consistently DOMINATED** across all 9 ADMET endpoints:

- LogD (Lipophilicity)
- KSOL (Kinetic Solubility)
- HLM/MLM CLint (Liver Clearance)
- Caco-2 Permeability & Efflux
- Plasma & Brain Protein Binding

## Deploy to AWS® in a SNAP!

<figure style="float: right; width: 260px;">
<img alt="workbench_mascot" src="images/workbench_hero.svg">
<figcaption>Commander Workbench: "I deploy endpoints before breakfast!"</figcaption>
</figure>

This is where it gets **DELICIOUSLY EASY**. While lesser frameworks make you wrestle with Docker containers, SageMaker configurations, and IAM policies... Workbench just DOES IT.

```python
from workbench.api import DataSource, FeatureSet, ModelType, ModelFramework

# BEHOLD! The power of simplicity!
ds = DataSource("my_molecules.csv", name="admet_data")
fs = ds.to_features("admet_features", id_column="mol_id")

# Create a ChemProp model with ONE COMMAND
model = FeatureSet("admet_features").to_model(
    name="my-chemprop-model",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,  # THE MAGIC!
    target_column="logd",
    feature_list=["smiles"],
    description="ChemProp model for LogD prediction",
)

# Deploy to AWS® Endpoint - WORLD DOMINATION ACHIEVED!
endpoint = model.to_endpoint()
```

**That's it.** No 47-page AWS documentation. No crying at 3 AM. Just RESULTS.

## Multi-Task: Because ONE Prediction is for AMATEURS

Why train 9 models when you can train ONE that predicts EVERYTHING?

```python
# ALL 9 ADMET ENDPOINTS IN ONE GLORIOUS MODEL!
ADMET_TARGETS = [
    'logd', 'ksol', 'hlm_clint', 'mlm_clint',
    'caco_2_papp_a_b', 'caco_2_efflux',
    'mppb', 'mbpb', 'mgmb'
]

model = feature_set.to_model(
    name="admet-multi-task-SUPREME",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.CHEMPROP,
    target_column=ADMET_TARGETS,  # List = Multi-task!
    feature_list=["smiles"],
    tags=["chemprop", "multitask", "world_domination"],
)
```

## Confidence Estimates: Know Your DOOM Level

<figure style="float: left; width: 220px; margin-right: 20px;">
<img alt="confidence_mascot" src="images/confidence_cat.svg">
<figcaption>Confidence Cat: "I'm 95% sure this molecule is garbage."</figcaption>
</figure>

Every Workbench model comes with **built-in uncertainty quantification**. Don't just get a prediction—know HOW SURE you can be about that prediction!

This is CRITICAL for drug discovery because:

- **High confidence** = Trust this prediction, move forward!
- **Low confidence** = This molecule is weird, get more data!

## PyTorch: Sometimes Better

Yes ChemProp is good but for some/assays PyTorch models on RDKit descriptors are still **incredibly effective** and can sometimes outperform ChemProp.:

```python
model = feature_set.to_model(
    name="my-pytorch-model",
    model_type=ModelType.REGRESSOR,
    model_framework=ModelFramework.PYTORCH,  #Go PyTorch!
    target_column="logd",
    feature_list=fs.feature_columns,  # All the RDKit goodness
)
```

## The Leaderboard Speaks!

The [OpenADMET Challenge](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge) pitted models against each other across 9 ADMET properties. The verdict?

**ChemProp models built with Workbench DEFAULT settings** dominated the competition. No hyperparameter wizardry. No secret sauce. Just good architecture deployed the EASY way.

## Get Started NOW!

1. **Install Workbench**: `pip install workbench`
2. **Connect to AWS®**: [Setup Guide](../getting_started/index.md)
3. **Deploy ChemProp**: Use the code above!
4. **ACHIEVE VICTORY!**

---

## Workbench is in Beta!

<figure style="float: right; width: 200px;">
<img alt="broken_robot" src="images/broken_robot.svg">
<figcaption>BetaBot: "I'm still learning... be gentle!"</figcaption>
</figure>

**ATTENTION BRAVE SOULS!** Workbench is currently in **BETA** — which means we're still squashing bugs, polishing features, and occasionally setting things on fire (metaphorically... mostly).

We're looking for **fearless beta testers** to help us make Workbench even MORE powerful! If you want early access to cutting-edge ML deployment tools and don't mind the occasional glitch, **WE WANT YOU!**

**Sign up for the beta program:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)

### Questions? CONFUSED? TERRIFIED? Need Help?

Running into issues? Have questions? Just want to chat about molecules and neural networks? We're here for you:

<img align="right" src="../images/scp.png" width="180">

- **Support Email:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join the Hive Mind](https://discord.gg/WHAJuz8sw8) — get real-time help from the team and community!
- **Dashboard:** [Workbench Dashboard](https://workbench-dashboard.com/)


*Now go forth and CONQUER your ADMET predictions!*

---

