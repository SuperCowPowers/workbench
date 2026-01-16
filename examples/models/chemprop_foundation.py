"""ChemProp Foundation Model Examples

This script demonstrates how to use pretrained foundation models (like CheMeleon)
with ChemProp in Workbench. Foundation models provide pretrained MPNN weights that
can significantly improve performance, especially on smaller datasets.

CheMeleon is a descriptor-based foundation model pretrained on 1M PubChem molecules
to predict Mordred molecular descriptors. This gives the MPNN a strong prior for
molecular representation learning.

References:
- CheMeleon: https://github.com/JacksonBurns/chemeleon
- Paper: https://arxiv.org/abs/2506.15792
"""

from workbench.api import FeatureSet, Model, ModelType, ModelFramework, Endpoint

# Recreate Flag in case you want to recreate the artifacts
recreate = False

# =============================================================================
# Single-Target CheMeleon Foundation Model
# =============================================================================
# Basic usage: Load CheMeleon pretrained weights and fine-tune on your dataset.
# The MPNN weights are initialized from CheMeleon, and a new FFN head is trained
# for your specific task.

if recreate or not Model("chemeleon-logd").exists():
    feature_set = FeatureSet("open_admet_all")
    m = feature_set.to_model(
        name="chemeleon-logd",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",
        feature_list=["smiles"],
        description="CheMeleon foundation model fine-tuned for LogD prediction",
        tags=["chemprop", "chemeleon", "foundation", "open_admet"],
        hyperparameters={
            "from_foundation": "CheMeleon",  # Load pretrained MPNN weights
            "n_folds": 5,
            "max_epochs": 100,  # Fewer epochs needed with pretrained weights
        },
    )
    m.set_owner("BW")

# Create an Endpoint
if recreate or not Endpoint("chemeleon-logd").exists():
    m = Model("chemeleon-logd")
    end = m.to_endpoint(tags=["chemprop", "chemeleon", "foundation", "open_admet"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()


# =============================================================================
# Foundation Model with MPNN Freezing (Two-Phase Training)
# =============================================================================
# Advanced usage: Freeze the MPNN for initial epochs to stabilize the FFN,
# then unfreeze and fine-tune the entire model. This approach is recommended
# when you have limited data or want more stable training.

if recreate or not Model("chemeleon-logd-frozen").exists():
    feature_set = FeatureSet("open_admet_all")
    m = feature_set.to_model(
        name="chemeleon-logd-frozen",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",
        feature_list=["smiles"],
        description="CheMeleon with frozen MPNN warmup for stable fine-tuning",
        tags=["chemprop", "chemeleon", "foundation", "frozen", "open_admet"],
        hyperparameters={
            "from_foundation": "CheMeleon",
            "freeze_mpnn_epochs": 10,  # Phase 1: Train FFN only for 10 epochs
            "max_epochs": 100,  # Total epochs (10 frozen + 90 fine-tuning)
            "n_folds": 5,
        },
    )
    m.set_owner("BW")

# Create an Endpoint
if recreate or not Endpoint("chemeleon-logd-frozen").exists():
    m = Model("chemeleon-logd-frozen")
    end = m.to_endpoint(tags=["chemprop", "chemeleon", "foundation", "frozen", "open_admet"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()


# =============================================================================
# Multi-Task Foundation Model
# =============================================================================
# Foundation models work with multi-task learning too. This combines CheMeleon's
# pretrained representations with multi-task regression across 9 ADMET endpoints.

ADMET_TARGETS = [
    "logd",
    "ksol",
    "hlm_clint",
    "mlm_clint",
    "caco_2_papp_a_b",
    "caco_2_efflux",
    "mppb",
    "mbpb",
    "mgmb",
]

if recreate or not Model("chemeleon-mt").exists():
    feature_set = FeatureSet("open_admet_all")
    m = feature_set.to_model(
        name="chemeleon-mt",
        model_type=ModelType.REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column=ADMET_TARGETS,  # Multi-task: 9 ADMET endpoints
        feature_list=["smiles"],
        description="CheMeleon foundation model for multi-task ADMET prediction",
        tags=["chemprop", "chemeleon", "foundation", "multitask", "open_admet"],
        hyperparameters={
            "from_foundation": "CheMeleon",
            "freeze_mpnn_epochs": 10,
            "max_epochs": 100,
            "n_folds": 5,
        },
    )
    m.set_owner("BW")

# Create an Endpoint
if recreate or not Endpoint("chemeleon-mt").exists():
    m = Model("chemeleon-mt")
    end = m.to_endpoint(tags=["chemprop", "chemeleon", "foundation", "multitask", "open_admet"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()


# =============================================================================
# Hybrid Foundation Model (CheMeleon + Molecular Descriptors)
# =============================================================================
# Combine CheMeleon's pretrained MPNN with additional molecular descriptors.
# The extra features are concatenated with the MPNN output before the FFN,
# providing complementary information to the learned representations.

TOP_LOGD_SHAP_FEATURES = [
    "mollogp",
    "fr_halogen",
    "peoe_vsa6",
    "nbase",
    "peoe_vsa7",
    "peoe_vsa9",
    "peoe_vsa1",
    "mi",
    "bcut2d_mrlow",
    "slogp_vsa1",
]

if recreate or not Model("chemeleon-logd-hybrid").exists():
    feature_set = FeatureSet("open_admet_logd")

    # Hybrid mode: SMILES (for CheMeleon MPNN) + molecular descriptors
    hybrid_features = ["smiles"] + TOP_LOGD_SHAP_FEATURES

    m = feature_set.to_model(
        name="chemeleon-logd-hybrid",
        model_type=ModelType.UQ_REGRESSOR,
        model_framework=ModelFramework.CHEMPROP,
        target_column="logd",
        feature_list=hybrid_features,
        description="CheMeleon hybrid model with MPNN + top SHAP molecular descriptors",
        tags=["chemprop", "chemeleon", "foundation", "hybrid", "open_admet"],
        hyperparameters={
            "from_foundation": "CheMeleon",
            "freeze_mpnn_epochs": 10,
            "max_epochs": 100,
            "n_folds": 5,
        },
    )
    m.set_owner("BW")

# Create an Endpoint
if recreate or not Endpoint("chemeleon-logd-hybrid").exists():
    m = Model("chemeleon-logd-hybrid")
    end = m.to_endpoint(tags=["chemprop", "chemeleon", "foundation", "hybrid", "open_admet"])
    end.set_owner("BW")
    end.auto_inference()
    end.cross_fold_inference()


# =============================================================================
# Custom Foundation Model (Your Own Pretrained Weights)
# =============================================================================
# You can also use your own pretrained Chemprop model as a foundation.
# This is useful if you have domain-specific pretrained weights.
#
# Example (uncomment to use):
#
# if recreate or not Model("my-custom-foundation-model").exists():
#     feature_set = FeatureSet("my_dataset")
#     m = feature_set.to_model(
#         name="my-custom-foundation-model",
#         model_type=ModelType.UQ_REGRESSOR,
#         model_framework=ModelFramework.CHEMPROP,
#         target_column="my_target",
#         feature_list=["smiles"],
#         description="Fine-tuned from custom pretrained model",
#         hyperparameters={
#             "from_foundation": "/path/to/my_pretrained_model.pt",  # Path to .pt file
#             "freeze_mpnn_epochs": 5,
#             "max_epochs": 50,
#         },
#     )
