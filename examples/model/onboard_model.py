"""Onboard a model both interactively and programmatically."""

import logging

# SageWorks Imports
from sageworks.api.model import Model, ModelType

# Setup logging
log = logging.getLogger("sageworks")

# Grab one of the test models
test_model = Model("abalone-regression")

# Onboard the model interactively
test_model.onboard()

# Onboard the model programmatically
model_type = ModelType.REGRESSOR
target_column = "class_number_of_rings"
feature_list = ["length", "diameter", "height", "whole_weight", "shucked_weight", "viscera_weight", "shell_weight"]
endpoints = ["abalone-regression-end"]
owner = "sageworks"
test_model.onboard_with_args(model_type, target_column, feature_list, endpoints, owner)
