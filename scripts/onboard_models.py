"""Script that loops through all models and checks if they are ready"""

import logging

# SageWorks Imports
from sageworks.web_views.artifacts_text_view import ArtifactsTextView
from sageworks.core.artifacts.model_core import ModelCore

# Setup logging
log = logging.getLogger("sageworks")

# Create a instance of the ArtifactsTextView
artifacts_text_view = ArtifactsTextView()

# Get all the models
models = artifacts_text_view.models_summary()
for model_name in models["Model Group"]:
    m = ModelCore(model_name)
    if m.ready():
        log.debug(f"Model {model_name} is ready!")
    else:
        log.important(f"Model {model_name} is not ready...Calling onboard.... ")
        m.onboard()
