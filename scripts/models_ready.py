"""Script that loops through all models and checks if they are ready"""
import logging
import time

# SageWorks Imports
from sageworks.views.artifacts_text_view import ArtifactsTextView
from sageworks.artifacts.models.model import Model

# Setup logging
log = logging.getLogger("sageworks")

# Create a instance of the ArtifactsTextView
artifacts_text_view = ArtifactsTextView()

# Get all the models
models = artifacts_text_view.models_summary()
for model_name in models["Model Group"]:
    m = Model(model_name)
    if m.ready():
        log.debug(f"Model {model_name} is ready!")
    else:
        log.important(f"Model {model_name} is not ready...Calling make_ready.... ")
        m.make_ready()
