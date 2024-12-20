"""Script that loops through all models and checks if they are ready"""

import logging

# Workbench Imports
from workbench.api import Meta, Model

# Setup logging
log = logging.getLogger("workbench")

# Get all the models
models = Meta().models()
for model_name in models["Model Group"]:
    m = Model(model_name)
    if m.ready():
        log.important(f"Model {model_name} is ready!")
    else:
        log.important(f"Model {model_name} is not ready...Calling onboard.... ")
        m.onboard()
