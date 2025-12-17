import logging
from workbench.api import Meta

log = logging.getLogger("workbench")

# List all the models in AWS
meta = Meta()
models = meta.models()
log.info(f"Found {len(models)} models in AWS")
log.info(models)
