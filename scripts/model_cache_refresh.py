import time
from sageworks.cached.cached_meta import CachedMeta
from sageworks.cached.cached_model import CachedModel

# Instantiate all the Model objects
models = CachedMeta().models()["Model Group"].tolist()
for model_name in models:
    model = CachedModel(model_name)
    time.sleep(1)  # Don't spam AWS

# Let the Cache Threads return the results
print("Cache thread spin down...")
CachedModel._shutdown()
