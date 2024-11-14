import time
from sageworks.cached.cached_meta import CachedMeta
from sageworks.cached.cached_model import CachedModel
from sageworks.api import Model

# Instantiate all the Model objects
models = CachedMeta().models()["Model Group"].tolist()
models = models[:10]  # Ten for testing

# Normal Models
print("Model()")
for model_name in models:
    start_time = time.time()
    model = Model(model_name)
    print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")

print("CachedModel()")
for model_name in models:
    start_time = time.time()
    model = CachedModel(model_name)
    print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")

# Sleep to let's Cache Threads spin down
print("Cache spindown...")
time.sleep(5)
