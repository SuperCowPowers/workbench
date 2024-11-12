import time
from sageworks.cached import CachedMeta
from sageworks.api.cached_model import CachedModel

# Instantiate all the Model objects
models = CachedMeta().models()
for model_name in models["Model Group"]:
    start_time = time.time()
    model = CachedModel(model_name)
    print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")

# Sleep to let's Cache Threads spin down
print("Cache spindown...")
time.sleep(5)
