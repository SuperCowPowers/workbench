import time
from workbench.cached.cached_meta import CachedMeta
from workbench.cached.cached_model import CachedModel
from workbench.api import Model
from workbench.utils.workbench_cache import disable_refresh

# Instantiate all the Model objects
models = CachedMeta().models()["Model Group"].tolist()
# models = models[:10]  # Ten for testing

TEST_NORMAL = True
TEST_CACHE_REFRESH = True
TEST_CACHE_NO_REFRESH = False

# Normal Models
if TEST_NORMAL:
    print("\nModel()\n")
    for model_name in models:
        start_time = time.time()
        model = Model(model_name)
        print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")


# Cached Models (Refresh Enabled)
if TEST_CACHE_REFRESH:
    print("\n\nCachedModel()\n")
    for model_name in models:
        start_time = time.time()
        model = CachedModel(model_name)
        print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")

# Cached Models (Refresh Disabled)
if TEST_CACHE_NO_REFRESH:
    no_refresh_start_time = time.time()
    with disable_refresh():
        print("\n\nCachedModel() No Refresh\n")
        for model_name in models:
            start_time = time.time()
            model = CachedModel(model_name)
            print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")
    print(f"Total Time {len(models)} models: {time.time() - no_refresh_start_time:.3f} seconds")


# Sleep to let's Cache Threads spin down
print("Cache spin down...")
time.sleep(5)
