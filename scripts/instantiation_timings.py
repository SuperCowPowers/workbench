import time
from sageworks.cached.cached_meta import CachedMeta
from sageworks.cached.cached_model import CachedModel
from sageworks.api import Model
from sageworks.utils.sageworks_cache import disable_refresh

# Instantiate all the Model objects
models = CachedMeta().models()["Model Group"].tolist()
# models = models[:10]  # Ten for testing

TEST_NORMAL = False
TEST_CACHE_REFRESH = False
TEST_CACHE_NO_REFRESH = True

# Normal Models
if TEST_NORMAL:
    print("Model()")
    for model_name in models:
        start_time = time.time()
        model = Model(model_name)
        print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")


# Cached Models (Refresh Enabled)
if TEST_CACHE_REFRESH:
    print("CachedModel()")
    for model_name in models:
        start_time = time.time()
        model = CachedModel(model_name)
        print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")

# Cached Models (Refresh Disabled)
if TEST_CACHE_NO_REFRESH:
    no_refresh_start_time = time.time()
    with disable_refresh():
        print("CachedModel() No Refresh")
        for model_name in models:
            start_time = time.time()
            model = CachedModel(model_name)
            print(f"{model_name} instantiation Time: {time.time() - start_time:.3f} seconds")
    print(f"Total Time {len(models)} models: {time.time() - no_refresh_start_time:.3f} seconds")


# Sleep to let's Cache Threads spin down
print("Cache spin down...")
time.sleep(5)
