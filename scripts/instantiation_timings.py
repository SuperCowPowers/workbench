from sageworks.api.cached_model import CachedModel

# Instantiate the Model object
for _ in range(20):
    model = CachedModel("abalone-regression")
