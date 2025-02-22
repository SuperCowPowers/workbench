from workbench.api.model import Model

# Grab the abalone regression Model
model = Model("abalone-regression")

# By default, an Endpoint is serverless, but you can
# make it 'realtime' by setting serverless=False
model.to_endpoint(name="abalone-regression", tags=["abalone", "regression"], serverless=True)
