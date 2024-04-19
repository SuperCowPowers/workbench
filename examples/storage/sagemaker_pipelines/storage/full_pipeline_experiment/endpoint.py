from sageworks.api.model import Model

# Create the abalone_regression Endpoint
model = Model("sp-abalone-regression")
model.to_endpoint(name="sp-abalone-regression-end", tags=["abalone", "regression"])
