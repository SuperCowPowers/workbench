from sageworks.api.model import Model

# Grab the abalone-regression Model
model = Model("abalone-regression")

# Perform a health check on the model
# Note: The health_check() method returns 'issues' if there are any
#       problems, so if there are no issues, the model is healthy
health_issues = model.health_check()
if not health_issues:
    print("Model is Healthy")
else:
    print("Model has issues")
    print(health_issues)

# Get the model metrics and regression predictions
print(model.get_inference_metrics())
print(model.get_inference_predictions())
