from sageworks.api import Model

# Instantiate the Model object
for _ in range(20):
    model = Model("abalone-regression", skip_health_check=True)
