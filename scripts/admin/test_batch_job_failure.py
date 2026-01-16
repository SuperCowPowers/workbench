from workbench.api import FeatureSet, Model, Endpoint

# Grab a FeatureSet
my_features = FeatureSet("aqsol_features")
model = Model("aqsol-regression")
end = Endpoint("aqsol-regression")

# Now give the endpoint an incorrect method on purpose to test batch job failure handling
end.non_existent_method()
print("Batch job failure test completed.")
