from sageworks.transforms.pandas_transforms.data_to_pandas import DataToPandas
from sageworks.transforms.pandas_transforms.pandas_to_features import PandasToFeatures

# Grab the data from the DataSource
data_to_pandas = DataToPandas("abalone_data")
data_to_pandas.transform()
df = data_to_pandas.get_output()

# Convert the regression target to a categorical type and drop the old column
bins = [0, 7, 12, float("inf")]
labels = ["young", "adult", "old"]
df["clam_age_class"] = pd.cut(df["class_number_of_rings"], bins=bins, labels=labels)
df.drop("class_number_of_rings", axis=1, inplace=True)

# Create the FeatureSet
pandas_to_features = PandasToFeatures("abalone_classification")
pandas_to_features.set_input(df, target_column="clam_age_class")
pandas_to_features.set_output_tags(["abalone", "classification"])
pandas_to_features.transform()
