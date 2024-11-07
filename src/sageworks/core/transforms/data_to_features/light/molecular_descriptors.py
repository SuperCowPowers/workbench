"""MolecularDescriptors: Compute a Feature Set based on RDKit Descriptors

   Note: An alternative to using this class is to use the `compute_molecular_descriptors` function directly.
         df_features = compute_molecular_descriptors(df)
         to_features = PandasToFeatures("my_feature_set")
            to_features.set_input(df_features, id_column="id")
            to_features.set_output_tags(["blah", "whatever"])
            to_features.transform()
"""

# Local Imports
from sageworks.core.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight
from sageworks.utils.chem_utils import compute_molecular_descriptors


class MolecularDescriptors(DataToFeaturesLight):
    """MolecularDescriptors: Create a FeatureSet (RDKit Descriptors) from a DataSource

    Common Usage:
        ```python
        to_features = MolecularDescriptors(data_uuid, feature_uuid)
        to_features.set_output_tags(["aqsol", "whatever"])
        to_features.transform()
        ```
    """

    def __init__(self, data_uuid: str, feature_uuid: str):
        """MolecularDescriptors Initialization

        Args:
            data_uuid (str): The UUID of the SageWorks DataSource to be transformed
            feature_uuid (str): The UUID of the SageWorks FeatureSet to be created
        """

        # Call superclass init
        super().__init__(data_uuid, feature_uuid)

    def transform_impl(self, **kwargs):
        """Compute a Feature Set based on RDKit Descriptors"""

        # Compute/add all the Molecular Descriptors
        self.output_df = compute_molecular_descriptors(self.input_df)


if __name__ == "__main__":
    """Exercise the MolecularDescriptors Class"""
    from sageworks.api.data_source import DataSource

    full_test = False

    # Unit Test: Create the class with inputs
    unit_test = MolecularDescriptors("aqsol_data", "aqsol_mol_descriptors")
    unit_test.input_df = DataSource("aqsol_data").pull_dataframe()[:100]
    unit_test.transform_impl()
    output_df = unit_test.output_df
    print(output_df.shape)
    print(output_df.head())

    # Full Test: Create the class with inputs and outputs and invoke the transform
    if full_test:
        data_to_features = MolecularDescriptors("aqsol_data", "aqsol_mol_descriptors")
        data_to_features.set_output_tags(["logS", "public"])
        query = 'SELECT id, "group", solubility, smiles FROM aqsol_data'
        data_to_features.transform(id_column="id", query=query)
