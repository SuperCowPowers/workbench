"""RDKitDescriptors: Compute a Feature Set based on RDKit Descriptors"""
import sys
import pandas as pd

# Local Imports
from sageworks.transforms.data_to_features.light.data_to_features_light import (
    DataToFeaturesLight,
)
from sageworks.utils import pandas_utils

# Third Party Imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit import RDLogger
except ImportError:
    print("RDKit Python module not found! pip install rdkit")
    sys.exit(1)


class RDKitDescriptors(DataToFeaturesLight):
    """RDKitDescriptors: Create a FeatureSet (RDKit Descriptors) from a DataSource

    Common Usage:
        to_features = RDKitDescriptors(data_uuid, feature_uuid)
        to_features.set_output_tags(["aqsol", "rdkit", "whatever"])
        to_features.transform()
    """

    def __init__(self, data_uuid: str, feature_uuid: str):
        """RDKitDescriptors Initialization"""

        # Call superclass init
        super().__init__(data_uuid, feature_uuid)

        # Turn off warnings for RDKIT (revisit this)
        RDLogger.DisableLog("rdApp.*")

    def transform_impl(self, target: str, **kwargs):
        """Compute a Feature Set based on RDKit Descriptors
        Args:
            target(str): The name of the target column
        """

        # Set up our target column
        self.target = target

        # Check the input DataFrame has the required columns
        if "smiles" not in self.input_df.columns:
            raise ValueError("Input DataFrame must have a 'smiles' column")
        if target not in self.input_df.columns:
            raise ValueError(f"Input DataFrame must have a '{target}' column")

        # Compute/add all the RDKIT Descriptors
        self.output_df = self.compute_rdkit_descriptors(self.input_df)

        # Drop any NaNs (and INFs)
        self.output_df = pandas_utils.drop_nans(self.output_df, how="any")

    def compute_rdkit_descriptors(self, process_df: pd.DataFrame) -> pd.DataFrame:
        """Compute and add all the RDKit Descriptors
        Args:
            process_df(pd.DataFrame): The DataFrame to process and generate RDKit Descriptors
        Returns:
            pd.DataFrame: The input DataFrame with all the RDKit Descriptors added
        """

        # Hack:
        process_df = process_df.head(100)

        # Conversion to Molecules
        molecules = [Chem.MolFromSmiles(smile) for smile in process_df["smiles"]]

        # Now get all the RDKIT Descriptors
        all_descriptors = [x[0] for x in Descriptors._descList]

        # There's an overflow issue that happens with the IPC descriptor, so we'll remove it
        # See: https://github.com/rdkit/rdkit/issues/1527
        if "Ipc" in all_descriptors:
            all_descriptors.remove("Ipc")

        # FIXME: Stupid hack
        all_descriptors = all_descriptors[:10]
        print(f"Using {len(all_descriptors)} descriptors")
        print(all_descriptors)

        # Super useful Molecular Descriptor Calculator Class
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptors)
        column_names = calc.GetDescriptorNames()

        descriptor_values = [calc.CalcDescriptors(m) for m in molecules]
        df_features = pd.DataFrame(descriptor_values, columns=column_names)
        return pd.concat([process_df, df_features], axis=1)


if __name__ == "__main__":
    """Exercise the RDKitDescriptors Class"""

    # Create the class with inputs and outputs and invoke the transform
    data_to_features = RDKitDescriptors("logs_test_data_clean", "test_rdkit_features")
    data_to_features.set_output_tags(["logS", "test", "proprietary"])
    data_to_features.transform(target="udm_asy_res_value", id_column="udm_mol_bat_id", event_time_column="udm_asy_date")
