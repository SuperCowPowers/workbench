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

    # from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit import RDLogger
except ImportError:
    print("RDKit Python module not found! pip install rdkit")
    sys.exit(0)


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

    def transform_impl(self, **kwargs):
        """Compute a Feature Set based on RDKit Descriptors"""

        # Check the input DataFrame has the required columns
        if "smiles" not in self.input_df.columns:
            raise ValueError("Input DataFrame must have a 'smiles' column")

        # Compute/add all the RDKIT Descriptors
        self.output_df = self.compute_rdkit_descriptors(self.input_df)

        # Drop any NaNs (and INFs)
        self.output_df = pandas_utils.drop_nans(self.output_df, how="all")

    def compute_rdkit_descriptors(self, process_df: pd.DataFrame) -> pd.DataFrame:
        """Compute and add all the RDKit Descriptors
        Args:
            process_df(pd.DataFrame): The DataFrame to process and generate RDKit Descriptors
        Returns:
            pd.DataFrame: The input DataFrame with all the RDKit Descriptors added
        """
        self.log.important("Computing RDKit Descriptors...")

        # Conversion to Molecules
        molecules = [Chem.MolFromSmiles(smile) for smile in process_df["smiles"]]

        # Now get all the RDKIT Descriptors
        # all_descriptors = [x[0] for x in Descriptors._descList]

        # There's an overflow issue that happens with the IPC descriptor, so we'll remove it
        # See: https://github.com/rdkit/rdkit/issues/1527
        # if "Ipc" in all_descriptors:
        #    all_descriptors.remove("Ipc")

        # Get the descriptors that are most useful for Solubility
        best_descriptors = [
            "MolLogP",
            "MolWt",
            "TPSA",
            "NumHDonors",
            "NumHAcceptors",
            "NumRotatableBonds",
            "NumAromaticRings",
            "NumSaturatedRings",
            "NumAliphaticRings",
            "NumAromaticCarbocycles",
        ]
        best_20_descriptors = best_descriptors + [
            "HeavyAtomCount",
            "RingCount",
            "Chi0",
            "Chi1",
            "Kappa1",
            "Kappa2",
            "Kappa3",
            "LabuteASA",
            "FractionCSP3",
            "HallKierAlpha",
        ]
        best_30_descriptors = best_20_descriptors + [
            "SMR_VSA1",
            "SlogP_VSA1",
            "EState_VSA1",
            "VSA_EState1",
            "PEOE_VSA1",
            "NumValenceElectrons",
            "NumRadicalElectrons",
            "MaxPartialCharge",
            "MinPartialCharge",
            "MaxAbsPartialCharge",
        ]
        best_40_descriptors = best_30_descriptors + [
            "MolMR",
            "ExactMolWt",
            "NOCount",
            "NumHeteroatoms",
            "NumAmideBonds",
            "FpDensityMorgan1",
            "FpDensityMorgan2",
            "FpDensityMorgan3",
            "MaxEStateIndex",
            "MinEStateIndex",
        ]

        # Super useful Molecular Descriptor Calculator Class
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(best_40_descriptors)
        column_names = calc.GetDescriptorNames()

        descriptor_values = [calc.CalcDescriptors(m) for m in molecules]
        df_features = pd.DataFrame(descriptor_values, columns=column_names)
        return pd.concat([process_df, df_features], axis=1)


if __name__ == "__main__":
    """Exercise the RDKitDescriptors Class"""

    # Create the class with inputs and outputs and invoke the transform
    data_to_features = RDKitDescriptors("aqsol_data", "aqsol_rdkit_features")
    data_to_features.set_output_tags(["logS", "rdkit", "public"])
    query = 'SELECT id, "group", solubility, smiles FROM aqsol_data'
    data_to_features.transform(target_column="solubility", id_column="udm_mol_bat_id", query=query)
