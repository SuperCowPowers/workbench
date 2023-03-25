"""RDKitDescriptors: Compute a Feature Set based on RDKit Descriptors"""
import sys
import pandas as pd

# Local Imports
from sageworks.transforms.data_to_features.light.data_to_features_light import DataToFeaturesLight
from sageworks.transforms.pandas_transforms import pandas_utils

# Third Party Imports
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    from rdkit.ML.Descriptors import MoleculeDescriptors
    from rdkit import RDLogger
except ImportError:
    print('RDKit Python module not found! pip install rdkit-pypi')
    sys.exit(1)


class RDKitDescriptors(DataToFeaturesLight):
    def __init__(self, input_uuid: str = None, output_uuid: str = None):
        """RDKitDescriptors: Compute a Feature Set based on RDKit Descriptors"""

        # Call superclass init
        super().__init__(input_uuid, output_uuid)

        # Turn off warnings for RDKIT (revisit this)
        RDLogger.DisableLog('rdApp.*')

    def transform_impl(self, **kwargs):
        """Compute a Feature Set based on RDKit Descriptors"""

        # Note: The parent class manages the input_df and output_df
        #       So we simply need to grab the input and produce the output

        # Remove all the dataframe columns except for ID and SMILES
        self.output_df = self.input_df[['id', 'smiles', 'solubility']]

        # Compute/add all the RDKIT Descriptors
        self.output_df = self.compute_rdkit_descriptors()

        # Drop any NaNs (and INFs)
        self.output_df = pandas_utils.drop_nans(self.output_df, how='any')

    def compute_rdkit_descriptors(self):
        """Compute and add all the RDKit Descriptors"""

        # Conversion to Molecules
        molecules = [Chem.MolFromSmiles(smile) for smile in self.input_df['smiles']]

        # Now get all the RDKIT Descriptors
        all_descriptors = ([x[0] for x in Descriptors._descList])

        # There's an overflow issue that happens with the IPC descriptor, so we'll remove it
        # See: https://github.com/rdkit/rdkit/issues/1527
        if 'Ipc' in all_descriptors:
            all_descriptors.remove('Ipc')

        # Super useful Molecular Descriptor Calculator Class
        calc = MoleculeDescriptors.MolecularDescriptorCalculator(all_descriptors)
        column_names = calc.GetDescriptorNames()
        descriptor_values = [calc.CalcDescriptors(m) for m in molecules]
        df_features = pd.DataFrame(descriptor_values, columns=column_names)
        return pd.concat([self.output_df, df_features], axis=1)


# Simple test of the RDKitDescriptors functionality
def test():
    """Test the RDKitDescriptors Class"""

    # Create the class with inputs and outputs and invoke the transform
    input_uuid = 'aqsol_data'
    output_uuid = 'test_rdkit_features'
    RDKitDescriptors(input_uuid, output_uuid).transform(delete_existing=True)


if __name__ == "__main__":
    test()
