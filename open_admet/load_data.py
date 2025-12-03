"""Load the Open ADMET data using Workbench API"""

from workbench.api import DataSource, Endpoint, ParameterStore, DFStore
from workbench.core.transforms.pandas_transforms import PandasToFeatures


def main():
    # Access the Parameter Store and DataFrame Store
    params = ParameterStore()
    df_store = DFStore()

    # First load the training data into a DataSource (AWS Athena)
    # ds = DataSource("train_data.csv", name="open_admet")
    ds = DataSource("open_admet")
    """
    df = ds.pull_dataframe()
    
    # Run the data through our RDKit+Mordred Feature Endpoint
    rdkit_end = Endpoint("smiles-to-taut-md-stereo-v1")
    df_features = rdkit_end.inference(df)
    
    # Temp: Shove this into the DFStore for inspection/use later
    df_store.upsert("/workbench/datasets/open_admet_featurized", df_features)
    """
    df_features = df_store.get("/workbench/datasets/open_admet_featurized")

    # Grab the Feature List created by the Endpoint
    features = params.get("/workbench/feature_lists/rdkit_mordred_stereo_v1")

    # Now Split these into separate FeatureSets for each assay
    # Note: There are two columns molecule_name and smiles in the DataSource that aren't assays
    #       every other column is an assay that we want to create a FeatureSet for
    assay_columns = [col for col in ds.columns if col not in ["molecule_name", "smiles"]]
    for assay in assay_columns:
        fs_name = f"open_admet_{assay}"

        # Pull all rows with non-null values for this assay
        df_assay = df_features.dropna(subset=[assay])

        # Just keep the molecule_name, smiles, assay, and feature columns
        keep_columns = ["molecule_name", "smiles", assay] + features
        df_assay = df_assay[keep_columns]

        # Write out the CSV file locally to csv_files/ directory
        # df_assay[["molecule_name", "smiles", assay]].to_csv(f"csv_files/{fs_name}.csv", index=False)

        # Create a Feature Set (takes a while)
        print(f"Creating FeatureSet: {fs_name} with {len(df_assay)} entries")
        to_features = PandasToFeatures(fs_name)
        to_features.set_input(df_assay, id_column="molecule_name")
        to_features.set_output_tags(["open_admet", assay])
        to_features.transform()


if __name__ == "__main__":
    main()
