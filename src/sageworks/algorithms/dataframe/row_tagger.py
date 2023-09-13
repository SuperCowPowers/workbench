"""RowTagger: Generalized Row Tagger (domain specific logic should be captured in a subclass"""
import pandas as pd

# SageWorks Imports
from sageworks.algorithms.table.light import feature_spider


# Class: RowTagger
class RowTagger:
    """RowTagger: Encapsulate business logic and special cases for tagging rows in a dataframe
    - CoIncident (with reasonable difference in target value)
    - High Target Gradient (HTG) Neighborhood
    """

    def __init__(
        self,
        dataframe: pd.DataFrame,
        features: list,
        id_column: str,
        target: str,
        min_dist: float,
        min_target_diff: float,
    ):
        # Set up some parameters
        self.id_column = id_column
        self.min_dist = min_dist
        self.min_target_diff = min_target_diff

        # Do a validation check on the dataframe
        self.df = dataframe
        self.validate_input_data()

        # We need the feature spider for the more advanced tags
        self.f_spider = feature_spider.FeatureSpider(self.df, features, id_column=self.id_column, target=target)

        # Add a 'tags' column (if it doesn't already exist)
        if "tags" not in self.df.columns:
            self.df["tags"] = [set() for _ in range(len(self.df.index))]

    def validate_input_data(self):
        # Make sure it's a dataframe
        if not isinstance(self.df, pd.DataFrame):
            print("Input is NOT a DataFrame!")
            return False

        # Make sure it has some rows and columns
        rows, columns = self.df.shape
        if rows == 0:
            print("Input DataFrame has 0 rows!")
            return False
        if columns == 0:
            print("Input DataFrame has 0 columns!")
            return False

        # Make sure it has an ID column (domain specific stuff in subclass)
        if self.id_column not in self.df.columns:
            print("Input DataFrame needs a ID column!")
            return False

        # AOK
        return True

    def tag_rows(self) -> pd.DataFrame:
        """Run all the current registered taggers"""
        # The taggers that all take file names we want to run
        taggers = [self.coincident, self.high_gradients]
        for tagger in taggers:
            tagger()
        return self.df

    def coincident(self):
        """Find observations with the SAME features that have different target values"""
        coincident_indexes = self.f_spider.coincident(self.min_target_diff, verbose=False)

        # We get back index offsets (not labels) so we need to use iloc
        for index in coincident_indexes:
            self.df["tags"].iloc[index].add("coincident")

    def high_gradients(self):
        """Find observations close in feature space with a high difference in target values
        High Target Gradient (HTG)"""
        htg_indexes = self.f_spider.high_gradients(self.min_dist, self.min_target_diff, verbose=False)

        # We get back index offsets (not labels) so we need to use iloc
        for index in htg_indexes:
            self.df["tags"].iloc[index].add("htg")


def test():
    """Test for the RowTagger Class"""

    # Set some pandas options
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)

    # Make some fake data
    data = {
        "ID": [
            "id_0",
            "id_0",
            "id_2",
            "id_3",
            "id_4",
            "id_5",
            "id_6",
            "id_7",
            "id_8",
            "id_9",
        ],
        "feat1": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat2": [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
        "feat3": [0.1, 0.1, 0.2, 1.6, 2.5, 0.1, 0.1, 0.2, 1.6, 2.5],
        "price": [31, 60, 62, 40, 20, 31, 61, 60, 40, 20],
    }
    data_df = pd.DataFrame(data)

    # Create the class and run the taggers
    row_tagger = RowTagger(
        data_df,
        features=["feat1", "feat2", "feat3"],
        id_column="ID",
        target="price",
        min_dist=2.0,
        min_target_diff=1.0,
    )
    data_df = row_tagger.tag_rows()
    print(data_df)


if __name__ == "__main__":
    test()
