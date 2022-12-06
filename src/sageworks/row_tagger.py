"""RowTagger: Encapsulate business logic and special cases for tagging rows in a dataframe"""
# License: Apache 2.0 ©SuperCowPowers LLC
from collections import defaultdict
import pandas as pd

# Local Imports
from sageworks import feature_spider


# Class: RowTagger
class RowTagger:
    """RowTagger: Encapsulate business logic and special cases for tagging rows in a dataframe"""
    """Domain Specific (will be refactored later)
        - Stereoisomers
        - Replicates
        - CoIncident (with reasonable difference in target value)
        - High Target Gradient (HTG) Neighborhood
        - Activity Cliff Group Candidate (subset of HTG with additional Logic)"""

    def __init__(self, dataframe: pd.DataFrame, features: list, id_column: str, target: str,
                 source: str, min_dist: float, min_target_diff: float):

        # Set up some parameters
        self.id_column = id_column
        self.min_dist = min_dist
        self.min_target_diff = min_target_diff

        # Do a validation check on the dataframe
        self.df = dataframe
        self.validate_input_data()

        # We need the feature spider for the more advanced tags
        self.f_spider = feature_spider.FeatureSpider(self.df, features, id_column=self.id_column,
                                                     target=target, source=source)

        # Add a 'tags' column (if it doesn't already exist)
        if 'tags' not in self.df.columns:
            self.df['tags'] = [set() for _ in range(len(self.df.index))]

    def validate_input_data(self):
        # Make sure it's a dataframe
        if not isinstance(self.df, pd.DataFrame):
            print('Input is NOT a DataFrame!')
            return False

        # Make sure it has some rows and columns
        rows, columns = self.df.shape
        if rows == 0:
            print('Input DataFrame has 0 rows!')
            return False
        if columns == 0:
            print('Input DataFrame has 0 columns!')
            return False

        # Domain Specific (refactor later)
        if 'SMILES' not in self.df.columns:
            print('Input DataFrame needs a SMILES column!')
            return False
        if self.id_column not in self.df.columns:
            print('Input DataFrame needs a ID column!')
            return False

        # AOK
        return True

    def tag_rows(self) -> pd.DataFrame:
        """Run all the current registered taggers"""
        # The taggers that all take file names we want to run
        taggers = [self.stereo_isomers, self.replicants, self.coincident, self.high_gradients]
        for tagger in taggers:
            tagger()
        return self.df

    def stereo_isomers(self):
        """Tag all SMILES strings that are stereo isomers"""
        grouper = defaultdict(list)
        for index, smile in zip(self.df.index, self.df['SMILES']):
            # Remove the @ symbols from the SMILES string
            no_ats = smile.replace('@', '')

            # Check if the original SMILES string is the same as this one
            # If not the same original SMILES than add to the grouper
            if not any([self.df['SMILES'].loc[g_index] == smile for g_index in grouper[no_ats]]):
                grouper[no_ats].append(index)

        # Now finally mark multiples within a group as a stereo_isomer
        for group, index_list in grouper.items():
            if len(index_list) > 1:
                # LOC uses the index ^label^ (which is what we want)
                for index in index_list:
                    self.df['tags'].loc[index].add('stereo_isomer')

    def replicants(self):
        """Tag all the ID strings that represents a replicant experiment"""
        grouper = defaultdict(list)
        for index, id_string in zip(self.df.index, self.df['ID']):
            # Split off the last -N in the ID string
            no_rep_index = '-'.join(id_string.split('-')[:-1])
            grouper[no_rep_index].append(index)

        # Now finally mark multiples within a group as a stereo_isomer
        for group, index_list in grouper.items():
            if len(index_list) > 1:
                # LOC uses the index ^label^ (which is what we want)
                for index in index_list:
                    self.df['tags'].loc[index].add('replicant')

    def coincident(self):
        """Find observations with the SAME features that have different target values"""
        coincident_indexes = self.f_spider.coincident(self.min_target_diff, verbose=False)

        # We get back index offsets (not labels) so we need to use iloc
        for index in coincident_indexes:
            self.df['tags'].iloc[index].add('coincident')

    def high_gradients(self):
        """Find observations close in feature space with a high difference in target values
           High Target Gradient (HTG) """
        htg_indexes = self.f_spider.high_gradients(self.min_dist, self.min_target_diff, verbose=False)

        # We get back index offsets (not labels) so we need to use iloc
        for index in htg_indexes:
            self.df['tags'].iloc[index].add('htg')


def test():
    """Test for the RowTagger Class"""

    # Set some pandas options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Make some fake data
    data = {'ID': ['IVC-0-1', 'IVC-1-1', 'IVC-1-2', 'IVC-3-1', 'IVC-4-1',
                   'IVC-5-1', 'IVC-6-1', 'IVC-6-2', 'IVC-8-1', 'IVC-9-1'],
            'SMILES': ['CC1(C)[C@@H]2C[C@H]1C2(C)C',
                       'CC1(C)[C@H]2C[C@@H]1C2(C)C',
                       'C[C@]12O[C@H]1C[C@H]1C[C@@H]2C1(C)C',
                       'C[C@]12O[C@H]1C[C@H]1C[C@@H]2C1(C)C',
                       'CC(C)[C@@H]1CC[C@@H](C)C[C@H]1OC(=O)[C@H](C)O',
                       'CC1(C)[C@@H]2C[C@H]1C2(C)C',
                       'CC1(C)[C@H]2C[C@@H]1C2(C)C',
                       'C[C@]12O[C@H]1C[C@H]1C[C@@H]2C1(C)C',
                       'C[C@]12O[C@H]1C[C@H]1C[C@@H]2C1(C)C',
                       'CC(C)[C@@H]1CC[C@@H](C)C[C@H]1OC(=O)[C@H](C)O'
                       ],
            'feat1': [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
            'feat2': [1.0, 1.0, 1.1, 3.0, 4.0, 1.0, 1.0, 1.1, 3.0, 4.0],
            'feat3': [0.1, 0.1, 0.2, 1.6, 2.5, 0.1, 0.1, 0.2, 1.6, 2.5],
            'source': ['A', 'A', 'B', 'A', 'B', 'A', 'A', 'B', 'A', 'B'],
            'logS': [-3.1, -6.0, -6.0, -4.0, -2.0, -3.1, -6.0, -6.0, -4.0, -2.0]}
    data_df = pd.DataFrame(data)

    # Create the class and run the taggers
    row_tagger = RowTagger(data_df, ['feat1', 'feat2', 'feat3'], id_column='ID', target='logS', source='source',
                           min_dist=2.0, min_target_diff=1.0)
    data_df = row_tagger.tag_rows()
    print(data_df)


if __name__ == '__main__':
    test()
