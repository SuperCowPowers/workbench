"""FeatureSpider: A Spider for data/feature investigation and QA"""
# License: Apache 2.0 ©SuperCowPowers LLC
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


# Feature Spider Class
class FeatureSpider:
    def __init__(self, df: pd.DataFrame, features: list, id_column: str, target: str, source: str):

        # Check for expected columns (used later)
        for column in ['SMILES', id_column, target, source]:
            if column not in df.columns:
                print(f'DataFrame does not have required {column} Column!')
                return

        # Set internal vars that are used later
        self.df = df
        self.id_column = id_column
        self.target = target
        self.source = source
        self.features = features

        # Build our KNN model pipeline with StandardScalar
        knn = KNeighborsRegressor(n_neighbors=10, weights='distance')
        self.pipe = make_pipeline(StandardScaler(), knn)

        # Fit Model on features and target
        y = df[target]
        X = df[features]
        self.pipe.fit(X, y)

        # Grab the Standard Scalar and KNN from the pipeline model
        # Note: These handles need to be constructed after the fit
        self.scalar = self.pipe['standardscaler']
        self.knn = self.pipe['kneighborsregressor']

    def get_feature_matrix(self):
        """Return the KNN Model Internal Feature Matrix"""
        return self.knn._fit_X

    def predict(self, pred_df: pd.DataFrame) -> list:
        """Provide a prediction from the KNN Pipeline model (knn_prediction)"""
        return self.pipe.predict(pred_df[self.features])

    def confidence_scores(self, pred_df: pd.DataFrame, model_preds=None) -> list:
        """Compute Confidence Scores for each Prediction"""

        # Get all the KNN information relevant to this calculation
        neighbor_info = self.neighbor_info(pred_df)

        # Handles for all the relevant info
        knn_preds = neighbor_info['knn_prediction']
        target_values = neighbor_info['knn_target_values']
        distances = neighbor_info['knn_distances']

        # We can score confidence even if we don't have model predictions (less good)
        if model_preds is None:
            model_preds = knn_preds
            stddev_multiplier = 1.5
        else:
            stddev_multiplier = 1.0

        # Now a big loop over all these values to compute the confidence scores
        confidence_scores = []
        for pred, knn_pred, str_val_list, str_dist_list in zip(model_preds, knn_preds, target_values, distances):
            # Each of these is a string of a list (yes a bit cheesy)
            vals = [float(val) for val in str_val_list.split(', ')]
            _ = [float(dis) for dis in str_dist_list.split(', ')]  # dist current not used

            # Compute stddev of the target values
            knn_stddev = np.std(vals)

            # Confidence Score
            conf = 0.5 * (2.0 - abs(float(pred) - float(knn_pred)))
            conf -= knn_stddev * stddev_multiplier

            # Confidence score has min-max of 0-1
            conf = min(max(conf, 0), 1)

            confidence_scores.append(conf)

        # Return the confidence scores
        return confidence_scores

    def neighbor_info(self, pred_df: pd.DataFrame) -> pd.DataFrame:
        """Provide information on the neighbors (prediction, knn_target_values, knn_distances)"""

        # Make sure we have all the features
        if not set(self.features) <= set(pred_df.columns):
            print(f'DataFrame does not have required features: {self.features}')
            return None

        # Run through scaler
        x_scaled = self.scalar.transform(pred_df[self.features])

        # Add the data to a copy of the dataframe
        results_df = pd.DataFrame()
        results_df['knn_prediction'] = self.knn.predict(x_scaled)

        # Get the Neighbors Information
        neigh_dist, neigh_ind = self.knn.kneighbors(x_scaled)
        target_values = self.knn._y[neigh_ind]

        # Note: We're assuming that the Neighbor Index is the same order/cardinality as the dataframe
        results_df['knn_target_values'] = [', '.join([str(val) for val in values]) for values in target_values]
        results_df['knn_distances'] = [', '.join([str(dis) for dis in distances]) for distances in neigh_dist]

        # Do we have Source data
        if self.source is not None:
            for index in neigh_ind:
                print(self.df.iloc[index][self.source])
            results_df['sources'] = [', '.join([self.df.iloc[index][self.source][0] for index in neigh_ind])]
        return results_df

    def neighbor_ids(self, pred_df) -> pd.DataFrame:
        """Provide id + smiles for the neighbors (knn_ids, knn_smiles)"""

        # Run through scaler
        x_scaled = self.scalar.transform(pred_df[self.features])

        # Add the data to a copy of the dataframe
        results_df = pred_df.copy()

        # Neighbor ID/SMILE lookups
        neigh_dist, neigh_ind = self.knn.kneighbors(x_scaled)
        results_df['knn_ids'] = [', '.join(self.df.iloc[index][self.id_column] for index in indexes) for indexes in neigh_ind]
        results_df['knn_smiles'] = [', '.join(self.df.iloc[index]['SMILES'] for index in indexes) for indexes in neigh_ind]
        return results_df

    def coincident(self, target_diff, verbose=True):
        """Convenience method that calls high_gradients with a distance of 0"""
        return self.high_gradients(0.0, target_diff, verbose)

    def high_gradients(self, within_distance: float, target_diff: float, verbose: bool = True) -> list:
        """This basically loops over all the X features in the KNN model
           - Grab the neighbors distances and indices
           - For neighbors `within_distance`* grab target values
           - If target values have a difference > `target_diff`
              - List out the details of the observations and the distance, target diff
           *Note: standardized feature space
        """
        global_htg_set = set()
        for my_index, obs in enumerate(self.knn._fit_X):
            neigh_distances, neigh_indexes = self.knn.kneighbors([obs])
            neigh_distances = neigh_distances[0]  # Returns a list within a list so grab the inner list
            neigh_indexes = neigh_indexes[0]  # Returns a list within a list so grab the inner list
            target_values = self.knn._y[neigh_indexes]

            # Grab the info for this observation
            my_id = self.df.iloc[my_index][self.id_column]
            my_smile = self.df.iloc[my_index]['SMILES']
            my_target = self.knn._y[my_index]

            # Loop through the neighbors
            # Note: by definition this observation will be in the neighbors so account for that
            my_htg_set = set()
            for n_index, dist, target in zip(neigh_indexes, neigh_distances, target_values):

                # Skip myself
                if n_index == my_index:
                    continue

                # Compute target differences `within_distance` feature space
                _diff = abs(my_target - target)
                if dist <= within_distance and _diff > target_diff:

                    # Update the individual HTG set for this observation
                    my_htg_set.add((n_index, dist, _diff, target))

                    # Add both (me and my neighbor) to the global high gradient index list
                    global_htg_set.add(my_index)
                    global_htg_set.add(n_index)

            # Okay we've computed our HTG set for this observation
            # Print out all my HTG neighbors if the verbose flag is set
            if verbose and my_htg_set:
                source = self.df.iloc[my_index][self.source]
                print(f"\nOBSERVATION: {my_id}")
                print(f"\t{my_id}({my_target:.2f}):{my_smile} {source}")
                for htg_neighbor, dist, _diff, target in my_htg_set:
                    neighbor_id = self.df.iloc[htg_neighbor][self.id_column]
                    neighbor_smile = self.df.iloc[htg_neighbor]['SMILES']
                    n_source = self.df.iloc[htg_neighbor][self.source]
                    print(f"\t{neighbor_id}({target:.2f}):{neighbor_smile} {n_source} TD:{_diff:.2f} FD:{dist}")

        # Return the global list of indexes that are part of high target gradient (HTG) pairs
        return list(global_htg_set)


def test():
    """Test for the Feature Spider Class"""
    # Set some pandas options
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    # Make some fake data
    data = {'ID': ['IVC-0', 'IVC-1', 'IVC-2', 'IVC-3', 'IVC-4',
                   'IVC-5', 'IVC-6', 'IVC-7', 'IVC-8', 'IVC-9'],
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
    f_spider = FeatureSpider(data_df, ['feat1', 'feat2', 'feat3'], id_column='ID', target='logS', source='source')
    preds = f_spider.predict(data_df)
    print(preds)
    coincident = f_spider.coincident(2)
    print('COINCIDENT')
    print(coincident)
    high_gradients = f_spider.high_gradients(2, 2)
    print('\nHIGH GRADIENTS')
    print(high_gradients)

    # Run some neighbor methods
    query_df = data_df[data_df['ID'] == 'IVC-0'].copy()
    print(f_spider.neighbor_info(query_df))


if __name__ == '__main__':
    test()
