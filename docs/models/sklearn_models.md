# Scikit-Learn Models

Workbench supports any scikit-learn estimator — classifiers, regressors, clustering, and more. Specify the class name and import string, and Workbench handles training, deployment, and inference on AWS.

## Creating a Scikit-Learn Model

### Classification (RandomForest)

```python
from workbench.api import FeatureSet

fs = FeatureSet("wine_features")

model = fs.to_model(
    model_class="RandomForestClassifier",
    model_import_str="from sklearn.ensemble import RandomForestClassifier",
    name="wine-rfc-class",
    target_column="wine_class",
    description="Wine RandomForest Classification",
    tags=["wine", "random-forest"],
)
```

### Clustering (KMeans)

```python
from workbench.api import FeatureSet

fs = FeatureSet("abalone_features")

model = fs.to_model(
    model_class="KMeans",
    model_import_str="from sklearn.cluster import KMeans",
    name="abalone-kmeans",
    target_column="class_number_of_rings",
    description="Abalone KMeans Clustering",
    tags=["abalone", "kmeans"],
)
```

### Clustering (DBSCAN)

```python
from workbench.api import FeatureSet

fs = FeatureSet("abalone_features")

model = fs.to_model(
    model_class="DBSCAN",
    model_import_str="from sklearn.cluster import DBSCAN",
    name="abalone-dbscan",
    target_column="class_number_of_rings",
    description="Abalone DBSCAN Clustering",
    tags=["abalone", "dbscan"],
    train_all_data=True,
)
```

## Hyperparameters

Pass scikit-learn constructor arguments directly via `hyperparameters`:

```python
model = fs.to_model(
    model_class="RandomForestClassifier",
    model_import_str="from sklearn.ensemble import RandomForestClassifier",
    name="wine-rfc-tuned",
    target_column="wine_class",
    hyperparameters={
        "n_estimators": 200,
        "max_depth": 10,
        "min_samples_split": 5,
    },
)
```

## Supported Estimators

Any scikit-learn class that follows the estimator API works. Common choices:

| Task | Estimator | Import |
|------|-----------|--------|
| **Classification** | RandomForestClassifier | `sklearn.ensemble` |
| **Classification** | GradientBoostingClassifier | `sklearn.ensemble` |
| **Classification** | SVC | `sklearn.svm` |
| **Regression** | Ridge | `sklearn.linear_model` |
| **Regression** | BayesianRidge | `sklearn.linear_model` |
| **Regression** | KNeighborsRegressor | `sklearn.neighbors` |
| **Clustering** | KMeans | `sklearn.cluster` |
| **Clustering** | DBSCAN | `sklearn.cluster` |

!!! tip
    Use `train_all_data=True` to train on the full dataset without a holdout split — useful for clustering or when you need maximum training data.

!!! note "Examples"
    Full code listings: [`examples/models/random_forest.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/models/random_forest.py), [`examples/models/knn.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/models/knn.py), [`examples/models/dbscan.py`](https://github.com/SuperCowPowers/workbench/blob/main/examples/models/dbscan.py)

---

## Questions?

<img align="right" src="/images/scp.png" width="180">

The SuperCowPowers team is happy to answer any questions you may have about AWS® and Workbench.

- **Support:** [workbench@supercowpowers.com](mailto:workbench@supercowpowers.com)
- **Discord:** [Join us on Discord](https://discord.gg/WHAJuz8sw8)
- **Website:** [supercowpowers.com](https://www.supercowpowers.com)
