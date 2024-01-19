import json
import numpy as np


####################################################################################
# Used to parse JSON strings from ArtifactsTextView.model_summary()['Model Metrics']
####################################################################################


def get_reg_metric(scores: str, metric: str):
    """
    scores(str): json string with model metrics, list of a single dict [{'metric': float, 'metric': float, ...}]
    metric(str): key in metrics dictionary for metric of choice
    """
    if not scores:
        return None
    return json.loads(scores)[0].get(metric, None)


def get_class_metric_ave(scores: str, metric: str):
    """ "
    scores(str): json string with model metrics, list[dict] [{'class': int, 'roc_auc': float, ...}, {'class': int, 'roc_auc': float, ...}]
    metric(str): key in metrics dictionary for metric of choice
    """
    if not scores:
        return None
    scores_list = json.loads(scores)
    return np.mean([d.get(metric, np.nan) for d in scores_list])


def get_class_metric(scores: str, category: int, metric: str):
    """ "
    scores(str): json string with model metrics, list[dict] [{'class': int, 'roc_auc': float, ...}, {'class': int, 'roc_auc': float, ...}]
    category(int): classification category to pull roc_auc for
    metric(str): key in metrics dictionary for metric of choice
    """
    if not scores:
        return None
    elif len(json.loads(scores)) < category:
        return None
    return json.loads(scores)[category].get(metric, None)
