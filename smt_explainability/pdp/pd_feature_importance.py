import numpy as np
from .partial_dependence import partial_dependence


def compute_pd_feature_importance(features, is_categorical, pd_results):
    importances = []
    for i, feature in enumerate(features):
        pd = pd_results[i]["average"]
        if is_categorical[feature]:
            importance = (np.max(pd) - np.min(pd)) / 4
        else:
            k = len(pd)
            mean_pd = np.mean(pd)
            importance = np.power(np.sum((pd - mean_pd) ** 2) / (k - 1), 0.5)
        importances.append(importance)
    return importances


def pd_feature_importance(
    model,
    x,
    features,
    *,
    sample_weight=None,
    categorical_feature_indices=None,
    percentiles=(0.05, 0.95),
    grid_resolution=100,
    method="uniform",
    ratio_samples=None,
    seed=None,
):
    pd_results = partial_dependence(
        model,
        x,
        features,
        sample_weight=sample_weight,
        categorical_feature_indices=categorical_feature_indices,
        percentiles=percentiles,
        grid_resolution=grid_resolution,
        method=method,
        kind="average",
        ratio_samples=ratio_samples,
        seed=seed,
    )

    is_categorical = [False] * x.shape[1]
    if categorical_feature_indices is not None:
        for feature_idx in categorical_feature_indices:
            is_categorical[feature_idx] = True

    importances = compute_pd_feature_importance(features, is_categorical, pd_results)
    return importances
