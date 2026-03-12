from .shap_values import compute_shap_values
import numpy as np


def compute_shap_feature_importance(instances, model, x, is_categorical, *, method="kernel"):
    shap_values = compute_shap_values(
        model,
        instances,
        x,
        is_categorical,
        method=method,
    )
    feature_importances = np.abs(shap_values).mean(axis=0)
    return feature_importances
