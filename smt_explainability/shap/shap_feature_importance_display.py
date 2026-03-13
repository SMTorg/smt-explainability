from .shap_feature_importance import compute_shap_feature_importance
import numpy as np


class ShapFeatureImportanceDisplay:
    def __init__(self, feature_importances, feature_names):
        self.feature_importances = feature_importances
        self.feature_names = feature_names

    @classmethod
    def from_surrogate_model(
        cls,
        model,
        x,
        *,
        method="kernel",
        features=None,
        feature_names=None,
        categorical_feature_indices=None,
        seed=None,
        random_state=None,
    ):
        if random_state is not None:
            import warnings

            warnings.warn(
                "random_state is deprecated, use seed instead",
                DeprecationWarning,
                stacklevel=2,
            )
            seed = seed or random_state

        if features is None:
            features = [i for i in range(x.shape[1])]

        if feature_names is None:
            feature_names = [rf"$x_{i}$" for i in features]
        elif len(feature_names) != x.shape[1]:
            raise ValueError("Length of feature names is not the same as the number of dimensions in x.")

        if len(features) <= x.shape[1]:
            feature_names = [feature_names[feature_idx] for feature_idx in features]
        else:
            raise ValueError("Length of features exceed number of dimensions in x.")

        num_features = x.shape[1]
        # boolean flags for categorical variable indicator
        is_categorical = [False] * num_features
        if categorical_feature_indices is not None:
            for feature_idx in categorical_feature_indices:
                is_categorical[feature_idx] = True
        # compute feature importance
        feature_importances = compute_shap_feature_importance(
            x,
            model,
            x,
            is_categorical,
            method=method,
            seed=seed,
        )
        feature_importances = np.array([feature_importances[feature_idx] for feature_idx in features])

        display = ShapFeatureImportanceDisplay(feature_importances, feature_names)
        return display

    def plot(self, *, figsize=None, sort=False):
        import matplotlib.pyplot as plt

        # from matplotlib.ticker import ScalarFormatter
        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": "cmr10",
                "axes.formatter.use_mathtext": True,
            }
        )

        num_features = len(self.feature_importances)
        feature_names = np.array(self.feature_names)
        feature_importances = np.array(self.feature_importances)

        if figsize is None:
            length = max(5, int(num_features * 0.6))
            width = 4
        else:
            length = figsize[0]
            width = figsize[1]

        if sort:
            vis_feature_names = feature_names[np.argsort(feature_importances * -1)]
            vis_feature_importances = feature_importances[np.argsort(feature_importances * -1)]
        else:
            vis_feature_names = feature_names
            vis_feature_importances = feature_importances

        indexes = np.arange(num_features)
        fig, ax = plt.subplots(1, 1, figsize=(length, width))
        ax.bar(
            indexes,
            vis_feature_importances,
            color="blue",
            edgecolor="black",
            linewidth=0.8,
        )
        ax.set_xticks(indexes)
        ax.set_xticklabels(vis_feature_names, fontsize=14)
        ax.set_ylabel("Feature Importance", fontsize=14)
        ax.grid(color="black", alpha=0.2)
        ax.yaxis.set_tick_params(labelsize=14)
        ax.set_axisbelow(True)
        fig.tight_layout()

        return fig
