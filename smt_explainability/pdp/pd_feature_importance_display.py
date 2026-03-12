from .pd_feature_importance import pd_feature_importance
import numpy as np


class PDFeatureImportanceDisplay:
    def __init__(self, feature_importances, feature_names):
        self.feature_importances = feature_importances
        self.feature_names = feature_names

    @classmethod
    def from_surrogate_model(
        cls,
        model,
        x,
        *,
        features=None,
        feature_names=None,
        sample_weight=None,
        categorical_feature_indices=None,
        percentiles=(0.05, 0.95),
        grid_resolution=100,
        method="uniform",
        ratio_samples=None,
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

        feature_importances = pd_feature_importance(
            model,
            x,
            features,
            sample_weight=sample_weight,
            categorical_feature_indices=categorical_feature_indices,
            percentiles=percentiles,
            grid_resolution=grid_resolution,
            method=method,
            ratio_samples=ratio_samples,
            seed=seed,
        )
        display = PDFeatureImportanceDisplay(feature_importances, feature_names)
        return display

    def plot(
        self,
        *,
        sort=False,
        figsize=None,
    ):
        import matplotlib.pyplot as plt

        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": "cmr10",
                "axes.formatter.use_mathtext": True,
            }
        )

        if figsize is None:
            length = max(5, int(len(self.feature_importances) * 0.6))
            width = 4
        else:
            length = figsize[0]
            width = figsize[1]

        feature_names = np.array(self.feature_names)
        feature_importances = np.array(self.feature_importances)

        if sort:
            vis_feature_names = feature_names[np.argsort(feature_importances * -1)]
            vis_feature_importances = feature_importances[np.argsort(feature_importances * -1)]
        else:
            vis_feature_names = feature_names
            vis_feature_importances = feature_importances

        indexes = np.arange(len(vis_feature_importances))
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
