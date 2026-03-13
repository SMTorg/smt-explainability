from .partial_dependence import partial_dependence

from typing import Union, Dict
import numpy as np


class PartialDependenceDisplay:
    def __init__(
        self,
        pd_results,
        *,
        features,
        feature_names,
        is_categorical,
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
        self.pd_results = pd_results
        self.features = features
        self.feature_names = feature_names
        self.is_categorical = is_categorical
        self.seed = seed

    @classmethod
    def from_surrogate_model(
        cls,
        model,
        x,
        features,
        *,
        sample_weight=None,
        categorical_feature_indices=None,
        feature_names=None,
        percentiles=(0.05, 0.95),
        grid_resolution=100,
        kind="average",
        ratio_samples=None,
        categories_map=None,
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

        pd_results = partial_dependence(
            model,
            x,
            features,
            sample_weight=sample_weight,
            categorical_feature_indices=categorical_feature_indices,
            percentiles=percentiles,
            grid_resolution=grid_resolution,
            kind=kind,
            ratio_samples=ratio_samples,
            categories_map=categories_map,
            seed=seed,
        )

        target_features = set()
        for feature in features:
            if isinstance(feature, int):
                target_features.add(feature)
            else:
                for f in feature:
                    target_features.add(f)

        # list to store the features are categorical or not in x
        is_categorical = [False] * x.shape[1]
        if categorical_feature_indices is not None:
            for feature_idx in categorical_feature_indices:
                is_categorical[feature_idx] = True

        display = PartialDependenceDisplay(
            pd_results,
            features=features,
            feature_names=feature_names,
            is_categorical=is_categorical,
            seed=seed,
        )
        return display

    def _plot_ice_lines(self, categorical, preds, feature_values, n_ice_to_plot, ax, individual_line_kw):
        # np.random.default_rng(None) handles using the global random state safely.
        rng = np.random.default_rng(self.seed)
        if categorical:
            medianprops = {
                "color": "black",
            }
            boxprops = {
                "facecolor": "None",
            }
            values = []
            for i in range(preds.shape[1]):
                values.append(preds[:, i])
            ax.boxplot(
                values,
                patch_artist=True,
                medianprops=medianprops,
                boxprops=boxprops,
            )
        else:
            # subsample ICE
            ice_lines_idx = rng.choice(preds.shape[0], n_ice_to_plot, replace=False)
            ice_lines_subsampled = preds[ice_lines_idx, :]
            # plot the subsampled ICE
            for ice_idx, ice in enumerate(ice_lines_subsampled):
                # line_idx = np.unravel_index(
                #     pd_plot_idx * n_total_lines_by_plot + ice_idx, self.lines_.shape
                # )
                # self.lines_[line_idx] = ax.plot(
                #     feature_values, ice.ravel(), **individual_line_kw
                # )[0]
                ax.plot(feature_values[0], ice.ravel(), **individual_line_kw)
            # individual_line_kw_label = individual_line_kw
            # individual_line_kw_label['label'] = "ICE"
            # ax.plot([], [], **individual_line_kw_label)

    @staticmethod
    def _plot_average_dependence(categorical, kind_plot, avg_preds, feature_values, ax, line_kw):
        # print(avg_preds, categorical, kind_plot)
        if categorical:
            if kind_plot == "both":
                ax.scatter(
                    np.arange(1, len(avg_preds) + 1),
                    avg_preds,
                    color="blue",
                    label="Average",
                )
            else:
                ax.bar(
                    np.arange(1, len(avg_preds) + 1),
                    avg_preds,
                    color="blue",
                    edgecolor="black",
                    linewidth=0.8,
                )
                ax.set_xticks(np.arange(1, len(avg_preds) + 1))
        else:
            ax.plot(feature_values[0], avg_preds, **line_kw)

    def _plot_one_way_partial_dependence(
        self,
        kind,
        categorical,
        preds,
        avg_preds,
        feature_values,
        feature_categories,
        feature_idx,
        n_ice_lines,
        ax,
        ice_lines_kw,
        pd_line_kw,
        legend_location,
    ):
        if kind in ["individual", "both"]:
            self._plot_ice_lines(categorical, preds, feature_values, n_ice_lines, ax, ice_lines_kw)

        if kind in ["average", "both"]:
            self._plot_average_dependence(categorical, kind, avg_preds.ravel(), feature_values, ax, pd_line_kw)

        if kind in ["individual", "both"]:
            max_val = preds.max()
            min_val = preds.min()
        else:
            max_val = avg_preds.max()
            min_val = avg_preds.min()
        max_val = max_val + 0.05 * (max_val - min_val)
        min_val = min_val - 0.05 * (max_val - min_val)
        ax.set_ylim([min_val, max_val])

        if self.feature_names is None:
            ax.set_xlabel(rf"$x_{feature_idx}$", fontsize=18)
        else:
            ax.set_xlabel(self.feature_names[feature_idx], fontsize=18)
        ax.xaxis.set_tick_params(labelsize=14)
        ax.yaxis.set_tick_params(labelsize=14)

        if not ax.get_ylabel():
            ax.set_ylabel("Partial dependence", fontsize=18)

        if categorical:
            ax.set_xticklabels(feature_categories[0])
        # if pd_line_kw.get("label", None) and kind != "individual" and not categorical:
        #     ax.legend()
        if kind == "both":
            ax.legend(
                # loc='upper left',
                loc=legend_location,
                fontsize=18,
            )
        ax.grid(
            color="black",
            alpha=0.2,
        )

    def _plot_two_way_partial_dependence(
        self,
        kind,
        categorical,
        avg_preds,
        feature_values,
        feature_categories,
        feature_idx,
        ax,
        z_level,
        contour_kw,
        heatmap_kw,
        annot_heatmap,
    ):
        if kind == "individual":
            pass
        else:
            if categorical:
                import matplotlib.pyplot as plt

                plt.rcParams.update(
                    {
                        "text.usetex": False,
                        "font.family": "serif",
                        "font.serif": "cmr10",
                        "axes.formatter.use_mathtext": True,
                    }
                )

                # default_im_kw = dict(interpolation="nearest", cmap="viridis")
                default_im_kw = dict(interpolation="nearest", cmap="Blues")
                im_kw = {**default_im_kw, **heatmap_kw}

                data = avg_preds
                im = ax.imshow(data, aspect="auto", **im_kw)
                cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)

                text = np.empty_like(data, dtype=object)
                # print text with appropriate color depending on background
                thresh = (data.max() + data.min()) / 2.0
                if annot_heatmap:
                    for flat_index in range(data.size):
                        row, col = np.unravel_index(flat_index, data.shape)
                        color = cmap_max if data[row, col] < thresh else cmap_min
                        values_format = ".1e"
                        text_data = format(data[row, col], values_format)

                        text_kwargs = dict(ha="center", va="center", color=color)
                        text[row, col] = ax.text(col, row, text_data, **text_kwargs)

                # fig = ax.figure
                # fig.colorbar(im, ax=ax)
                cbar = plt.colorbar(im, ax=ax)
                cbar.set_label("Partial dependence", fontsize=14)
                cbar.ax.tick_params(labelsize=12)

                if len(feature_categories[1]) > 0:
                    xticks = np.arange(len(feature_values[1]))
                    xticklabels = feature_categories[1]
                else:
                    n = 5
                    delta = max(1, len(feature_values[1]) // n)
                    xticks = np.arange(0, len(feature_values[1]), delta)
                    xticklabels = [feature_values[1][i] for i in xticks]
                    xticklabels = [f"{tick:.3f}" for tick in xticklabels]

                if len(feature_categories[0]) > 0:
                    yticks = np.arange(len(feature_values[0]))
                    yticklabels = feature_categories[0]
                else:
                    n = 5
                    delta = max(1, len(feature_values[0]) // n)
                    yticks = np.arange(0, len(feature_values[0]), delta)
                    yticklabels = [feature_values[0][i] for i in yticks]
                    yticklabels = [f"{tick:.3f}" for tick in yticklabels]

                ax.set(
                    xticks=xticks,
                    yticks=yticks,
                    xticklabels=xticklabels,
                    yticklabels=yticklabels,
                )

                if self.feature_names is not None:
                    xlabel = self.feature_names[feature_idx[1]]
                    ylabel = self.feature_names[feature_idx[0]]
                else:
                    xlabel = rf"$x_{feature_idx[1]}$"
                    ylabel = rf"$x_{feature_idx[0]}$"
                ax.set_xlabel(xlabel, fontsize=14)
                ax.set_ylabel(ylabel, fontsize=14)

                if len(feature_categories[1]) > 0:
                    plt.setp(ax.get_xticklabels(), rotation="vertical")
            else:
                xx, yy = np.meshgrid(feature_values[0], feature_values[1])
                z = avg_preds.T
                cs = ax.contour(xx, yy, z, levels=z_level, linewidths=0.5, colors="k")
                # contour_idx = np.unravel_index(pd_plot_idx, self.contours_.shape)
                ax.contourf(
                    xx,
                    yy,
                    z,
                    levels=z_level,
                    vmax=z_level[-1],
                    vmin=z_level[0],
                    **contour_kw,
                )
                ax.clabel(cs, fmt="%2.2f", colors="k", fontsize=12, inline=True)

                # create the decile line for the vertical axis
                if self.feature_names is None:
                    ax.set_xlabel(rf"$x_{feature_idx[0]}$", fontsize=14)
                    ax.set_ylabel(rf"$x_{feature_idx[1]}$", fontsize=14)
                else:
                    ax.set_xlabel(self.feature_names[feature_idx[0]], fontsize=14)
                    ax.set_ylabel(self.feature_names[feature_idx[1]], fontsize=14)
                ax.xaxis.set_tick_params(labelsize=12)
                ax.yaxis.set_tick_params(labelsize=12)

    def plot(
        self,
        *,
        n_cols=3,
        line_kw=None,
        ice_lines_kw=None,
        pd_line_kw=None,
        contour_kw=None,
        bar_kw=None,
        heatmap_kw=None,
        centered=False,
        pdp_lim=None,
        max_num_ice_lines=250,
        annot_heatmap=False,
        figsize=None,
        legend_locations: Union[str, Dict] = "best",
    ):
        import matplotlib.pyplot as plt
        from matplotlib.gridspec import GridSpecFromSubplotSpec

        plt.rcParams.update(
            {
                "text.usetex": False,
                "font.family": "serif",
                "font.serif": "cmr10",
                "axes.formatter.use_mathtext": True,
            }
        )

        if isinstance(legend_locations, str):
            legend_locations_ = [legend_locations] * len(self.pd_results)
        elif isinstance(legend_locations, dict):
            legend_locations_ = ["best"] * len(self.pd_results)
            for i in legend_locations:
                legend_locations_[i] = legend_locations[i]
        else:
            raise TypeError("Wrong type of legend locations. It must be string or dictionary.")

        kind = []
        for pd_result in self.pd_results:
            keys = pd_result.keys()
            if (len(pd_result["grid_values"]) > 1) & ("average" in keys):
                kind.append("average")
            else:
                if ("average" in keys) and ("individual" in keys):
                    kind.append("both")
                elif ("average" in keys) and ("individual" not in keys):
                    kind.append("average")
                else:
                    kind.append("individual")

        n_results = len(self.pd_results)
        _, ax = plt.subplots()

        if not centered:
            pd_results_ = self.pd_results
        else:
            pd_results_ = []
            for kind_plot, pd_result in zip(kind, self.pd_results):
                current_results = {"grid_values": pd_result["grid_values"]}

                if "grid_categories" in pd_result:
                    current_results["grid_categories"] = pd_result["grid_categories"]

                if kind_plot in ("individual", "both"):
                    preds = pd_result["individual"]
                    preds = preds - preds[:, 0, None]
                    current_results["individual"] = preds

                if kind_plot in ("average", "both"):
                    avg_preds = pd_result["average"]
                    avg_preds = avg_preds - avg_preds[0, None]
                    current_results["average"] = avg_preds

                pd_results_.append(current_results)

        if pdp_lim is None:
            pdp_lim = {}
            for kind_plot, pd_result in zip(kind, pd_results_):
                values = pd_result["grid_values"]
                preds = pd_result["average"] if kind_plot == "average" else pd_result["individual"]
                min_pd = preds.min()
                max_pd = preds.max()

                # expand the limits to account so that the plotted lines do not touch
                # the edges of the plot
                span = max_pd - min_pd
                min_pd -= 0.05 * span
                max_pd += 0.05 * span

                n_features = len(values)
                old_min_pd, old_max_pd = pdp_lim.get(n_features, (min_pd, max_pd))
                min_pd = min(min_pd, old_min_pd)
                max_pd = max(max_pd, old_max_pd)
                pdp_lim[n_features] = (min_pd, max_pd)

        contains_categories = []
        for feature in self.features:
            if isinstance(feature, int):
                contains_categories.append(self.is_categorical[feature])
            else:
                contains_categories_ = [self.is_categorical[f] for f in feature]
                contains_categories.append(np.max(contains_categories_) == 1)

        if line_kw is None:
            line_kw = {}
        if ice_lines_kw is None:
            ice_lines_kw = {}
        if pd_line_kw is None:
            pd_line_kw = {}
        if bar_kw is None:
            bar_kw = {}
        if heatmap_kw is None:
            heatmap_kw = {}
        if contour_kw is None:
            contour_kw = {}
        default_contour_kws = {"alpha": 0.75}
        contour_kw = {**default_contour_kws, **contour_kw}

        is_average_plot = [kind_plot == "average" for kind_plot in kind]
        if all(is_average_plot):
            # only average plots are requested
            n_ice_lines = 0
        else:
            # we need to determine the number of ICE samples computed
            ice_plot_idx = is_average_plot.index(False)
            n_ice_lines = pd_results_[ice_plot_idx]["individual"].shape[0]
            n_ice_lines = min(n_ice_lines, max_num_ice_lines)

        ax.set_axis_off()
        # bounding_ax_ = ax
        fig = ax.figure

        n_cols = min(n_cols, n_results)
        n_rows = int(np.ceil(n_results / float(n_cols)))
        # self.axes_ = np.empty((n_rows, n_cols), dtype=object)
        axes_ = np.empty((n_rows, n_cols), dtype=object)
        if figsize is None:
            fig.set_size_inches(n_cols * 7, n_rows * 5)
        else:
            fig.set_size_inches(figsize[0], figsize[1])

        # axes_ravel = self.axes_.ravel()
        axes_ravel = axes_.ravel()
        gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=ax.get_subplotspec())
        # for i, spec in zip(range(n_results), gs):
        #     axes_ravel[i] = fig.add_subplot(spec)
        for i in range(n_results):
            axes_ravel[i] = fig.add_subplot(gs[i])

        # create contour levels for two-way plots
        if 2 in pdp_lim:
            z_level = np.linspace(*pdp_lim[2], num=8)
        else:
            z_level = None

        for plot_idx, (axi, pd_result, kind_plot, feature_idx, cat) in enumerate(
            zip(
                axes_.ravel(),
                pd_results_,
                kind,
                self.features,
                contains_categories,
            )
        ):
            # print(feature_idx)
            avg_preds = None
            preds = None
            feature_values = pd_result["grid_values"]
            feature_categories = None
            if cat:
                feature_categories = pd_result["grid_categories"]

            if kind_plot == "individual":
                preds = pd_result["individual"]
            elif kind_plot == "average":
                avg_preds = pd_result["average"]
            else:  # kind_plot == "both"
                preds = pd_result["individual"]
                avg_preds = pd_result["average"]

            legend_location = legend_locations_[plot_idx]

            if len(feature_values) == 1:
                # define the line-style for the current plot
                # default_line_kws = {
                #     "color": "C0",
                #     "label": "average" if kind_plot == "both" else None,
                # }
                default_line_kws = {
                    "color": "C0",
                    "label": "Average" if kind_plot == "both" else None,
                }
                if kind_plot == "individual":
                    default_ice_lines_kws = {
                        "alpha": 0.3,
                        "linewidth": 0.5,
                    }
                    default_pd_lines_kws = {}
                elif kind_plot == "both":
                    # by default, we need to distinguish the average line from
                    # the individual lines via color and line style
                    default_ice_lines_kws = {
                        "alpha": 0.3,
                        "linewidth": 0.5,
                        "color": "tab:gray",
                        "label": r"ICE",
                    }
                    default_pd_lines_kws = {
                        "color": "blue",
                        "linestyle": "--",
                        "linewidth": 2.5,
                        "label": r"Average",
                    }
                else:
                    default_ice_lines_kws = {}
                    default_pd_lines_kws = {
                        "color": "blue",
                        "linewidth": 2.5,
                    }

                ice_lines_kw = {
                    **default_line_kws,
                    **default_ice_lines_kws,
                    **line_kw,
                    **ice_lines_kw,
                }
                del ice_lines_kw["label"]

                pd_line_kw = {
                    **default_line_kws,
                    **default_pd_lines_kws,
                    **line_kw,
                    **pd_line_kw,
                }

                default_bar_kws = {"color": "C0"}
                bar_kw = {**default_bar_kws, **bar_kw}

                default_heatmap_kw = {}
                heatmap_kw = {**default_heatmap_kw, **heatmap_kw}

                self._plot_one_way_partial_dependence(
                    kind_plot,
                    cat,
                    preds,
                    avg_preds,
                    feature_values,
                    feature_categories,
                    feature_idx,
                    n_ice_lines,
                    axi,
                    ice_lines_kw,
                    pd_line_kw,
                    legend_location,
                )

            else:
                self._plot_two_way_partial_dependence(
                    kind_plot,
                    cat,
                    avg_preds,
                    feature_values,
                    feature_categories,
                    feature_idx,
                    axi,
                    z_level,
                    contour_kw,
                    heatmap_kw,
                    annot_heatmap,
                )

        return fig
