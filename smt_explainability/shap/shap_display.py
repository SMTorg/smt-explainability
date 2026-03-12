from .shap_values import compute_shap_values

from typing import Union
from matplotlib.ticker import ScalarFormatter
import numpy as np
import matplotlib as mpl
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

NOMINAL_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
]


class ShapDisplay:
    def __init__(
        self,
        instances,
        shap_values,
        is_categorical,
        *,
        feature_names=None,
        categories_map=None,
    ):
        if feature_names is None:
            num_features = shap_values.shape[1]
            feature_names = [rf"$x_{i}$" for i in range(num_features)]

        self.instances = instances
        self.shap_values = shap_values
        self.feature_names = feature_names
        self.is_categorical = is_categorical
        self.categories_map = categories_map

    @classmethod
    def from_surrogate_model(
        cls,
        instances,
        model,
        x,
        *,
        method="kernel",
        feature_names=None,
        categorical_feature_indices=None,
        categories_map=None,
    ):
        num_features = x.shape[1]

        # boolean flags for categorical variable indicator
        is_categorical = [False] * num_features
        if categorical_feature_indices is not None:
            for feature_idx in categorical_feature_indices:
                is_categorical[feature_idx] = True

        shap_values = compute_shap_values(
            model,
            instances,
            x,
            is_categorical,
            method=method,
        )
        display = ShapDisplay(
            instances,
            shap_values,
            is_categorical,
            feature_names=feature_names,
            categories_map=categories_map,
        )
        return display

    def individual_plot(
        self,
        *,
        index=None,
        figsize=None,
    ):
        num_features = self.shap_values.shape[1]
        feature_names = self.feature_names
        is_categorical = self.is_categorical
        categories_map = self.categories_map

        if len(self.instances) > 1 and index is None:
            raise ValueError

        if index is not None:
            instance = self.instances[index]
            shap_values = self.shap_values[index]
        else:
            instance = self.instances
            shap_values = self.shap_values
        instance = instance.reshape(
            -1,
        )
        shap_values = shap_values.reshape(
            -1,
        )

        if figsize is None:
            length = 6
            width = max(4, int(0.4 * num_features))
        else:
            length = figsize[0]
            width = figsize[1]

        ticks = list()
        for feature_index in range(num_features):
            feature_name = feature_names[feature_index]
            value = instance[feature_index]
            if is_categorical[feature_index] == 1:
                if categories_map is not None:
                    cat_value = categories_map[feature_index][value]
                    tick = f"{feature_name}: {cat_value}"
                else:
                    tick = f"{feature_name}: {value:.3f}"
            else:
                tick = f"{feature_name}: {value:.3f}"
            ticks.append(tick)

        ticks = np.array(ticks)

        ticks = ticks[np.argsort(shap_values)]
        shap_values = shap_values[np.argsort(shap_values)]
        colors = ["blue" if value >= 0 else "red" for value in shap_values]

        fig, ax = plt.subplots(1, 1, figsize=(length, width))
        ax.barh(np.arange(num_features), shap_values, color=colors)
        ax.set_yticks(np.arange(num_features))
        ax.set_yticklabels(ticks)
        ax.set_xlabel("SHAP Value", fontsize=14)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=14)
        formatter = ScalarFormatter()
        formatter.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(formatter)
        fig.tight_layout()

        return fig

    def dependence_plot(
        self,
        features=None,
        *,
        n_cols=3,
        figsize=None,
        sort_based_on_importance: Union[bool, dict] = True,
        max_num_entities: Union[int, dict] = 15,
        selected_entities=None,
        selected_entity_values=None,
    ):
        is_categorical = self.is_categorical
        categories_map = self.categories_map

        # create arguments for dependence plot with categorical variables
        categories_args = dict()
        for feature_id in range(len(is_categorical)):
            if is_categorical[feature_id] == 1:
                category_args = dict()
                if isinstance(sort_based_on_importance, dict):
                    if feature_id in sort_based_on_importance:
                        category_args["sort_based_on_importance"] = sort_based_on_importance[feature_id]
                    else:
                        category_args["sort_based_on_importance"] = True
                elif isinstance(sort_based_on_importance, bool):
                    category_args["sort_based_on_importance"] = sort_based_on_importance
                else:
                    raise ValueError

                if isinstance(max_num_entities, dict):
                    if feature_id in max_num_entities:
                        category_args["max_num_entities"] = max_num_entities[feature_id]
                    else:
                        category_args["max_num_entities"] = 15
                elif isinstance(max_num_entities, int):
                    category_args["max_num_entities"] = max_num_entities
                else:
                    raise ValueError

                if isinstance(selected_entities, dict):
                    if feature_id in selected_entities:
                        category_args["selected_entities"] = selected_entities[feature_id]
                    else:
                        category_args["selected_entities"] = None
                else:
                    category_args["selected_entities"] = None

                if isinstance(selected_entity_values, dict):
                    if feature_id in selected_entity_values:
                        category_args["selected_entity_values"] = selected_entity_values[feature_id]
                    else:
                        category_args["selected_entity_values"] = None
                else:
                    category_args["selected_entity_values"] = None

                categories_args[feature_id] = category_args

        n_rows = int(np.ceil(len(features) / n_cols))
        # fig, axs = plt.subplots(n_rows, n_cols, figsize=(length, width))
        # delete empty subplot
        # fig = delete_empty_axis(len(features), fig, axs)

        _, axs = plt.subplots()
        axs.set_axis_off()
        axes_ = np.empty((n_rows, n_cols), dtype=object)
        figure_ = axs.figure
        if figsize is None:
            figure_.set_size_inches(n_cols * 7, n_rows * 5)
        else:
            figure_.set_size_inches(figsize[0], figsize[1])

        axes_ravel = axes_.ravel()
        gs = GridSpecFromSubplotSpec(n_rows, n_cols, subplot_spec=axs.get_subplotspec())

        for i, spec in zip(range(len(features)), gs):
            axes_ravel[i] = figure_.add_subplot(spec)

        formatter = ScalarFormatter()
        formatter.set_powerlimits((-3, 3))

        for i, feature_idx in enumerate(features):
            # row = i // n_cols
            col = i % n_cols
            # if n_rows == 1:
            #     ax = axs[col]
            # else:
            #     ax = axs[row, col]
            ax = axes_ravel[i]

            if is_categorical[feature_idx] == 1:
                categorical_dependence_plot(
                    feature_idx,
                    ax,
                    self.instances,
                    self.shap_values,
                    categories_args,
                    categories_map,
                )
                ax.yaxis.set_major_formatter(formatter)
            else:
                numeric_dependence_plot(
                    feature_idx,
                    ax,
                    self.instances,
                    self.shap_values,
                )
                ax.yaxis.set_major_formatter(formatter)
                ax.xaxis.set_major_formatter(formatter)

            ax.set_xlabel(self.feature_names[feature_idx], fontsize=18)
            if col == 0:
                ax.set_ylabel("SHAP Value", fontsize=18)
            ax.xaxis.set_tick_params(labelsize=14)
            ax.yaxis.set_tick_params(labelsize=14)
            ax.grid(color="black", alpha=0.2)

        figure_.tight_layout()

        return figure_

    def interaction_plot(
        self,
        feature_pairs,
        *,
        n_cols=3,
        n_color=10,
        figsize=None,
        sort_based_on_importance: Union[bool, dict] = True,
        max_num_entities: Union[int, dict] = 15,
        selected_entities=None,
        selected_entity_values=None,
    ):
        is_categorical = self.is_categorical
        categories_map = self.categories_map

        # create arguments for interaction plot with categorical variables
        categories_args = dict()
        for feature_id in range(len(is_categorical)):
            if is_categorical[feature_id] == 1:
                category_args = dict()

                if isinstance(sort_based_on_importance, dict):
                    if feature_id in sort_based_on_importance:
                        category_args["sort_based_on_importance"] = sort_based_on_importance[feature_id]
                    else:
                        category_args["sort_based_on_importance"] = True
                elif isinstance(sort_based_on_importance, bool):
                    category_args["sort_based_on_importance"] = sort_based_on_importance
                else:
                    raise ValueError

                if isinstance(max_num_entities, dict):
                    if feature_id in max_num_entities:
                        category_args["max_num_entities"] = max_num_entities[feature_id]
                    else:
                        category_args["max_num_entities"] = 15
                elif isinstance(max_num_entities, int):
                    category_args["max_num_entities"] = max_num_entities
                else:
                    raise ValueError

                if isinstance(selected_entities, dict):
                    if feature_id in selected_entities:
                        category_args["selected_entities"] = selected_entities[feature_id]
                    else:
                        category_args["selected_entities"] = None
                else:
                    category_args["selected_entities"] = None

                if isinstance(selected_entity_values, dict):
                    if feature_id in selected_entity_values:
                        category_args["selected_entity_values"] = selected_entity_values[feature_id]
                    else:
                        category_args["selected_entity_values"] = None
                else:
                    category_args["selected_entity_values"] = None

                categories_args[feature_id] = category_args

        n_rows = int(np.ceil(len(feature_pairs) / n_cols))
        if figsize is None:
            length = n_cols * 7
            width = n_rows * 5
        else:
            length = figsize[0]
            width = figsize[1]

        fig, axs = plt.subplots(n_rows, n_cols, figsize=(length, width))
        # delete empty subplot
        fig = delete_empty_axis(len(feature_pairs), fig, axs)
        formatter = ScalarFormatter()
        formatter.set_powerlimits((-3, 3))

        for i in range(len(feature_pairs)):
            feature_i = feature_pairs[i][0]
            feature_j = feature_pairs[i][1]
            row = i // n_cols
            col = i % n_cols

            if n_rows == 1:
                ax = axs[col]
            else:
                ax = axs[row, col]

            if is_categorical[feature_j] == 0:
                numeric_interaction_plot(
                    feature_i,
                    feature_j,
                    ax,
                    self.instances,
                    self.shap_values,
                    self.feature_names,
                    n_color,
                )
            else:
                categorical_interaction_plot(
                    feature_i,
                    feature_j,
                    ax,
                    self.instances,
                    self.shap_values,
                    self.feature_names,
                    categories_args,
                    categories_map,
                )

            ax.yaxis.set_major_formatter(formatter)
            ax.xaxis.set_major_formatter(formatter)
            ax.set_ylabel("SHAP Value", fontsize=14)
            ax.xaxis.set_tick_params(labelsize=12)
            ax.yaxis.set_tick_params(labelsize=12)
            ax.grid(color="black", alpha=0.2)

        fig.tight_layout()

        return fig

    def summary_plot(
        self,
        *,
        figsize=None,
        n_color=10,
        include_cat=False,
        max_num_features=10,
    ):
        num_features = self.shap_values.shape[1]
        feature_names = np.array(self.feature_names)
        feature_indexes = np.arange(num_features)

        # set figsize
        if figsize is None:
            length = 6
            width = max(4, int(0.4 * num_features))
        else:
            length = figsize[0]
            width = figsize[1]

        feature_importance = np.abs(self.shap_values).mean(axis=0)
        sorted_feature_indexes = feature_indexes[np.argsort(feature_importance * -1)]

        # filter categorical variables according to include_cat param
        if include_cat:
            selected_feature_indexes = sorted_feature_indexes
        else:
            selected_feature_indexes = list()
            for i, feature_index in enumerate(sorted_feature_indexes):
                if self.is_categorical[feature_index] == 0:
                    selected_feature_indexes.append(feature_index)
            selected_feature_indexes = np.array(selected_feature_indexes)

        # filter based on max num features
        vis_feature_indexes = selected_feature_indexes[:max_num_features]
        other_feature_indexes = list()
        for feature_idx in selected_feature_indexes:
            if feature_idx not in vis_feature_indexes:
                other_feature_indexes.append(feature_idx)

        cmap = plt.cm.plasma  # noqa
        cmaplist = [cmap(i) for i in range(cmap.N)]
        # create the new map
        cmap = mpl.colors.LinearSegmentedColormap.from_list("color_map", cmaplist, cmap.N)
        bounds = np.linspace(0, 1, n_color + 1)
        norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        x_vals = list()
        y_vals = list()
        color_flags = list()
        y_ticklabels = list()
        vis_index = 0
        for i in range(len(vis_feature_indexes)):
            feature_idx = vis_feature_indexes[i]
            min_feature_value = self.instances[:, feature_idx].min()
            max_feature_value = self.instances[:, feature_idx].max()
            range_feature_value = max_feature_value - min_feature_value
            color_flags_ = (self.instances[:, feature_idx] - min_feature_value) / range_feature_value

            x_vals_ = self.shap_values[:, feature_idx]
            y_vals_ = jitter_y_based_on_x(
                x_vals_,
                [vis_index] * len(x_vals_),
                20,
                self.shap_values.min(),
                self.shap_values.max(),
                max_strength=0.1,
            )

            x_vals.append(x_vals_)
            y_vals.append(y_vals_)
            color_flags.append(color_flags_)
            y_ticklabels.append(feature_names[feature_idx])
            vis_index -= 1

        if len(other_feature_indexes) > 0:
            feature_idx = other_feature_indexes[0]
            min_feature_value = self.instances[:, feature_idx].min()
            max_feature_value = self.instances[:, feature_idx].max()
            range_feature_value = max_feature_value - min_feature_value
            color_flags_ = (self.instances[:, feature_idx] - min_feature_value) / range_feature_value
            x_vals_ = self.shap_values[:, feature_idx]
            for feature_idx in other_feature_indexes[1:]:
                min_feature_value = self.instances[:, feature_idx].min()
                max_feature_value = self.instances[:, feature_idx].max()
                range_feature_value = max_feature_value - min_feature_value

                color_flags_ = np.concatenate(
                    [
                        color_flags_,
                        (self.instances[:, feature_idx] - min_feature_value) / range_feature_value,
                    ]
                )
                x_vals_ = np.concatenate(
                    [
                        x_vals_,
                        self.shap_values[:, feature_idx],
                    ]
                )

            y_vals_ = jitter_y_based_on_x(
                x_vals_,
                [vis_index] * len(x_vals_),
                20,
                self.shap_values.min(),
                self.shap_values.max(),
                max_strength=0.1,
            )
            x_vals.append(x_vals_)
            y_vals.append(y_vals_)
            color_flags.append(color_flags_)
            y_ticklabels.append("Others")

        fig, ax = plt.subplots(1, 1, figsize=(length, width))
        im = None
        num_vis_features = len(x_vals)
        vis_indexes = np.arange(0, -1 * num_vis_features, -1)
        for i in range(num_vis_features):
            im = ax.scatter(
                x_vals[i],
                y_vals[i],
                c=color_flags[i],
                # cmap=cmap,
                s=8,
                norm=norm,
            )

        ax.axvline(0, color="black")
        ax.set_yticks(vis_indexes)
        ax.set_yticklabels(y_ticklabels)
        ax.set_xlabel("SHAP Values", fontsize=14)
        ax.grid(color="black", alpha=0.2)
        formatter = ScalarFormatter()
        formatter.set_powerlimits((-3, 3))
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_tick_params(labelsize=12)
        ax.yaxis.set_tick_params(labelsize=13)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label("Feature values", fontsize=12)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"], fontsize=12)

        fig.tight_layout()

        return fig


def jitter_y_based_on_x(x, y, num_grid, min_value, max_value, max_strength=0.2):
    delta = (max_value - min_value) / num_grid
    groups = x // delta
    prop_groups = dict()
    for value in np.unique(groups):
        prop_groups[value] = np.mean(groups == value)

    jitter_scales = np.ones(len(y))
    for i in range(len(groups)):
        if prop_groups[groups[i]] < 2 / num_grid:
            jitter_scale = 0
        else:
            jitter_scale = (prop_groups[groups[i]] - 0.1) * max_strength
            jitter_scale = min(max_strength, jitter_scale)
        jitter_scales[i] = jitter_scale

    y = np.array(y)
    y_jitter = y + np.random.randn(len(y)) * jitter_scales
    return y_jitter


def numeric_dependence_plot(feature_idx, ax, instances, shap_values):
    ax.scatter(
        instances[:, feature_idx],
        shap_values[:, feature_idx],
        color="blue",
        s=5,
    )


def categorical_dependence_plot(
    feature_idx,
    ax,
    instances,
    shap_values,
    categories_args,
    categories_map,
):
    medianprops = {"color": "black"}
    boxprops = {"facecolor": "None"}

    selected_entities_ = categories_args[feature_idx]["selected_entities"]
    selected_entity_values = categories_args[feature_idx]["selected_entity_values"]
    max_num_entities = categories_args[feature_idx]["max_num_entities"]
    sort_based_on_importance = categories_args[feature_idx]["sort_based_on_importance"]

    unique_entities = np.unique(instances[:, feature_idx])
    unique_entities = np.sort(unique_entities)

    if selected_entities_ is not None:
        selected_entities = np.array(selected_entities_)
    elif selected_entity_values is not None:
        selected_entities = [get_key(value, categories_map[feature_idx]) for value in selected_entity_values]
        selected_entities = np.array(selected_entities)
    else:
        selected_entities = unique_entities

    entity_importance = np.zeros(len(selected_entities))
    for i, entity in enumerate(selected_entities):
        shap_values_ = shap_values[instances[:, feature_idx] == entity][:, feature_idx]
        entity_importance[i] = np.abs(np.mean(shap_values_))

    if sort_based_on_importance:
        selected_entities = selected_entities[np.argsort(entity_importance * -1)]

    if (selected_entities_ is None) and (selected_entity_values is None):
        selected_entities = selected_entities[:max_num_entities]

    other_entities = np.array([e for e in unique_entities if e not in selected_entities])

    disp_shap_values = list()
    avg_shap_values = list()
    xtick_labels = list()
    for i, entity in enumerate(selected_entities):
        shap_values_ = shap_values[instances[:, feature_idx] == entity][:, feature_idx]
        disp_shap_values.append(shap_values_)
        avg_shap_values.append(np.mean(shap_values_))
        if categories_map is None:
            xtick_labels.append(entity)
        else:
            xtick_labels.append(categories_map[feature_idx][entity])

    if len(other_entities) > 0:
        other_shap_values = shap_values[instances[:, feature_idx] == other_entities[0]][:, feature_idx]
        for entity in other_entities[1:]:
            other_shap_values = np.concatenate(
                [
                    other_shap_values,
                    shap_values[instances[:, feature_idx] == entity][:, feature_idx],
                ]
            )
        disp_shap_values.append(other_shap_values)
        avg_shap_values.append(np.mean(other_shap_values))
        xtick_labels.append("Others")

    ax.boxplot(
        disp_shap_values,
        patch_artist=True,
        medianprops=medianprops,
        boxprops=boxprops,
    )
    ax.scatter(
        np.arange(1, len(avg_shap_values) + 1),
        avg_shap_values,
        color="blue",
        label="Average",
    )
    ax.set_xticklabels(xtick_labels)
    ax.legend(fontsize=18)


def numeric_interaction_plot(
    feature_i,
    feature_j,
    ax,
    instances,
    shap_values,
    feature_names,
    n_color,
):
    feature_i_values = instances[:, feature_i]
    feature_j_values = instances[:, feature_j]

    min_feature_j = feature_j_values.min()
    max_feature_j = feature_j_values.max()
    cmap = plt.cm.plasma  # noqa
    cmaplist = [cmap(i) for i in range(cmap.N)]
    # create the new map
    cmap = mpl.colors.LinearSegmentedColormap.from_list("color_map", cmaplist, cmap.N)
    bounds = np.linspace(min_feature_j, max_feature_j, n_color + 1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.scatter(
        feature_i_values,
        shap_values[:, feature_i],
        s=5,
        c=feature_j_values,
        norm=norm,
    )
    cbar = plt.colorbar(im, ax=ax, format=lambda x, _: f"{x:.3f}")
    cbar.set_label(feature_names[feature_j], fontsize=14)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel(feature_names[feature_i], fontsize=14)


def categorical_interaction_plot(
    feature_i,
    feature_j,
    ax,
    instances,
    shap_values,
    feature_names,
    categories_args,
    categories_map,
):
    feature_i_values = instances[:, feature_i]
    feature_j_values = instances[:, feature_j]
    feature_j_values = feature_j_values.astype(int)

    selected_entities_ = categories_args[feature_j]["selected_entities"]
    selected_entity_values = categories_args[feature_j]["selected_entity_values"]
    max_num_entities = categories_args[feature_j]["max_num_entities"]
    sort_based_on_importance = categories_args[feature_j]["sort_based_on_importance"]

    unique_entities = np.unique(feature_j_values)
    unique_entities = np.sort(unique_entities)

    if selected_entities_ is not None:
        selected_entities = np.array(selected_entities_)
    elif selected_entity_values is not None:
        selected_entities = [get_key(value, categories_map[feature_j]) for value in selected_entity_values]
        selected_entities = np.array(selected_entities)
    else:
        selected_entities = unique_entities

    entity_importance = np.zeros(len(selected_entities))
    for entity_id, entity in enumerate(selected_entities):
        importance = np.abs(shap_values[:, feature_j][feature_j_values == entity]).mean()
        entity_importance[entity_id] = importance

    if sort_based_on_importance:
        selected_entities = selected_entities[np.argsort(entity_importance * -1)]

    if (selected_entities_ is None) and (selected_entity_values is None):
        selected_entities = selected_entities[:max_num_entities]

    other_entities = np.array([e for e in unique_entities if e not in selected_entities])

    entities_color_dict = dict()
    handles = list()
    for j, entity in enumerate(selected_entities):
        entities_color_dict[entity] = NOMINAL_COLORS[j]
        if categories_map is not None:
            entity_name = categories_map[feature_j][entity]
        else:
            entity_name = entity
        handles.append(mpl.patches.Patch(color=NOMINAL_COLORS[j], label=entity_name))
    for entity in other_entities:
        entities_color_dict[entity] = "black"
    if len(other_entities) > 0:
        handles.append(mpl.patches.Patch(color="black", label="Others"))

    colors = [entities_color_dict[entity] for entity in feature_j_values]
    colors = np.array(colors)

    if len(other_entities) > 0:
        ax.scatter(
            feature_i_values[colors == "black"],
            shap_values[:, feature_i][colors == "black"],
            c="black",
            s=5,
        )
    ax.scatter(
        feature_i_values[colors != "black"],
        shap_values[:, feature_i][colors != "black"],
        c=colors[colors != "black"],
        s=5,
    )
    ax.legend(
        handles=handles,
        title=feature_names[feature_j],
        ncol=3,
        title_fontsize=12,
        fontsize=10,
    )
    ax.set_xlabel(feature_names[feature_i], fontsize=14)


def get_key(val, dictionary):
    for key, value in dictionary.items():
        if val == value:
            return key
    raise KeyError


def delete_empty_axis(num_plots, fig, axs):
    if len(axs.shape) == 1:
        n_rows = 1
        n_cols = axs.shape[0]
    else:
        n_rows = axs.shape[0]
        n_cols = axs.shape[1]

    for i in range(num_plots, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        if n_rows == 1:
            ax = axs[col]
        else:
            ax = axs[row, col]
        fig.delaxes(ax=ax)
    return fig
