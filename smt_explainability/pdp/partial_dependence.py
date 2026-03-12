"""Partial dependence for classification and regression models"""

# Authors: Muhammad Daffa Robani
import numpy as np
from scipy.stats.mstats import mquantiles
from typing import Union, List, Tuple


def grid_from_x(
    x,
    features,
    percentiles,
    grid_resolution,
    is_categorical,
    method,
):
    if isinstance(features, int):
        features = tuple([features])

    grid_values = []
    for i in features:
        if method == "sample":
            axis = x[:, i]
        elif method == "unique" or is_categorical[i]:
            axis = np.unique(x[:, i])
        elif method == "uniform":
            emp_percentiles = mquantiles(x[:, i], prob=percentiles, axis=0)
            axis = np.linspace(
                emp_percentiles[0],
                emp_percentiles[1],
                num=grid_resolution,
                endpoint=True,
            )
        else:
            raise ValueError("Method must be 'sample' / 'unique' / 'uniform'.")

        grid_values.append(axis)

    if method in ["unique", "uniform"]:
        grid_2d = cartesian(grid_values)
    else:
        grid_2d = non_cartesian(grid_values)
    return grid_2d, grid_values


def _partial_dependence_brute(
    model,
    grid_cartesian,
    grid_values,
    features,
    x,
    method,
    sample_weight=None,
    ratio_samples=None,
    seed=None,
):
    if isinstance(features, int):
        features = tuple([features])

    nsamp = len(x)
    lengths = [len(grid_value) for grid_value in grid_values]
    predictions = []
    averaged_predictions = []
    rng = np.random.default_rng(seed)

    if ratio_samples is None:
        nsamples = nsamp
        x_samp = x.copy()
    else:
        nsamples = int(ratio_samples * nsamp)
        index = rng.choice(nsamp, nsamples, replace=False)
        x_samp = x[index]

    for new_values in grid_cartesian:
        x_eval = x_samp.copy()
        for i, feature in enumerate(features):
            x_eval[:, feature] = new_values[i]
        try:
            pred = model.predict_values(x_eval)
        except AttributeError:
            pred = model.predict(x_eval)
        averaged_pred = np.average(pred, weights=sample_weight)

        predictions.append(pred)
        averaged_predictions.append(averaged_pred)

    predictions = np.array(predictions).T
    averaged_predictions = np.array(averaged_predictions).T

    if method != "sample":
        predictions = predictions.reshape([nsamples] + lengths)
        averaged_predictions = averaged_predictions.reshape(lengths)

    return averaged_predictions, predictions


def partial_dependence(
    model,
    x,
    features: Union[List, Tuple],
    *,
    sample_weight=None,
    categorical_feature_indices: List = None,
    percentiles=(0.05, 0.95),
    grid_resolution=100,
    kind="average",
    method="uniform",
    ratio_samples=None,
    categories_map=None,
    seed=None,
):
    """
    Partial dependence.

    Parameters
    ----------
    - model
    -

    Returns
    ----------


    """
    # to do: check if the model is fitted
    pass
    # to do: check if the model in SMT supports multi class multi label classification
    pass
    # to do: check sample weight

    if sample_weight is not None:
        pass

    for i, feature in enumerate(features):
        if type(feature) in [tuple, list]:
            if len(feature) == 1:
                features[i] = feature[0]
            elif len(feature) == 2:
                features[i] = tuple(feature)
            else:
                if method != "sample":
                    raise ValueError("Interaction features can't be more than two.")

    # list to store the features are categorical or not in x
    is_categorical = [False] * x.shape[1]
    if categorical_feature_indices is not None:
        for feature_idx in categorical_feature_indices:
            is_categorical[feature_idx] = True

    pdp_results = []
    for feature in features:
        # create grid
        grid_cartesian, grid_values = grid_from_x(
            x,
            feature,
            percentiles,
            grid_resolution,
            is_categorical,
            method,
        )

        # predictions
        averaged_predictions, predictions = _partial_dependence_brute(
            model,
            grid_cartesian,
            grid_values,
            feature,
            x,
            method,
            sample_weight,
            ratio_samples,
            seed=seed,
        )

        # storing values
        pdp_result = {"grid_values": grid_values}
        if kind == "average":
            pdp_result["average"] = averaged_predictions
        elif kind == "individual":
            pdp_result["individual"] = predictions
        else:
            pdp_result["average"] = averaged_predictions
            pdp_result["individual"] = predictions

        # if there's categorical features, store grid categories
        if isinstance(feature, int):
            is_categories = [is_categorical[feature]]
        else:
            is_categories = [is_categorical[f] for f in feature]
        has_categories = max(is_categories) is True

        if has_categories == 1:
            grid_categories = []
            for i, (is_category, grid_values_) in enumerate(zip(is_categories, grid_values)):
                if is_category:
                    if categories_map is not None:
                        if isinstance(feature, int):
                            f = feature
                        else:
                            f = feature[i]
                        grid_categories_ = [categories_map[f][i] for i in grid_values_]
                    else:
                        grid_categories_ = [value for value in grid_values_]
                else:
                    grid_categories_ = []

                grid_categories_ = np.array(grid_categories_)
                grid_categories.append(grid_categories_)
            pdp_result["grid_categories"] = grid_categories

        pdp_results.append(pdp_result)

    return pdp_results


def cartesian(arrays, out=None):
    """Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray of shape (M, len(arrays)), default=None
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray of shape (M, len(arrays))
        Array containing the cartesian products formed of input arrays.
        If not provided, the `dtype` of the output array is set to the most
        permissive `dtype` of the input arrays, according to NumPy type
        promotion.

    Notes
    -----
    This function may not be used on more than 32 arrays
    because the underlying numpy functions do not support it.

    Examples
    --------
    >>> from sklearn.utils.extmath import cartesian
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])
    """
    arrays = [np.asarray(x) for x in arrays]
    shape = (len(x) for x in arrays)

    ix = np.indices(shape)
    ix = ix.reshape(len(arrays), -1).T

    if out is None:
        dtype = np.result_type(*arrays)  # find the most permissive dtype
        if dtype.str[:2] != "<U":
            out = np.empty_like(ix, dtype=dtype)
        else:
            out = np.empty_like(ix, dtype="object")

    for n, arr in enumerate(arrays):
        out[:, n] = arrays[n][ix[:, n]]

    return out


def non_cartesian(arrays, out=None):
    arrays = [np.asarray(x) for x in arrays]

    ref = np.zeros((len(arrays[0]), len(arrays)))
    if out is None:
        out = np.empty_like(ref, dtype="object")

    for n, array in enumerate(arrays):
        out[:, n] = array
    return out
