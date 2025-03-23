from itertools import product
from scipy.special import comb
import math
import numpy as np


def compute_shap_values(model, x_obs, x_ref, is_categorical, *, method="kernel"):
    shap_values = list()
    # number of observations
    n_obs = len(x_obs)
    for i in range(n_obs):
        if method == "kernel":
            x = x_obs[i : i + 1, :].reshape(1, -1)
            shap_value = kernel_shap_values(model, x, x_ref, is_categorical)
        elif method == "exact":
            x = x_obs[i : i + 1, :].reshape(
                -1,
            )
            shap_value = exact_shap_values(model, x, x_ref, is_categorical)
        else:
            raise ValueError("Invalid method. It must be either 'kernel' or 'exact'.")
        shap_values.append(shap_value)
    shap_values = np.array(shap_values)
    return shap_values


def kernel_shap_values(model, x, x_ref, is_categorical):
    num_features = x.shape[1]
    mask = create_mask_array(num_features)
    reference_values = np.ones(mask.shape)
    for i in range(len(reference_values)):
        reference_values[i, :] = get_reference_feature_values(x_ref, is_categorical)
    s_ref = (mask == 0) * reference_values
    s_real = (mask == 1) * x
    s_full = s_ref + s_real

    weights = np.apply_along_axis(calculate_weight_for_kernel_shap, 1, mask)
    shap_value = weighted_least_squares_for_kernel_shap(
        mask, s_full, weights, reference_values, model
    )
    return shap_value


def exact_shap_values(model, x, x_ref, is_categorical):
    num_features = x_ref.shape[1]
    shap_value = np.zeros(num_features)

    combinations_try = generate_binary_combinations(num_features)
    for feature_idx in range(num_features):
        # filter the combinations where the i-th feature is not included
        indices = np.where(combinations_try[:, feature_idx] == 0)[0]
        combinations = combinations_try[indices, :]

        x_s = np.zeros(combinations.shape)
        x_s_with_true_f = np.zeros(combinations.shape)
        weight = np.zeros(combinations.shape[0])

        for s in range(combinations.shape[0]):
            subset = combinations[s, :]
            # subset with true f
            subset_with_true_f = subset.copy()
            subset_with_true_f[feature_idx] = 1

            # compute the marginal contribution
            x_ref_current = get_reference_feature_values(x_ref, is_categorical)
            x_subset = x_ref_current.copy()
            x_subset[subset_with_true_f == 1] = x[subset_with_true_f == 1]
            x_s_with_true_f[s, :] = x_subset

            x_subset = x_ref_current.copy()
            x_subset[subset == 1] = x[subset == 1]
            x_s[s, :] = x_subset

            # compute shapley weight
            weight[s] = math.factorial(subset.sum()) * math.factorial(
                num_features - subset.sum() - 1
            )
            weight[s] = weight[s] / math.factorial(num_features)

        f_s_with_true_f = model.predict_values(x_s_with_true_f)
        f_s = model.predict_values(x_s)
        marginal_contributions = f_s_with_true_f - f_s
        shap_value[feature_idx] = np.dot(weight, marginal_contributions)

    return shap_value


def create_mask_array(m):
    mask = np.array(list(product(range(2), repeat=m)))
    # remove mask where all elements are 0 / 1
    mask = mask[(~np.all(mask == 0, axis=1)) & (~np.all(mask == 1, axis=1))]
    return mask


def get_reference_feature_values(x, is_categorical):
    # get reference values for each feature
    # if the feature is categorical/ordinal -> random
    # else -> mean
    num_features = x.shape[1]
    reference_values = np.zeros(num_features)
    for feature_idx in range(num_features):
        if is_categorical[feature_idx]:
            # mode = stats.mode(x[:, feature_idx], keepdims=False)[0]
            # reference_values[feature_idx] = mode
            # reference_values[feature_idx] = np.random.choice(x[:, feature_idx])
            reference_values[feature_idx] = np.min(x[:, feature_idx])
        else:
            mean = np.mean(x[:, feature_idx])
            reference_values[feature_idx] = mean
    return reference_values


def calculate_weight_for_kernel_shap(mask_row):
    m = len(mask_row)
    z = np.sum(mask_row)
    numerator = m - 1
    denominator = comb(m, z) * z * (m - z)
    weight = numerator / denominator
    return weight


def weighted_least_squares_for_kernel_shap(
    mask, s_full, weights, reference_values, model
):
    y = model.predict_values(s_full)
    b0 = model.predict_values(reference_values)
    y = y - b0

    w = np.diag(weights)

    b = np.dot(
        np.linalg.inv(np.dot(np.dot(mask.transpose(), w), mask)),
        np.dot(np.dot(mask.transpose(), w), y),
    )
    b = b.reshape(
        -1,
    )
    return b


def generate_binary_combinations(n):
    num_combinations = 2**n
    combinations = [
        list(format(i, f"0{(num_combinations - 1).bit_length()}b"))
        for i in range(num_combinations)
    ]
    combinations = np.array(combinations, dtype=int)
    return combinations
