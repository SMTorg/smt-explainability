# %%
import sys

dirname = "/Users/muhammaddaffarobani/Documents/personal_research/smt"
if dirname not in sys.path:
    sys.path.append(dirname)

from smt.utils.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
)
from smt.applications.mixed_integer import (
    MixedIntegerKrigingModel,
)
from smt.surrogate_models import (
    KPLS,
    MixIntKernelType,
    MixHrcKernelType,
)
from sklearn.metrics import mean_squared_error
from smt.explainability_tools import (
    PartialDependenceDisplay,
    PDFeatureImportanceDisplay,
    PDFeatureInteractionDisplay,
    ShapFeatureImportanceDisplay,
    ShapDisplay,
)

import time
import numpy as np


# %%
# Objective function
def cantilever_deflection(x):
    norm_inertia_vals = [
        0.0833,
        0.139,
        0.380,
        0.0796,
        0.133,
        0.363,
        0.0859,
        0.136,
        0.360,
        0.0922,
        0.138,
        0.369,
    ]

    I = int(x[0])
    L = x[1]
    S = x[2]
    norm_inertia = norm_inertia_vals[I]

    E = 200e9
    P = 50e3

    y = (P * L**3) / (3 * E * S**2 * norm_inertia)

    return y


def cantilever_deflection_vectorized(X):
    y = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        y[i] = cantilever_deflection(X[i, :])
    return y


class GroundTruthModel:
    def predict_values(self, X):
        return cantilever_deflection_vectorized(X)


# %%
"""Problem definition"""
# Design space
ds = DesignSpace(
    [
        CategoricalVariable(
            [
                "type_A",
                "type_B",
                "type_C",
                "type_D",
                "type_E",
                "type_F",
                "type_G",
                "type_H",
                "type_I",
                "type_J",
                "type_K",
                "type_L",
            ]
        ),
        FloatVariable(10.0, 20.0),
        FloatVariable(1.0, 2.0),
    ]
)
categorical_feature_idx = [0]

f_obj = cantilever_deflection

# Create training and testing data
n_data = 200
n_train = int(0.8 * n_data)
n_test = n_data - n_train
xdoe, _ = ds.sample_valid_x(n_data)
y_doe = [f_obj(xdoe[i]) for i in range(len(xdoe))]

X_tr, y_tr = xdoe[:n_train, :], y_doe[:n_train]
X_te, y_te = xdoe[n_train:, :], y_doe[n_train:]
y_tr, y_te = np.array(y_tr), np.array(y_te)

# Name of the features
feature_names = [r"$I$", r"$L$", r"$S$"]

# create mapping for the categories
categories_map = dict()
inverse_categories_map = dict()
for feature_idx in categorical_feature_idx:
    categories_map[feature_idx] = {
        i: value for i, value in enumerate(ds._design_variables[feature_idx].values)
    }
    inverse_categories_map[feature_idx] = {
        value: i for value, i in enumerate(ds._design_variables[feature_idx].values)
    }


# %%
class GroundTruthModel:
    def predict_values(self, X):
        return cantilever_deflection_vectorized(X)


sm = GroundTruthModel()
start_time = time.time()
y_pred = sm.predict_values(X_te)
rmse = mean_squared_error(y_te, y_pred, squared=False)
rrmse = rmse / y_te.mean()
print(f"RMSE: {rmse:.4f}")
print(f"rRMSE: {rrmse:.4f}")


# %%
model = sm
X = X_tr
features = [
    (1, 0),
    (
        1,
        2,
    ),
]
categorical_features = categorical_feature_idx

pdd = PartialDependenceDisplay.from_surrogate_model(
    model,
    X,
    features,
    categorical_features=categorical_features,
    feature_names=feature_names,
    grid_resolution=20,
    kind="both",
    centered=True,
    figsize=[25, 10],
    inverse_categories_map=inverse_categories_map,
)
