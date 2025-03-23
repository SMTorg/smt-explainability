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
from smt.explainability_tools import ShapFeatureImportanceDisplay, ShapDisplay

import time
import numpy as np


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
n_data = 100
n_train = int(0.8 * n_data)
n_test = n_data - n_train
xdoe, _ = ds.sample_valid_x(n_data)
y_doe = [f_obj(xdoe[i]) for i in range(len(xdoe))]

X_tr, y_tr = xdoe[:n_train, :], y_doe[:n_train]
X_te, y_te = xdoe[n_train:, :], y_doe[n_train:]
y_tr, y_te = np.array(y_tr), np.array(y_te)

# create mapping for the categories
categories_map = dict()
for feature_idx in categorical_feature_idx:
    categories_map[feature_idx] = {
        i: value for i, value in enumerate(ds._design_variables[feature_idx].values)
    }

"""Modelling"""
sm = MixedIntegerKrigingModel(
    surrogate=KPLS(
        design_space=ds,
        categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
        hierarchical_kernel=MixHrcKernelType.ARC_KERNEL,
        theta0=np.array([4.43799547e-04, 4.39993134e-01, 1.59631650e00]),
        corr="squar_exp",
        n_start=1,
        cat_kernel_comps=[2],
        n_comp=2,
        print_global=False,
    ),
)
start_time = time.time()
sm.set_training_values(X_tr, np.array(y_tr))
sm.train()
print("run time (s):", time.time() - start_time)
y_pred = sm.predict_values(X_te)
rmse = mean_squared_error(y_te, y_pred, squared=False)
rrmse = rmse / y_te.mean()
print(f"RMSE: {rmse:.4f}")
print(f"rRMSE: {rrmse:.4f}")

instances = X_tr
model = sm
feature_names = [
    r"$I$",
    r"$L$",
    r"$S$",
]
cat_feature_indexes = [0]
# categories_map=None
shap_explainer = ShapDisplay.from_surrogate_model(
    instances,
    model,
    X_tr,
    feature_names=feature_names,
    cat_feature_indexes=cat_feature_indexes,
    categories_map=categories_map,
)

features = [0, 1, 2]

shap_explainer.dependence_plot(
    features, figsize=[25, 5], sort_based_on_importance=False, max_num_entities=5
)

shap_explainer.summary_plot(figsize=(5, 3), n_color=10)
