from smt.utils.sm_test_case import SMTestCase
from smt.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
)
from smt.sampling_methods import LHS
from smt.problems import WingWeight
from smt_explainability.problems import MixedCantileverBeam
from smt_explainability.shap import ShapDisplay

import unittest


class GroundTruthModel:
    def __init__(self, fun):
        self.fun = fun

    def predict_values(self, x):
        return self.fun(x)


class TestPartialDependenceNumerical(SMTestCase):
    def setUp(self):
        nsamples = 300
        n_train = int(0.8 * nsamples)
        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion="ese", seed=1)
        x = sampling(nsamples)
        y = fun(x)
        x_tr, _y_tr = x[:n_train, :], y[:n_train]
        x_te, _y_te = x[n_train:, :], y[n_train:]
        is_categorical = [False] * x.shape[1]

        feature_names = [
            r"$S_{w}$",
            r"$W_{fw}$",
            r"$A$",
            r"$\Delta$",
            r"$q$",
            r"$\lambda$",
            r"$t_{c}$",
            r"$N_{z}$",
            r"$W_{dg}$",
            r"$W_{p}$",
        ]

        # sm = KRG(
        #     theta0=[1e-2] * x.shape[1],
        #     print_prediction=False
        # )
        # sm.set_training_values(x, y)
        # sm.train()

        self.model = GroundTruthModel(fun)
        self.x_tr = x_tr
        self.x_te = x_te
        self.nsamples = nsamples
        self.is_categorical = is_categorical
        self.feature_names = feature_names
        self.index_for_individual_plot = 0
        self.feature_pairs_for_numerical_problem = [(0, 1), (2, 3)]

    def test_kernel_shap(self):
        shap_explainer = ShapDisplay.from_surrogate_model(
            self.x_te,
            self.model,
            self.x_tr,
            feature_names=self.feature_names,
            method="kernel",
        )
        shap_explainer.individual_plot(index=self.index_for_individual_plot)
        shap_explainer.dependence_plot([i for i in range(self.x_te.shape[1])])
        shap_explainer.interaction_plot(self.feature_pairs_for_numerical_problem)
        shap_explainer.summary_plot()
        assert shap_explainer.shap_values.shape == (
            self.x_te.shape[0],
            self.x_te.shape[1],
        )

    def test_exact_shap(self):
        shap_explainer = ShapDisplay.from_surrogate_model(
            self.x_te,
            self.model,
            self.x_tr,
            feature_names=self.feature_names,
            method="exact",
        )
        shap_explainer.individual_plot(index=self.index_for_individual_plot)
        shap_explainer.dependence_plot([i for i in range(self.x_te.shape[1])])
        shap_explainer.interaction_plot(self.feature_pairs_for_numerical_problem)
        shap_explainer.summary_plot()
        assert shap_explainer.shap_values.shape == (
            self.x_te.shape[0],
            self.x_te.shape[1],
        )


class TestPartialDependenceMixed(SMTestCase):
    def setUp(self):
        nsamples = 100
        n_train = int(0.8 * nsamples)

        fun = MixedCantileverBeam()
        ds = DesignSpace(
            [
                CategoricalVariable(values=[str(i + 1) for i in range(12)]),
                FloatVariable(10.0, 20.0),
                FloatVariable(1.0, 2.0),
            ]
        )
        x = fun.sample(nsamples)
        y = fun(x)
        x_tr, _y_tr = x[:n_train, :], y[:n_train]
        x_te, _y_te = x[n_train:, :], y[n_train:]

        # Index for categorical features
        categorical_feature_indices = [0]
        # create mapping for the categories
        categories_map = dict()
        is_categorical = [False] * x.shape[1]
        for feature_idx in categorical_feature_indices:
            categories_map[feature_idx] = {i: value for i, value in enumerate(ds._design_variables[feature_idx].values)}
            is_categorical[feature_idx] = True

        feature_names = [r"$\tilde{I}$", r"$L$", r"$S$"]

        # sm = MixedIntegerKrigingModel(
        #     surrogate=KPLS(
        #         design_space=ds,
        #         categorical_kernel=MixIntKernelType.HOMO_HSPHERE,
        #         hierarchical_kernel=MixHrcKernelType.ARC_KERNEL,
        #         theta0=np.array([4.43799547e-04, 4.39993134e-01, 1.59631650e+00]),
        #         corr="squar_exp",
        #         n_start=1,
        #         cat_kernel_comps=[2],
        #         n_comp=2,
        #         print_global=False,
        #     ),
        # )
        # sm.set_training_values(x, np.array(y))
        # sm.train()

        self.model = GroundTruthModel(fun)
        self.x_tr = x_tr
        self.x_te = x_te
        self.categories_map = categories_map
        self.categorical_feature_indices = categorical_feature_indices
        self.nsamples = nsamples
        self.is_categorical = is_categorical
        self.feature_names = feature_names
        self.index_for_individual_plot = 0
        self.feature_pairs_for_mixed_problem = [(0, 1), (2, 0), (1, 2)]

    def test_kernel_shap(self):
        shap_explainer = ShapDisplay.from_surrogate_model(
            self.x_te,
            self.model,
            self.x_tr,
            feature_names=self.feature_names,
            categorical_feature_indices=self.categorical_feature_indices,
            categories_map=self.categories_map,
            method="kernel",
        )

        shap_explainer.individual_plot(index=self.index_for_individual_plot)
        shap_explainer.dependence_plot([i for i in range(self.x_te.shape[1])])
        shap_explainer.interaction_plot(self.feature_pairs_for_mixed_problem)
        shap_explainer.summary_plot()

        assert shap_explainer.shap_values.shape == (
            self.x_te.shape[0],
            self.x_te.shape[1],
        )

    def test_exact_shap(self):
        shap_explainer = ShapDisplay.from_surrogate_model(
            self.x_te,
            self.model,
            self.x_tr,
            feature_names=self.feature_names,
            categorical_feature_indices=self.categorical_feature_indices,
            categories_map=self.categories_map,
            method="kernel",
        )

        shap_explainer.individual_plot(index=self.index_for_individual_plot)
        shap_explainer.dependence_plot([i for i in range(self.x_te.shape[1])])
        shap_explainer.interaction_plot(self.feature_pairs_for_mixed_problem)
        shap_explainer.summary_plot()

        assert shap_explainer.shap_values.shape == (
            self.x_te.shape[0],
            self.x_te.shape[1],
        )


if __name__ == "__main__":
    unittest.main()
