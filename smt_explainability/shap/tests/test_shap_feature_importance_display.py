from smt.utils.sm_test_case import SMTestCase
from smt.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
)
from smt.sampling_methods import LHS
from smt.problems import WingWeight
from smt_explainability.problems import MixedCantileverBeam
from smt_explainability.shap import ShapFeatureImportanceDisplay

import unittest


class GroundTruthModel:
    def __init__(self, fun):
        self.fun = fun

    def predict_values(self, x):
        return self.fun(x)


class TestPartialDependenceNumerical(SMTestCase):
    def setUp(self):
        nsamples = 300
        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion="ese", seed=1)
        x = sampling(nsamples)
        fun(x)

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
        self.x = x
        self.nsamples = nsamples
        self.feature_names = feature_names

    def test_kernel_shap_feature_importance(self):
        shap_importance = ShapFeatureImportanceDisplay.from_surrogate_model(
            self.model,
            self.x,
            feature_names=self.feature_names,
            method="kernel",
        )
        shap_importance.plot()
        assert len(shap_importance.feature_importances) == self.x.shape[1]

    def test_exact_shap_feature_importance(self):
        shap_importance = ShapFeatureImportanceDisplay.from_surrogate_model(
            self.model,
            self.x,
            feature_names=self.feature_names,
            method="exact",
        )
        shap_importance.plot()
        assert len(shap_importance.feature_importances) == self.x.shape[1]


class TestPartialDependenceMixed(SMTestCase):
    def setUp(self):
        nsamples = 100

        fun = MixedCantileverBeam()
        DesignSpace(
            [
                CategoricalVariable(values=[str(i + 1) for i in range(12)]),
                FloatVariable(10.0, 20.0),
                FloatVariable(1.0, 2.0),
            ]
        )
        x = fun.sample(nsamples)
        fun(x)

        # Index for categorical features
        categorical_feature_indices = [0]

        is_categorical = [False] * x.shape[1]
        for feature_idx in categorical_feature_indices:
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
        self.x = x
        self.categorical_feature_indices = categorical_feature_indices
        self.nsamples = nsamples
        self.feature_names = feature_names

    def test_kernel_shap_feature_importance(self):
        shap_importance = ShapFeatureImportanceDisplay.from_surrogate_model(
            self.model,
            self.x,
            feature_names=self.feature_names,
            categorical_feature_indices=self.categorical_feature_indices,
            method="kernel",
        )
        shap_importance.plot()
        assert len(shap_importance.feature_importances) == self.x.shape[1]

    def test_exact_shap_feature_importance(self):
        shap_importance = ShapFeatureImportanceDisplay.from_surrogate_model(
            self.model,
            self.x,
            feature_names=self.feature_names,
            categorical_feature_indices=self.categorical_feature_indices,
            method="exact",
        )
        shap_importance.plot()
        assert len(shap_importance.feature_importances) == self.x.shape[1]


if __name__ == "__main__":
    unittest.main()
