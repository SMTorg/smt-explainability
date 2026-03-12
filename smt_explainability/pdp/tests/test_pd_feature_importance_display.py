from smt.utils.sm_test_case import SMTestCase
from smt.problems import WingWeight
from smt.sampling_methods import LHS

from smt_explainability.pdp import PDFeatureImportanceDisplay

from smt.problems import MixedCantileverBeam
from smt.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
)
import unittest


class GroundTruthModel:
    def __init__(self, fun):
        self.fun = fun

    def predict_values(self, x):
        return self.fun(x)


class TestPDFeatureImportance(SMTestCase):
    def test_pd_feature_importance_numerical(self):
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

        # model = sm
        model = GroundTruthModel(fun)

        pdd_importance = PDFeatureImportanceDisplay.from_surrogate_model(
            model, x, feature_names=feature_names
        )
        pdd_importance.plot()

        assert len(pdd_importance.feature_importances) == x.shape[1]

    def test_pd_feature_importance_mixed(self):
        nsamples = 100
        fun = MixedCantileverBeam()
        # Design space
        ds = DesignSpace(
            [
                CategoricalVariable(values=[str(i + 1) for i in range(12)]),
                FloatVariable(10.0, 20.0),
                FloatVariable(1.0, 2.0),
            ]
        )
        x = fun.sample(nsamples)
        fun(x)

        # Name of the features
        feature_names = [r"$\tilde{I}$", r"$L$", r"$S$"]
        # Index for categorical features
        categorical_feature_indices = [0]
        # create mapping for the categories
        categories_map = dict()
        for feature_idx in categorical_feature_indices:
            categories_map[feature_idx] = {
                i: value
                for i, value in enumerate(ds._design_variables[feature_idx].values)
            }

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

        # model = sm
        model = GroundTruthModel(fun)

        pdd_importance = PDFeatureImportanceDisplay.from_surrogate_model(
            model,
            x,
            feature_names=feature_names,
            categorical_feature_indices=categorical_feature_indices,
        )
        pdd_importance.plot()

        assert len(pdd_importance.feature_importances) == x.shape[1]


if __name__ == "__main__":
    unittest.main()
