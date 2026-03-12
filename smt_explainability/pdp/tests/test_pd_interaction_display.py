from smt.utils.sm_test_case import SMTestCase
from smt.problems import WingWeight
from smt.sampling_methods import LHS

from smt_explainability.pdp import PDFeatureInteractionDisplay

from smt.problems import MixedCantileverBeam
from smt.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
)
import unittest
import itertools
import random


class GroundTruthModel:
    def __init__(self, fun):
        self.fun = fun

    def predict_values(self, x):
        return self.fun(x)


class TestPDInteractionDisplayNumerical(SMTestCase):
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
        self.num_feature_pairs = 3
        self.feature_names = feature_names

    def test_pd_overall_interaction(self):
        overall_pd_interaction = PDFeatureInteractionDisplay.overall_interaction(
            self.model,
            self.x,
            feature_names=self.feature_names,
        )
        overall_pd_interaction.plot()
        assert len(overall_pd_interaction.h_scores) == self.x.shape[1]

    def test_pd_pairwise_interaction(self):
        feature_pairs = list(itertools.combinations([i for i in range(self.x.shape[1])], 2))
        random.shuffle(feature_pairs)
        feature_pairs = feature_pairs[: self.num_feature_pairs]

        pairwise_pd_interaction = PDFeatureInteractionDisplay.pairwise_interaction(
            self.model,
            self.x,
            feature_pairs,
            feature_names=self.feature_names,
        )
        pairwise_pd_interaction.plot()
        assert len(pairwise_pd_interaction.h_scores) == len(feature_pairs)


class TestPDInteractionDisplayMixed(SMTestCase):
    def setUp(self):
        nsamples = 100
        fun = MixedCantileverBeam()
        ds = DesignSpace(
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
        # create mapping for the categories
        categories_map = dict()
        for feature_idx in categorical_feature_indices:
            categories_map[feature_idx] = {i: value for i, value in enumerate(ds._design_variables[feature_idx].values)}

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

        # model = sm

        # self.model = sm
        self.model = GroundTruthModel(fun)
        self.x = x
        self.categorical_feature_indices = categorical_feature_indices
        self.num_feature_pairs = 3
        self.feature_names = feature_names

    def test_pd_overall_interaction(self):
        overall_pd_interaction = PDFeatureInteractionDisplay.overall_interaction(
            self.model,
            self.x,
            feature_names=self.feature_names,
            categorical_feature_indices=self.categorical_feature_indices,
        )
        overall_pd_interaction.plot()
        assert len(overall_pd_interaction.h_scores) == self.x.shape[1]

    def test_pd_pairwise_interaction(self):
        feature_pairs = list(itertools.combinations([i for i in range(self.x.shape[1])], 2))
        random.shuffle(feature_pairs)
        feature_pairs = feature_pairs[: self.num_feature_pairs]

        pairwise_pd_interaction = PDFeatureInteractionDisplay.pairwise_interaction(
            self.model,
            self.x,
            feature_pairs,
            feature_names=self.feature_names,
            categorical_feature_indices=self.categorical_feature_indices,
        )
        pairwise_pd_interaction.plot()
        assert len(pairwise_pd_interaction.h_scores) == len(feature_pairs)


if __name__ == "__main__":
    unittest.main()
