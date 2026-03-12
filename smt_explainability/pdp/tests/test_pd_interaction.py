from smt.utils.sm_test_case import SMTestCase
from smt.design_space import (
    DesignSpace,
    FloatVariable,
    CategoricalVariable,
)
from smt.sampling_methods import LHS
from smt.problems import WingWeight
from smt_explainability.problems import MixedCantileverBeam
from smt_explainability.pdp import pd_overall_interaction, pd_pairwise_interaction

import itertools
import unittest
import numpy as np


class GroundTruthModel:
    def __init__(self, fun):
        self.fun = fun

    def predict_values(self, x):
        return self.fun(x)


class TestPDInteractionNumerical(SMTestCase):
    def setUp(self):
        nsamples = 300
        fun = WingWeight()
        sampling = LHS(xlimits=fun.xlimits, criterion="ese", seed=1)
        x = sampling(nsamples)
        fun(x)

        # sm = KRG(
        #     theta0=[1e-2] * x.shape[1],
        #     print_prediction=False
        # )
        # sm.set_training_values(x, y)
        # sm.train()

        self.model = GroundTruthModel(fun)
        self.x = x
        self.num_feature_pairs = 3

    def test_overall_interaction(self):
        features = [i for i in range(self.x.shape[1])]
        overall_interaction = pd_overall_interaction(
            features,
            self.x,
            self.model,
        )

        assert len(overall_interaction) == len(features)

    def test_pairwise_interaction(self):
        feature_pairs = list(itertools.combinations([i for i in range(self.x.shape[1])], 2))
        rng = np.random.default_rng(1)
        rng.shuffle(feature_pairs)
        feature_pairs = feature_pairs[: self.num_feature_pairs]

        pairwise_interaction = pd_pairwise_interaction(
            feature_pairs,
            self.x,
            self.model,
        )

        assert len(pairwise_interaction) == len(feature_pairs)


class TestPDInteractionMixed(SMTestCase):
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

    def test_overall_interaction(self):
        features = [i for i in range(self.x.shape[1])]
        overall_interaction = pd_overall_interaction(
            features,
            self.x,
            self.model,
            categorical_feature_indices=self.categorical_feature_indices,
        )

        assert len(overall_interaction) == len(features)

    def test_pairwise_interaction(self):
        feature_pairs = list(itertools.combinations([i for i in range(self.x.shape[1])], 2))
        rng = np.random.default_rng(1)
        rng.shuffle(feature_pairs)
        feature_pairs = feature_pairs[: self.num_feature_pairs]

        pairwise_interaction = pd_pairwise_interaction(
            feature_pairs,
            self.x,
            self.model,
            categorical_feature_indices=self.categorical_feature_indices,
        )

        assert len(pairwise_interaction) == len(feature_pairs)


if __name__ == "__main__":
    unittest.main()
