import numpy as np
from smt.design_space import DesignSpace, FloatVariable, CategoricalVariable
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS
from smt_explainability.hsic_anova.hsic_adsg import (
    HsicAnovaAdsg,
    _apply_algebraic_distance,
    _center_kernel,
)


def test_algebraic_distance():
    # Test 1: Verify distance calculations natively
    X = np.array([[0.5, 0.5]])
    Y = np.array([[0.5, 0.8]])
    x_act = np.array([[True, False]])
    y_act = np.array([[True, False]])
    num_is_decreed = np.array([False, True])
    is_cat = np.array([False, False])

    dist = _apply_algebraic_distance(X, Y, x_act, y_act, num_is_decreed, is_cat)
    # distance on variable 0 should be 0 (0.5 - 0.5)
    assert dist[0, 0, 0] == 0.0
    # distance on variable 1 should be 0.0 because both are inactive
    assert dist[0, 0, 1] == 0.0


def test_hadamard_trace():
    # Test 2: Verify O(N^2) trace matching O(N^3)
    np.random.seed(42)
    N = 15
    A_raw = np.random.randn(N, N)
    K = A_raw @ A_raw.T

    # O(N^2) fast centering
    Kc = _center_kernel(K)

    # Naive O(N^3) centering
    J = np.ones((N, N))
    H = np.eye(N) - (1.0 / N) * J
    Kc_naive = H @ K @ H

    assert np.allclose(Kc, Kc_naive)


def _get_mock_data():
    np.random.seed(42)
    ds = DesignSpace([FloatVariable(0, 1), FloatVariable(0, 1), CategoricalVariable(["A", "B"])])
    ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value=[0.5, 1.0])
    ds.declare_decreed_var(decreed_var=2, meta_var=0, meta_value=[0.0, 0.5])
    sampler = LHS(xlimits=ds.get_num_bounds(), criterion="ese", seed=42)
    x_samp = sampler(30).copy()
    _, is_acting = ds.correct_get_acting(x_samp)
    x_samp[~is_acting[:, 1], 1] = 0.5
    x_samp[~is_acting[:, 2], 2] = 0.0
    y_samp = x_samp[:, 0] + np.where(is_acting[:, 1], x_samp[:, 1], 0.0)

    sm = KRG(design_space=ds, print_global=False)
    sm.set_training_values(x_samp, y_samp)
    sm.train()
    return sm, ds, x_samp


def test_hsic_adsg_median():
    # Test 3: End-to-end with median heuristic
    sm, ds, x_samp = _get_mock_data()
    explainer = HsicAnovaAdsg(model=sm, ds=ds)
    results, total_hsic = explainer.explain(
        X=x_samp, max_order=2, var_names=["x0", "x1", "x2"], use_kta=False, use_rf_prior=False
    )
    assert total_hsic > 0
    assert len(results) > 0


def test_hsic_adsg_kta():
    # Test 4: End-to-end with KTA optimization
    sm, ds, x_samp = _get_mock_data()
    explainer = HsicAnovaAdsg(model=sm, ds=ds)
    results, total_hsic = explainer.explain(X=x_samp, max_order=2, use_kta=True, use_rf_prior=False)
    assert total_hsic > 0


def test_hsic_adsg_provided_theta():
    # Test 5: End-to-end with provided theta scales and KTA
    sm, ds, x_samp = _get_mock_data()
    explainer = HsicAnovaAdsg(model=sm, ds=ds)
    dummy_theta = np.array([1.0, 0.0, 5.0])
    results_prov, _ = explainer.explain(
        X=x_samp, max_order=1, theta_scales=dummy_theta, use_smt_theta=True, use_kta=False, use_rf_prior=False
    )
    assert len(results_prov) > 0

    results_kta_mask, _ = explainer.explain(
        X=x_samp, max_order=1, theta_scales=dummy_theta, use_kta=True, use_rf_prior=False
    )
    assert len(results_kta_mask) > 0
