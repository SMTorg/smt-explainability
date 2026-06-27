import numpy as np
from smt.design_space import DesignSpace, FloatVariable, CategoricalVariable
from smt.surrogate_models import KRG
from smt.sampling_methods import LHS
from smt_explainability.hsic_anova.hsic_adsg import HsicAnovaAdsg


def test_hsic_adsg_coverage():
    """
    End-to-End integration test to guarantee high code coverage for hsic_adsg.py.
    This creates a mock hierarchical design space, trains a surrogate,
    and extracts the HSIC-ANOVA indices under various parameters.
    """
    np.random.seed(42)

    # 1. Create a Mixed Hierarchical Design Space
    ds = DesignSpace(
        [
            FloatVariable(0, 1),  # x0: Root
            FloatVariable(0, 1),  # x1: Decreed continuous
            CategoricalVariable(["A", "B"]),  # x2: Decreed categorical
        ]
    )

    # x1 acts if x0 > 0.5
    ds.declare_decreed_var(decreed_var=1, meta_var=0, meta_value=[0.5, 1.0])
    # x2 acts if x0 < 0.5 (just to test categorical distance)
    ds.declare_decreed_var(decreed_var=2, meta_var=0, meta_value=[0.0, 0.5])

    # 2. Sample Data
    sampler = LHS(xlimits=ds.get_num_bounds(), criterion="ese", seed=42)
    x_samp_raw = sampler(30)

    # SMT sampling generates continuous values for categorical bounds,
    # but we just need dummy data to run the distance algorithms.
    x_samp = x_samp_raw.copy()

    _, is_acting = ds.correct_get_acting(x_samp)

    # Impute inactive continuous
    x_samp[~is_acting[:, 1], 1] = 0.5
    # Impute inactive categorical
    x_samp[~is_acting[:, 2], 2] = 0.0

    # Mock output
    y_samp = x_samp[:, 0] + np.where(is_acting[:, 1], x_samp[:, 1], 0.0)

    # 3. Train Surrogate
    sm = KRG(design_space=ds, print_global=False)
    sm.set_training_values(x_samp, y_samp)
    sm.train()

    # 4. Run Explainer (Median Heuristic)
    explainer = HsicAnovaAdsg(model=sm, ds=ds)
    results_med, total_hsic_med = explainer.explain(
        X=x_samp, max_order=2, var_names=["x0", "x1", "x2"], use_kta=False, use_rf_prior=False
    )

    assert total_hsic_med > 0, "Total HSIC should be positive."
    assert len(results_med) > 0, "Should extract at least some combinations."

    # 5. Run Explainer (KTA Optimization) to cover _compute_theta_kta
    results_kta, total_hsic_kta = explainer.explain(X=x_samp, max_order=2, use_kta=True, use_rf_prior=False)
    assert total_hsic_kta > 0

    # 6. Run Explainer with provided theta_scales to cover the pre-computed branch
    dummy_theta = np.array([1.0, 0.0, 5.0])
    results_provided, _ = explainer.explain(
        X=x_samp, max_order=1, theta_scales=dummy_theta, use_smt_theta=True, use_kta=False, use_rf_prior=False
    )

    # 7. Run Explainer with KTA and theta_scales (to cover theta_mask initialization in L-BFGS-B)
    results_kta_mask, _ = explainer.explain(
        X=x_samp, max_order=1, theta_scales=dummy_theta, use_kta=True, use_rf_prior=False
    )
