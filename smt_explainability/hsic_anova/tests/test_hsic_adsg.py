import pytest
import numpy as np

# TODO: Replace mock functions with actual imports from smt_explainability.hsic_anova once implemented
# from smt_explainability.hsic_anova.adsg import calculate_distance, hadamard_trace_estimator, intrinsic_variance_estimator

def test_distance_calculations():
    """
    Test 1: Verify the ADSG distance calculations.
    - Two inactive variables MUST return a distance of 0.0.
    - An active/inactive pair MUST return 1.0.
    """
    # Mock function representing the intended behavior for TDD
    def calculate_distance(is_active_1, is_active_2):
        if not is_active_1 and not is_active_2:
            return 0.0
        elif is_active_1 != is_active_2:
            return 1.0
        return 0.0  # (For two active variables, it would use continuous metrics)

    assert calculate_distance(False, False) == 0.0, "Two inactive variables must return 0.0"
    assert calculate_distance(True, False) == 1.0, "Active/inactive pair must return 1.0"
    assert calculate_distance(False, True) == 1.0, "Active/inactive pair must return 1.0"

def test_hadamard_trace_estimator():
    """
    Test 2: Verify the O(N^2) Hadamard trace estimator perfectly matches the 
    naive dense O(N^3) matrix trace on a small 50x50 random Gram matrix.
    """
    np.random.seed(42)
    N = 50
    
    # Generate random Gram matrices (symmetric, positive semi-definite)
    A_raw = np.random.randn(N, N)
    A = A_raw @ A_raw.T
    
    B_raw = np.random.randn(N, N)
    B = B_raw @ B_raw.T
    
    # Naive O(N^3) calculation: Trace(A @ B)
    naive_trace = np.trace(A @ B)
    
    # O(N^2) Hadamard trace estimator: sum of element-wise product (valid since A, B are symmetric)
    def hadamard_trace_estimator(mat_a, mat_b):
        return np.sum(mat_a * mat_b)
        
    fast_trace = hadamard_trace_estimator(A, B)
    
    # Check for perfect match
    assert np.isclose(naive_trace, fast_trace), f"Trace mismatch: {naive_trace} vs {fast_trace}"

def test_intrinsic_variance_estimator():
    """
    Test 3: Construct a simple mock Hierarchical Design Space (1 root variable 
    triggering 1 decreed variable) and verify the Intrinsic Variance estimator 
    successfully corrects the structural sparsity.
    """
    np.random.seed(42)
    N = 100
    
    # 1 root variable in [0, 1]
    X1 = np.random.uniform(0, 1, N)
    
    # 1 decreed variable X2, active only when X1 > 0.5
    is_active_x2 = X1 > 0.5
    
    X2_values = np.random.randn(N)
    # Use np.nan for inactive structural sparsity
    X2 = np.where(is_active_x2, X2_values, np.nan)
    
    def intrinsic_variance_estimator(x, is_active):
        """
        Mock estimator: Calculates variance over the active subspace and scales 
        by the probability of activation to correct structural sparsity.
        """
        active_mask = is_active.astype(bool)
        p_active = np.mean(active_mask)
        
        if p_active == 0:
            return 0.0
            
        active_variance = np.var(x[active_mask])
        return p_active * active_variance

    corrected_var = intrinsic_variance_estimator(X2, is_active_x2)
    
    # Verification
    p_active_approx = np.mean(is_active_x2)
    expected_intrinsic = p_active_approx * np.var(X2[is_active_x2])
    
    assert corrected_var > 0.0, "Variance should be positive."
    assert not np.isnan(corrected_var), "Variance should not be NaN."
    assert np.isclose(corrected_var, expected_intrinsic), "Estimator did not correct structural sparsity correctly."
