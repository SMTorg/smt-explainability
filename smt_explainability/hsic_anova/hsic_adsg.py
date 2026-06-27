"""
HSIC-ANOVA Explainer for hierarchical design spaces (ADSG).
"""

import numpy as np
from itertools import combinations
from numba import njit
from scipy.optimize import minimize
from typing import Optional, List, Tuple, Dict, Any

# We assume smt is installed and accessible since this is an smt extension
from smt.design_space import CategoricalVariable


@njit
def _apply_algebraic_distance(
    X_num: np.ndarray,
    Y_num: np.ndarray,
    x_num_is_acting: np.ndarray,
    y_num_is_acting: np.ndarray,
    num_is_decreed: np.ndarray,
    is_categorical: np.ndarray,
) -> np.ndarray:
    """
    Computes the meta-decreed (Hallé-Hannan) distance for structured spaces.
    """
    nx_samples, n_features = X_num.shape
    ny_samples = Y_num.shape[0]

    D_features = np.zeros((nx_samples, ny_samples, n_features))

    for k1 in range(nx_samples):
        for k2 in range(ny_samples):
            for i in range(n_features):
                x_val = X_num[k1, i]
                y_val = Y_num[k2, i]

                if num_is_decreed[i]:
                    x_act = x_num_is_acting[k1, i]
                    y_act = y_num_is_acting[k2, i]

                    if not x_act and not y_act:
                        dist = 0.0
                    elif x_act != y_act:
                        dist = 1.0
                    else:
                        if is_categorical[i]:
                            dist = 1.0 if x_val != y_val else 0.0
                        else:
                            dist = (2 * np.abs(x_val - y_val)) / (np.hypot(1, x_val) * np.hypot(1, y_val))
                else:
                    if is_categorical[i]:
                        dist = 1.0 if x_val != y_val else 0.0
                    else:
                        dist = np.abs(x_val - y_val)

                D_features[k1, k2, i] = dist

    return D_features


def _center_kernel(K: np.ndarray) -> np.ndarray:
    """Centers a kernel matrix empirically in O(N^2) time."""
    return K - np.mean(K, axis=0, keepdims=True) - np.mean(K, axis=1, keepdims=True) + np.mean(K)


def _compute_kernel_from_dist(D_matrix: np.ndarray, length_scale: float = 1.0) -> np.ndarray:
    """Computes Gaussian kernel from distance matrix."""
    return np.exp(-(D_matrix**2) / (2 * length_scale**2))


def _compute_theta_median_heuristic(D_features: np.ndarray, alpha_smoothing: float = 3.0) -> np.ndarray:
    """
    Computes theta length scales using the median heuristic on the distance matrix.
    """
    n_features = D_features.shape[2]
    n_samples = D_features.shape[0]
    theta = np.zeros(n_features)
    for i in range(n_features):
        dist_triu = D_features[:, :, i][np.triu_indices(n_samples, k=1)]
        med = np.median(dist_triu)
        if med == 0.0:
            med = np.mean(dist_triu)
            if med == 0.0:
                med = 1.0

        smoothed_length_scale = alpha_smoothing * med
        theta[i] = 1.0 / (2 * smoothed_length_scale**2)
    return theta


def _compute_theta_kta(D_features: np.ndarray, Y: np.ndarray, theta_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Optimizes theta length scales by maximizing the Kernel Target Alignment (KTA).
    """
    n_samples = D_features.shape[0]
    n_features = D_features.shape[2]

    Y_dist = np.abs(Y - Y.T)
    med_Y = np.median(Y_dist[np.triu_indices(n_samples, k=1)])
    if med_Y == 0.0:
        med_Y = 1.0
    K_Y = np.exp(-(Y_dist**2) / (2 * med_Y**2))

    # Fast centering for K_Y
    K_Y_row = K_Y.mean(axis=1, keepdims=True)
    K_Y_col = K_Y.mean(axis=0, keepdims=True)
    Kc_Y = K_Y - K_Y_row - K_Y_col + K_Y.mean()

    norm_Kc_Y = np.linalg.norm(Kc_Y, "fro")
    D2_features = D_features**2

    def kta_objective(theta: np.ndarray) -> float:
        Kc_X_sum = np.zeros((n_samples, n_samples))
        for i in range(n_features):
            if theta[i] > 1e-6:
                K_i = np.exp(-theta[i] * D2_features[:, :, i])
                K_row = K_i.mean(axis=1, keepdims=True)
                K_col = K_i.mean(axis=0, keepdims=True)
                Kc_i = K_i - K_row - K_col + K_i.mean()
                Kc_X_sum += Kc_i

        inner_prod = np.sum(Kc_X_sum * Kc_Y)
        norm_Kc_X = np.linalg.norm(Kc_X_sum, "fro")

        if norm_Kc_X < 1e-10:
            return 0.0
        return -(inner_prod / (norm_Kc_X * norm_Kc_Y))

    if theta_mask is not None:
        theta_init = np.copy(theta_mask)
        bounds = [(0.0, 0.0) if theta_mask[i] <= 0.001 else (0.001, 100.0) for i in range(n_features)]
    else:
        theta_init = _compute_theta_median_heuristic(D_features, alpha_smoothing=2.0)
        bounds = [(0.0, 100.0) for _ in range(n_features)]

    res = minimize(
        kta_objective,
        theta_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 50, "disp": False},
    )

    theta_opt = res.x
    theta_opt[theta_opt < 1e-3] = 0.0
    return theta_opt


class HsicAnovaAdsg:
    """
    HSIC-ANOVA Explainer for hierarchical design spaces (ADSG).

    Computes global sensitivity indices using Kernel methods adapted for
    structured and mixed-integer spaces.
    """

    def __init__(self, model: Any, ds: Any) -> None:
        """
        Initialize the explainer with a surrogate model and its design space.

        Parameters
        ----------
        model : smt.surrogate_models.surrogate_model.SurrogateModel
            A surrogate model that implements `predict_values` or `predict`.
        ds : smt.utils.design_space.DesignSpace
            The structural definition of the design space.
        """
        self.model = model
        self.ds = ds

        # Parse design space to extract mask characteristics using provided logic
        self.num_is_decreed = np.array(self.ds.is_conditionally_acting, dtype=bool)
        self.is_categorical = np.array(
            [isinstance(v, CategoricalVariable) for v in self.ds.design_variables],
            dtype=bool,
        )

    def explain(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        max_order: int = 3,
        use_kta: bool = False,
        theta_scales: Optional[np.ndarray] = None,
        use_smt_theta: bool = True,
        var_names: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Computes HSIC-ANOVA decomposition up to max_order on the given inputs.

        Parameters
        ----------
        X : np.ndarray
            Input features array of shape (n_samples, n_features).
        y : np.ndarray, optional
            Output values of shape (n_samples, 1). If not provided, will be computed using self.model.
        max_order : int, optional
            Maximum interaction order to compute. Default is 3.
        use_kta : bool, optional
            Whether to optimize kernel length scales using Kernel Target Alignment.
        theta_scales : np.ndarray, optional
            Pre-computed length scales to use.
        use_smt_theta : bool, optional
            Whether to strictly use the provided `theta_scales`.
        var_names : list of str, optional
            Names of variables to use in results.

        Returns
        -------
        tuple
            - list of dict: Contains the ANOVA decomposition terms.
            - float: Total HSIC trace.
        """
        if y is None:
            try:
                y = self.model.predict_values(X)
            except AttributeError:
                y = self.model.predict(X)

        # Get dynamic activity matrix for the inputs
        _, x_is_acting = self.ds.correct_get_acting(X)
        x_is_acting = np.array(x_is_acting, dtype=bool)

        return self._hsic_anova_hierarchical(
            X,
            y,
            x_is_acting,
            theta_scales,
            var_names,
            max_order,
            use_smt_theta,
            use_kta,
        )

    def _hsic_anova_hierarchical(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        x_is_acting: np.ndarray,
        theta_scales: Optional[np.ndarray],
        var_names: Optional[List[str]],
        max_order: int,
        use_smt_theta: bool,
        use_kta: bool,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Internal routine to compute HSIC-ANOVA decomposition.
        """
        n_samples, n_features = X.shape

        if Y.ndim == 1:
            Y = Y.reshape(-1, 1)
        D_Y = np.abs(Y[:, None, :] - Y[None, :, :]).sum(axis=-1)
        l_Y = np.std(Y) if np.std(Y) > 0 else 1.0
        L = _compute_kernel_from_dist(D_Y, l_Y)
        Lc = _center_kernel(L)

        D_X_features = _apply_algebraic_distance(
            X_num=X,
            Y_num=X,
            x_num_is_acting=x_is_acting,
            y_num_is_acting=x_is_acting,
            num_is_decreed=self.num_is_decreed,
            is_categorical=self.is_categorical,
        )

        if use_kta and theta_scales is not None:
            theta = _compute_theta_kta(D_X_features, Y, theta_mask=theta_scales)
        elif use_kta:
            theta = _compute_theta_kta(D_X_features, Y)
        elif theta_scales is not None and use_smt_theta:
            theta = theta_scales
        else:
            theta = _compute_theta_median_heuristic(D_X_features)

        base_centered_kernels = []
        for i in range(n_features):
            K_i = np.exp(-theta[i] * (D_X_features[:, :, i] ** 2))
            Kc_i = _center_kernel(K_i)
            base_centered_kernels.append(Kc_i)

        K_global = np.ones((n_samples, n_samples))
        for Kc_i in base_centered_kernels:
            K_global *= 1 + Kc_i
        K_global -= 1

        total_trace = np.sum(K_global * Lc) / ((n_samples - 1) ** 2)

        results = []

        for order in range(1, max_order + 1):
            for combo in combinations(range(n_features), order):
                joint_acting = np.ones(n_samples, dtype=bool)

                K_A = base_centered_kernels[combo[0]].copy()
                if self.num_is_decreed[combo[0]]:
                    joint_acting &= x_is_acting[:, combo[0]]

                for idx in combo[1:]:
                    K_A *= base_centered_kernels[idx]
                    if self.num_is_decreed[idx]:
                        joint_acting &= x_is_acting[:, idx]

                p_A = np.mean(joint_acting)

                K_A *= Lc
                trace_val = np.sum(K_A) / ((n_samples - 1) ** 2)

                adj_trace_val = trace_val / (p_A**2 + 1e-8)

                if trace_val > 0.0001 * total_trace:
                    name = " & ".join([var_names[i] for i in combo]) if var_names else f"Combo {combo}"
                    results.append(
                        {
                            "order": order,
                            "combo": combo,
                            "name": name,
                            "trace": float(trace_val),
                            "adj_trace": float(adj_trace_val),
                            "p_A": float(p_A),
                        }
                    )

        results.sort(key=lambda x: x["adj_trace"], reverse=True)
        return results, float(total_trace)
