"""
Module for the unsupervised learning of control signals with DMDc
(unDMDc): DMD with control where the control input ``U`` is discovered
from the data instead of being provided.

The model is the same as in :class:`pydmd.DMDc`:

.. math::

    X_2 = A X_1 + B U

but ``U`` (and its sparsity pattern) are learned jointly with ``A`` and
``B`` by sequential least squares with hard thresholding, minimizing

.. math::

    \\min_{A, B, U} \\| A X_1 + B U - X_2 \\|_F + \\lambda \\| U \\|_0

A path of control signals with decreasing sparsity is computed; the
best point along the path is selected with an information criterion
(AIC by default) or other heuristics.

References:
- Fieseler, C., Zimmer, M. and Kutz, J.N., 2020. Unsupervised learning
  of control signals and their encodings in Caenorhabditis elegans
  whole-brain recordings. J. R. Soc. Interface, 17(173), p.20200459.
- Proctor, J.L., Brunton, S.L. and Kutz, J.N., 2016. Dynamic mode
  decomposition with control. SIAM J. Appl. Dyn. Syst., 15(1), 142-161.

Ported from the MATLAB implementation:
https://github.com/Charles-Fieseler/Learning_Control_Signals_MATLAB
"""

import warnings

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from sklearn.decomposition import NMF

from .dmdc import DMDc
from .snapshots import Snapshots
from .utils import compute_rank


class unDMDc(DMDc):
    """
    Dynamic Mode Decomposition with unsupervised control discovery.

    After :func:`fit`, the following are available:

    - :func:`unDMDc.all_U`, :func:`unDMDc.all_A`, :func:`unDMDc.all_B`:
      the sparsity path of control signals and dynamics, one entry per
      iteration;
    - :func:`unDMDc.U`, :func:`unDMDc.A`, :func:`unDMDc.B`: the entries
      selected by :func:`unDMDc.calc_best_control_signal`;
    - :func:`unDMDc.best_index`, :func:`unDMDc.objective_values`.

    The DMD eigenvalues/modes (``eigs``, ``modes``, ...) and
    :func:`reconstructed_data` come from the underlying DMDc fit at the
    selected point of the path, exactly like a :class:`pydmd.DMDc`
    fitted with the learned control.

    :param int num_controls: number of control signals (rows of `U`) to
        search for (`r_ctr` in the MATLAB code).
    :param int num_iter: number of sparsification iterations.
    :param float removal_fraction: fraction of non-zero entries removed
        per iteration (`iter_removal_fraction`).
    :param float re_up_fraction: fraction of entries added back per
        iteration (`iter_re_up_fraction`), experimental. Default is 0.
    :param bool only_positive_U: initialize with a non-negative
        factorization and treat negative entries of `U` as removable
        during thresholding.
    :param bool threshold_total_U: apply the hard threshold to the
        whole `U` matrix (weighted by the effect of `B`) instead of row
        by row.
    :param str init: initialization of `U`, either 'nmf' (non-negative
        matrix factorization of the naive DMD residual, as in MATLAB)
        or 'svd' (right singular vectors of the residual).
    :param numpy.ndarray custom_U: custom initialization for the
        control signals (shape (num_controls, m - lag)); overrides the
        initialization options above.
    :param bool smooth_initialization: smooth the residual before
        factorizing it (moving average then Gaussian).
    :param tuple initialization_smoothing_windows: (moving average,
        twice the Gaussian sigma) windows used by
        `smooth_initialization`.
    :param bool smooth_controller: Gaussian-smooth `U` at each
        iteration (rescaled so each row keeps a maximum of 1).
    :param int svd_rank: rank truncation for the final DMDc fit, see
        :class:`pydmd.DMDc`.
    :param int tlsq_rank: rank truncation for Total Least Square, see
        :class:`pydmd.DMDc`.
    :param opt: amplitudes computation flag, see :class:`pydmd.DMDc`.
    :param int svd_rank_omega: rank truncation of the augmented matrix
        omega, see :class:`pydmd.DMDc`.
    :param int lag: time lag between the snapshots.
    :param int seed: random seed used by the stochastic
        initialization.
    :param bool verbose: print the progress of the path computation.
    """

    def __init__(
        self,
        num_controls=15,
        num_iter=80,
        removal_fraction=0.05,
        re_up_fraction=0.0,
        only_positive_U=True,
        threshold_total_U=False,
        init="nmf",
        custom_U=None,
        smooth_initialization=False,
        initialization_smoothing_windows=(3, 6),
        smooth_controller=False,
        svd_rank=0,
        tlsq_rank=0,
        opt=False,
        svd_rank_omega=-1,
        lag=1,
        seed=13,
        verbose=False,
    ):
        super().__init__(
            svd_rank=svd_rank,
            tlsq_rank=tlsq_rank,
            opt=opt,
            svd_rank_omega=svd_rank_omega,
            lag=lag,
        )
        if init not in ("nmf", "svd"):
            raise ValueError("init must be either 'nmf' or 'svd'.")
        if num_controls < 1:
            raise ValueError("num_controls must be a positive integer.")
        if not 0 < removal_fraction <= 1:
            raise ValueError("removal_fraction must be in (0, 1].")

        self._num_controls = num_controls
        self._num_iter = num_iter
        self._removal_fraction = removal_fraction
        self._re_up_fraction = re_up_fraction
        self._only_positive_U = only_positive_U
        self._threshold_total_U = threshold_total_U
        self._init = init
        self._custom_U = custom_U
        self._smooth_initialization = smooth_initialization
        self._initialization_smoothing_windows = (
            initialization_smoothing_windows
        )
        self._smooth_controller = smooth_controller
        self._seed = seed
        self._verbose = verbose

        self._all_U = []
        self._all_A = []
        self._all_B = []
        self._best_index = None
        self._objective_function = None
        self._objective_values = None
        self._path_errors = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def num_controls(self):
        """Number of control signals searched for."""
        return self._num_controls

    @property
    def all_U(self):
        """List of control signal matrices along the sparsity path."""
        return self._all_U

    @property
    def all_A(self):
        """List of intrinsic dynamics matrices along the sparsity path."""
        return self._all_A

    @property
    def all_B(self):
        """List of control matrices along the sparsity path."""
        return self._all_B

    @property
    def best_index(self):
        """Index of the best model along the path."""
        return self._best_index

    @property
    def objective_function(self):
        """Name of the objective used to pick `best_index`."""
        return self._objective_function

    @property
    def objective_values(self):
        """Objective values along the path (higher is better)."""
        return self._objective_values

    @property
    def path_errors(self):
        """Spectral / Frobenius one-step errors along the path."""
        return self._path_errors

    @property
    def U(self):
        """Best control signal matrix (q x (m - lag))."""
        self._check_best()
        return self._all_U[self._best_index]

    @property
    def A(self):
        """Best intrinsic dynamics matrix (n x n)."""
        self._check_best()
        return self._all_A[self._best_index]

    @property
    def B(self):
        """Best control matrix (n x q), from the sparsification path."""
        self._check_best()
        return self._all_B[self._best_index]

    def _check_best(self):
        if self._best_index is None:
            raise RuntimeError(
                "No best control signal selected yet; call fit() or "
                "calc_best_control_signal()."
            )

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------
    def fit(self, X, objective="aic", opt=None):
        """
        Compute the path of sparse control signals, then fit a DMDc
        model at the best point of the path.

        :param X: the input snapshots, space by row and time by column.
        :type X: numpy.ndarray
        :param str objective: objective used to select the best model
            along the path, see :func:`calc_best_control_signal`. Pass
            None to skip the selection (it can be run later).
        :param opt: options passed to the objective function.
        :return: self
        :rtype: unDMDc
        """
        self._reset()
        self._all_U, self._all_A, self._all_B = [], [], []
        self._best_index = None
        self._objective_values = None

        self._snapshots_holder = Snapshots(X)
        n_samples = self.snapshots.shape[-1]
        if self._lag < 1:
            raise ValueError("Time lag must be positive.")
        X = self.snapshots
        X1 = X[:, : -self._lag]
        X2 = X[:, self._lag :]
        self._set_initial_time_dictionary(
            {"t0": 0, "tend": n_samples - 1, "dt": 1}
        )

        self._learn_path(X1, X2)

        if objective is not None:
            self.calc_best_control_signal(objective, opt)
        return self

    def _learn_path(self, X1, X2):
        """Sequential thresholding loop (port of learn_control_signals)."""
        n, m = X1.shape
        q = self._num_controls

        U = self._initialize_u(X1, X2)
        sparsity_pattern = np.zeros(U.shape, dtype=bool)

        self._all_U = [None] * self._num_iter
        self._all_A = [None] * self._num_iter
        self._all_B = [None] * self._num_iter
        all_err = np.zeros((self._num_iter, 2))
        self._path_errors = all_err

        sparse_func = np.abs if not self._only_positive_U else _identity
        has_stalled = False
        i = 0
        for i in range(self._num_iter):
            if self._verbose:
                print(f"Iteration {i + 1}/{self._num_iter}")

            # Step 1: get A and B given U.
            AB = _fit_ab(X1, X2, U)
            A, B = AB[:, :n], AB[:, n:]
            all_err[i, 0] = np.linalg.norm(A @ X1 + B @ U - X2, 2)
            if i > 0:
                self._all_A[i - 1] = A
                self._all_B[i - 1] = B

            # Step 2: sequential least squares + hard thresholding on U.
            U = np.linalg.lstsq(B, X2 - A @ X1, rcond=None)[0]
            U_nonsparse = U.copy()
            U[sparsity_pattern] = 0
            U_nonsparse[~sparsity_pattern] = 0
            re_up_pattern = np.zeros(U.shape, dtype=bool)
            has_stalled = True

            if self._threshold_total_U:
                # Normalize U by the effect it has on the data via B.
                U_effective = np.zeros_like(U)
                for i2 in range(q):
                    U_effective[i2] = U[i2] * np.abs(B[:, i2]).sum()
                U_eff_nonsparse = U_effective.copy()
                U_eff_nonsparse[~sparsity_pattern] = 0
                U_effective[sparsity_pattern] = 0
                tmp = sparse_func(U_effective).ravel()
                positive = tmp[tmp > 0]
                if positive.size > 0:
                    has_stalled = False
                    threshold = np.quantile(
                        positive, self._removal_fraction
                    )
                else:
                    threshold = 0.0
                sparsity_pattern = sparse_func(U_effective) < threshold
                if self._re_up_fraction > 0:
                    tmp2 = sparse_func(U_eff_nonsparse).ravel()
                    positive2 = tmp2[tmp2 > 0]
                    if positive2.size > 0:
                        threshold_top = max(
                            np.quantile(
                                positive2, 1 - self._re_up_fraction
                            ),
                            np.median(tmp),
                        )
                        re_up_pattern = (
                            sparse_func(U_eff_nonsparse) > threshold_top
                        )
            else:
                # Threshold per row of U.
                for i2 in range(q):
                    tmp = U[i2]
                    positive = tmp[tmp > 0]
                    if positive.size == 0:
                        threshold = 0.0
                    else:
                        has_stalled = False
                        threshold = max(
                            np.quantile(
                                positive, self._removal_fraction
                            ),
                            np.min(positive),
                        )
                    sparsity_pattern[i2] = sparse_func(tmp) <= threshold

                    if self._re_up_fraction > 0:
                        tmp2 = U_nonsparse[i2]
                        positive2 = tmp2[tmp2 > 0]
                        if positive2.size > 0:
                            threshold_top = max(
                                np.quantile(
                                    positive2, 1 - self._re_up_fraction
                                ),
                                np.median(tmp),
                            )
                            re_up_pattern[i2] = (
                                sparse_func(tmp2) > threshold_top
                            )

            if self._re_up_fraction > 0:
                sparsity_pattern = sparsity_pattern & ~re_up_pattern
            U[sparsity_pattern] = 0

            if self._smooth_controller:
                U = gaussian_filter1d(
                    U, sigma=1.0, axis=1, mode="nearest"
                )
                row_max = U.max(axis=1, keepdims=True)
                U = np.divide(
                    U, row_max, out=np.zeros_like(U), where=row_max > 0
                )

            self._all_U[i] = U.copy()
            all_err[i, 1] = np.linalg.norm(A @ X1 + B @ U - X2, "fro")
            if self._verbose:
                print(
                    "Number of nonzero control signals: "
                    f"{np.count_nonzero(U)}"
                )
            if has_stalled:
                if self._verbose:
                    print(
                        "All control signals are 0. Stopping early "
                        "(this is nothing to worry about)."
                    )
                break

        if has_stalled:
            # Fill the rest of the path to keep the output consistent.
            prev = max(i - 1, 0)
            if i == 0:
                AB = _fit_ab(X1, X2, self._all_U[0])
                self._all_A[0] = AB[:, :n]
                self._all_B[0] = AB[:, n:]
            for i2 in range(i, self._num_iter):
                self._all_U[i2] = self._all_U[prev].copy()
                self._all_A[i2] = self._all_A[prev]
                self._all_B[i2] = self._all_B[prev]
        else:
            # Final exact DMDc fit with the last control signal.
            AB = _fit_ab(X1, X2, self._all_U[-1])
            self._all_A[-1] = AB[:, :n]
            self._all_B[-1] = AB[:, n:]

    def _initialize_u(self, X1, X2):
        """Build the initial (dense) guess for U."""
        n, m = X1.shape
        q = self._num_controls
        if self._custom_U is not None:
            U = np.asarray(self._custom_U, dtype=float)
            if U.shape != (q, m):
                raise ValueError(
                    f"custom_U must have shape {(q, m)}, got {U.shape}."
                )
            return U.copy()

        # Initialize with the residuals of a naive DMD fit.
        err = X2 - (X2 @ np.linalg.pinv(X1)) @ X1
        err = np.real(err)
        if self._smooth_initialization:
            w_ma, w_gauss = self._initialization_smoothing_windows
            err = uniform_filter1d(err, size=w_ma, axis=1, mode="nearest")
            err = gaussian_filter1d(
                err, sigma=w_gauss / 2, axis=1, mode="nearest"
            )

        if self._only_positive_U and self._init == "nmf":
            # NMF requires a non-negative input; clip the negative part
            # of the naive DMD residual (as in MATLAB's nnmf usage).
            model = NMF(
                n_components=q, random_state=self._seed, max_iter=500
            )
            model.fit_transform(np.maximum(err, 0))
            H = model.components_
            U = H[:q]
            row_max = U.max(axis=1, keepdims=True)
            U = np.divide(
                U, row_max, out=np.zeros_like(U), where=row_max > 0
            )
            return U

        _, s, Vh = np.linalg.svd(err, full_matrices=False)
        if self._init == "svd" and q > s.size:
            raise ValueError(
                f"num_controls ({q}) exceeds the rank of the "
                f"initialization residual ({s.size})."
            )
        return Vh[: min(q, s.size)].copy()

    # ------------------------------------------------------------------
    # Best model selection along the path
    # ------------------------------------------------------------------
    def calc_best_control_signal(self, objective="aic", opt=None):
        """
        Select the best control signal along the path, according to the
        given objective (higher value is better). Then the underlying
        DMDc model (eigs, modes, reconstructed_data, ...) is fitted with
        the selected control signal.

        Implemented objectives (see the reference paper):

        - 'aic': Akaike information criterion on the 2-step prediction
          error, penalizing only the non-zero entries of `U` (default);
          `opt = (num_steps, do_aicc, formula_mode)`.
        - 'aic_window': AIC with a sparsity down-weighting derived from
          a time `window`; `opt = (num_steps, window)`.
        - 'acf': autocorrelation of the control signals.
        - 'variance_explained_by_A': variance of X2 explained by A.
        - 'variance_explained_by_B': variance explained by B alone.
        - 'variance_explained_ratio_A_to_B': ratio of the two above.

        :param str objective: name of the objective function.
        :param opt: options passed to the objective function, as a
            tuple or as a single value.
        :return: self
        :rtype: unDMDc
        """
        if not self._all_U:
            raise RuntimeError("fit() has not been called.")
        objectives = {
            "aic": self._objective_aic,
            "aic_window": self._objective_aic_window,
            "acf": self._objective_acf,
            "variance_explained_by_A": (
                self._objective_variance_explained_by_A
            ),
            "variance_explained_by_B": (
                self._objective_variance_explained_by_B
            ),
            "variance_explained_ratio_A_to_B": (
                self._objective_variance_explained_ratio_A_to_B
            ),
        }
        if objective not in objectives:
            raise ValueError(
                f"Unknown objective {objective!r}, expected one of "
                f"{sorted(objectives)}."
            )
        self._objective_function = objective

        func = objectives[objective]
        if opt is None:
            args = ()
        elif isinstance(opt, tuple):
            args = opt
        else:
            args = (opt,)

        vals = np.array([func(i, *args) for i in range(len(self._all_U))])
        # The last point of the path reuses the previous fit (as in the
        # MATLAB implementation, where it is a placeholder).
        vals[-1] = vals[-2]
        self._objective_values = vals
        self._best_index = int(np.argmax(vals))

        # Fit the underlying DMDc model with the selected control.
        super().fit(self.snapshots, self.U)
        return self

    def _objective_aic(
        self, i, num_steps=2, do_aicc=None, formula_mode="standard"
    ):
        return -aic_2step_dmdc(
            self.snapshots,
            self._all_U[i],
            self._all_A[i],
            self._all_B[i],
            num_steps=num_steps,
            do_aicc=False if do_aicc is None else do_aicc,
            formula_mode=formula_mode,
        )

    def _objective_aic_window(self, i, num_steps=2, window=10):
        lam = self.snapshots.shape[1] / window
        return -aic_multi_step_dmdc(
            self.snapshots,
            self._all_U[i],
            self._all_A[i],
            self._all_B[i],
            num_steps=num_steps,
            formula_mode="window",
            lam=lam,
        )[0]

    def _objective_acf(self, i, _=None):
        return _acf(self._all_U[i])

    def _objective_variance_explained_by_A(self, i, _=None):
        X1 = self.snapshots[:, : -self._lag]
        X2 = self.snapshots[:, self._lag :]
        residual = X2 - self._all_A[i] @ X1
        return np.mean(np.var(residual, axis=1) / np.var(X2, axis=1))

    def _objective_variance_explained_by_B(self, i, _=None):
        X2 = self.snapshots[:, self._lag :]
        residual = X2 - self._all_B[i] @ self._all_U[i]
        return 1 - np.mean(np.var(residual, axis=1) / np.var(X2, axis=1))

    def _objective_variance_explained_ratio_A_to_B(self, i, _=None):
        eps = 1e-8
        a = self._objective_variance_explained_by_A(i)
        b = self._objective_variance_explained_by_B(i)
        return a / (1 - b + eps)

    # ------------------------------------------------------------------
    # Cross-validation along the path
    # ------------------------------------------------------------------
    def calc_cross_validation_error(self, k=4, num_error_steps=1):
        """
        Compute the cross-validation error along the path, for a given
        number of folds and error steps.

        :param int k: number of folds (chaining scheme, see the MATLAB
            ``dmdc_cross_val``).
        :param int num_error_steps: number of prediction steps whose
            error is averaged.
        :return: array of shape (len(all_U), num_error_steps, k - 1)
        :rtype: numpy.ndarray
        """
        if not self._all_U:
            raise RuntimeError("fit() has not been called.")
        all_err = np.zeros((len(self._all_U), num_error_steps, k - 1))
        for i, this_u in enumerate(self._all_U):
            _, all_err[i] = dmdc_cross_val(
                self.snapshots, this_u, k, num_error_steps
            )
        return all_err


# ----------------------------------------------------------------------
# Module level helpers, ported from the MATLAB repository
# ----------------------------------------------------------------------
def _identity(x):
    return x


def _fit_ab(X1, X2, U):
    """Exact (least squares) DMDc fit: [A B] = X2 [X1; U]^dagger."""
    omega = np.vstack([X1, U])
    return X2 @ np.linalg.pinv(omega)


def _n_step_error(X, A, B, U, num_steps):
    """
    Port of ``calc_nstep_error.m``: n-step prediction error of a DMDc
    model. If `num_steps` is an integer the error is averaged over the
    steps 1..num_steps (inclusive), otherwise the errors are returned
    for each step in the given list.
    """
    inclusive = isinstance(num_steps, (int, np.integer))
    if inclusive:
        err_steps_to_save = list(range(1, int(num_steps) + 1))
    else:
        err_steps_to_save = [int(s) for s in num_steps]
    max_step = err_steps_to_save[-1]

    m = X.shape[1]
    X1 = X[:, : m - max_step]

    all_err = np.zeros(len(err_steps_to_save))
    X_hat = X1
    for i_step in range(1, max_step + 1):
        X_hat = A @ X_hat + B @ U[:, i_step - 1 : m - max_step + i_step - 1]
        if i_step in err_steps_to_save:
            idx = err_steps_to_save.index(i_step)
            all_err[idx] = np.linalg.norm(
                X_hat - X[:, i_step : m - max_step + i_step], "fro"
            )
    if inclusive:
        return float(np.mean(all_err))
    return all_err


def _noise_frobenius_norm(X):
    """
    Port of the noise estimate of ``calc_snr.m``: the Frobenius norm of
    the noise, estimated from the residual of the SVD truncated with
    the optimal hard threshold (Gavish & Donoho 2014).
    """
    _, s, _ = np.linalg.svd(X, full_matrices=False)
    rank = compute_rank(X, svd_rank=0)
    return float(np.sqrt(np.sum(s[rank:] ** 2)))


def _my_aic(formula_mode, do_aicc, RSS, k, n, num_signals, A, B, U, X, lam=1):
    """Port of ``my_aic.m``; see :func:`aic_2step_dmdc` for the meaning
    of the parameters."""
    if formula_mode == "standard":
        aic_out = 2 * num_signals * k + 2 * n + n * np.log(RSS)
    elif formula_mode == "multivariate":
        n = X.shape[0]
        aic_out = 2 * num_signals * k + 2 * n + n * np.log(RSS)
    elif formula_mode == "window":
        n = X.shape[0]
        aic_out = 2 * num_signals * k + 2 * n + lam * n * np.log(RSS)
    elif formula_mode in ("stanford", "stanford2"):
        err_cov = _noise_frobenius_norm(X) / n
        if err_cov == 0:
            warnings.warn("Noise estimate is zero, AIC is unreliable.")
            err_cov = np.finfo(float).eps
        aic_out = 2 * k * num_signals + RSS / err_cov
        if formula_mode == "stanford2":
            aic_out = 2 * k * num_signals + 2 * RSS / err_cov
    elif formula_mode == "one_step":
        aic_out = 2 * num_signals * (k / n + n) + n * np.log(RSS)
    elif formula_mode == "one_step_stanford":
        err_cov = _noise_frobenius_norm(X) / n
        if err_cov == 0:
            err_cov = np.finfo(float).eps
        aic_out = 2 * k * num_signals / n + RSS / err_cov
    else:
        raise ValueError(f"Unrecognized formula mode {formula_mode!r}.")

    if do_aicc:
        k_t = (
            np.count_nonzero(U) + np.count_nonzero(A) + np.count_nonzero(B)
        )
        correction = (2 * k_t**2 + 2 * k_t) / abs(n - k_t - 1)
        aic_out += correction
        if k_t > n:
            warnings.warn(
                "Number of parameters is larger than data; AICc may be "
                "difficult to interpret."
            )
    return aic_out


def aic_2step_dmdc(
    X, U, A=None, B=None, num_steps=2, do_aicc=False, formula_mode="stanford"
):
    """
    Port of ``aic_2step_dmdc.m``: AIC of a DMDc model, computed on the
    n-step prediction error and penalizing only the non-zero entries of
    the control signal (the effective number of parameters is
    ``nnz(U) / q``).

    :param numpy.ndarray X: the data matrix (n x m).
    :param numpy.ndarray U: the control signals (q x (m - 1)).
    :param numpy.ndarray A: the dynamics matrix, recomputed if None.
    :param numpy.ndarray B: the control matrix, recomputed if None.
    :param int num_steps: number of prediction steps for the error.
    :param bool do_aicc: apply the AICc finite-sample correction.
    :param str formula_mode: one of 'standard', 'stanford',
        'stanford2', 'one_step', 'one_step_stanford'.
    :return: the AIC value (lower is better).
    :rtype: float
    """
    X1 = X[:, : X.shape[1] - num_steps]
    if A is None or B is None:
        AB = _fit_ab(X[:, :-1], X[:, 1:], U)
        n = X.shape[0]
        A, B = AB[:, :n], AB[:, n:]

    RSS = _n_step_error(X, A, B, U, num_steps)
    RSS = max(RSS, np.finfo(float).eps)

    num_signals = U.shape[0]
    k = np.count_nonzero(U) / num_signals
    n = X1.shape[1]

    if formula_mode == "standard":
        n = X1.size
        aic = 2 * num_signals * (k + n) + n * np.log(RSS)
    else:
        aic = _my_aic(
            formula_mode, False, RSS, k, n, num_signals, A, B, U, X
        )

    if do_aicc:
        k_t = (
            np.count_nonzero(U) + np.count_nonzero(A) + np.count_nonzero(B)
        )
        aic += (2 * k_t**2 + 2 * k_t) / abs(n - k_t - 1)
    return float(aic)


def aic_multi_step_dmdc(
    X, U, A=None, B=None, num_steps=2, do_aicc=False,
    formula_mode="stanford", lam=1,
):
    """
    Port of ``aic_multi_step_dmdc.m``: AIC of a DMDc model evaluated at
    several prediction steps at once.

    :param num_steps: int or list of prediction steps to evaluate.
    :param float lam: the 'window' formula hyperparameter.
    :return: one AIC value (lower is better) per requested step.
    :rtype: numpy.ndarray
    """
    if isinstance(num_steps, (int, np.integer)):
        num_steps = [int(num_steps)]
    X1 = X[:, : X.shape[1] - num_steps[-1]]
    if A is None or B is None:
        AB = _fit_ab(X[:, :-1], X[:, 1:], U)
        n = X.shape[0]
        A, B = AB[:, :n], AB[:, n:]

    RSS_vec = _n_step_error(X, A, B, U, num_steps)

    num_signals = U.shape[0]
    k = np.count_nonzero(U) / num_signals
    n = X1.shape[1]

    return np.array(
        [
            _my_aic(
                formula_mode, do_aicc, RSS, k, n, num_signals, A, B, U, X, lam
            )
            for RSS in RSS_vec
        ]
    )


def _acf(U):
    """
    Autocorrelation heuristic (MATLAB ``acf`` external dependency):
    mean lag-1 autocorrelation of the non-constant rows of ``U``.
    """
    U = np.atleast_2d(U)
    values = []
    for row in U:
        centered = row - row.mean()
        denom = np.dot(centered, centered)
        if denom == 0:
            continue
        values.append(np.dot(centered[1:], centered[:-1]) / denom)
    return float(np.mean(values)) if values else 0.0


def dmdc_cross_val(X, U, num_folds=4, err_steps=1, inclusive=True):
    """
    Port of ``dmdc_cross_val.m`` (chaining mode): cross-validation of a
    DMDc model with a given control signal, holding out contiguous
    blocks of the data.

    :param numpy.ndarray X: the data matrix (n x m).
    :param numpy.ndarray U: the control signals (q x (m - 1)).
    :param int num_folds: number of windows, hence num_folds - 1 folds.
    :param int err_steps: number of prediction steps for the error.
    :param bool inclusive: average the error over the steps 1..err_steps.
    :return: mean error and per-fold error array.
    :rtype: (float, numpy.ndarray)
    """
    if inclusive:
        err_steps_to_save = list(range(1, err_steps + 1))
    else:
        err_steps_to_save = [err_steps]
    max_step = err_steps_to_save[-1]

    m = X.shape[1]
    window_starts = np.round(np.linspace(0, m - 1, num_folds + 1)).astype(int)
    window_starts = window_starts[1:]  # chaining
    folds = num_folds - 1

    all_err = np.zeros((len(err_steps_to_save), folds))
    n = X.shape[0]
    for f in range(folds):
        test_ind = np.arange(window_starts[f], window_starts[f + 1] + 1)
        train_ind = np.arange(0, window_starts[f] - max_step)

        X1 = X[:, train_ind]
        X2 = X[:, train_ind + 1]
        U1 = U[:, train_ind]

        AB = _fit_ab(X1, X2, U1)
        A, B = AB[:, :n], AB[:, n:]

        X1_t = X[:, test_ind]
        U1_t = U[:, test_ind[0] : test_ind[-1]]
        all_err[:, f] = _n_step_error(
            X1_t, A, B, U1_t, err_steps_to_save
        )

    if inclusive:
        all_err = all_err.mean(axis=0)
    return float(all_err.mean()), all_err
