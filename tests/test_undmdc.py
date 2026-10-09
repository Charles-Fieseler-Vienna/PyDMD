"""Tests for the unDMDc class (unsupervised control signal learning)."""
import numpy as np
import pytest

from pydmd import unDMDc


def make_synthetic(n=6, q=2, m=150, seed=0):
    """Linear system driven by short sparse control pulses."""
    rng = np.random.RandomState(seed)
    A = np.zeros((n, n))
    for i in range(n - 1):
        A[i, i + 1] = 0.5
    A[0, :] = 0.05
    B = rng.randn(n, q)
    U = np.zeros((q, m - 1))
    U[0, 20:25] = 1.0
    U[0, 70:75] = 1.0
    U[1, 110:116] = 1.0
    X = np.zeros((n, m))
    X[:, 0] = rng.randn(n)
    for k in range(m - 1):
        X[:, k + 1] = A @ X[:, k] + B @ U[:, k]
    return X, U, A, B


def support_rows(M):
    sup = np.zeros(M.shape[1], dtype=bool)
    for row in M:
        peak = max(row.max(), 1e-12)
        sup |= row > 0.5 * peak
    return sup


def test_init_and_properties():
    dmd = unDMDc(num_controls=3)
    assert dmd.num_controls == 3
    assert dmd._num_iter == 80
    assert dmd._removal_fraction == 0.05
    with pytest.raises(ValueError):
        unDMDc(num_controls=0)
    with pytest.raises(ValueError):
        unDMDc(num_controls=2, removal_fraction=1.5)
    with pytest.raises(ValueError):
        unDMDc(num_controls=2, init="bad")


def test_path_structure_and_sparsity():
    X, U, _, _ = make_synthetic()
    dmd = unDMDc(num_controls=2, num_iter=30, svd_rank=-1, seed=0)
    dmd.fit(X, objective=None)

    assert len(dmd.all_U) == 30
    assert len(dmd.all_A) == len(dmd.all_B) == 30
    for u, a, b in zip(dmd.all_U, dmd.all_A, dmd.all_B):
        assert u.shape == (2, X.shape[1] - 1)
        assert a.shape == (X.shape[0], X.shape[0])
        assert b.shape == (X.shape[0], 2)
    nnz = [np.count_nonzero(u) for u in dmd.all_U]
    assert nnz[0] > nnz[-1]
    assert all(nnz[i] >= nnz[i + 1] for i in range(len(nnz) - 1))
    # U property raises before selection.
    with pytest.raises(RuntimeError):
        dmd.U


def test_control_recovery():
    X, U, _, _ = make_synthetic()
    dmd = unDMDc(num_controls=2, num_iter=30, svd_rank=-1, seed=0)
    dmd.fit(X, objective="aic", opt=(2, False, "stanford"))

    assert dmd.best_index is not None
    assert dmd.objective_values.shape == (30,)
    sup, tru = support_rows(dmd.U), support_rows(U)
    assert (tru & sup).sum() == tru.sum()  # full recall
    assert (tru & sup).sum() == sup.sum()  # full precision


def test_reconstruction_and_modes():
    X, _, _, _ = make_synthetic()
    dmd = unDMDc(num_controls=2, num_iter=30, svd_rank=-1, seed=0)
    dmd.fit(X, objective="aic", opt=(2, False, "stanford"))
    rec = dmd.reconstructed_data()
    assert rec.shape == X.shape
    assert np.linalg.norm(rec - X) / np.linalg.norm(X) < 0.5
    assert dmd.eigs.shape == (X.shape[0],)
    assert dmd.modes.shape == (X.shape[0], X.shape[0])


def test_objectives_run():
    X, _, _, _ = make_synthetic(m=120)
    dmd = unDMDc(num_controls=2, num_iter=12, svd_rank=-1, seed=0)
    dmd.fit(X, objective=None)
    for objective in (
        "aic",
        "aic_window",
        "acf",
        "variance_explained_by_A",
        "variance_explained_by_B",
        "variance_explained_ratio_A_to_B",
    ):
        dmd.calc_best_control_signal(objective)
        assert 0 <= dmd.best_index < 12
    with pytest.raises(ValueError):
        dmd.calc_best_control_signal("not_an_objective")


def test_cross_validation():
    X, _, _, _ = make_synthetic(m=120)
    dmd = unDMDc(num_controls=2, num_iter=8, svd_rank=-1, seed=0)
    dmd.fit(X, objective=None)
    err = dmd.calc_cross_validation_error(k=4, num_error_steps=2)
    assert err.shape == (8, 2, 3)
    assert np.all(err >= 0)


def test_init_svd_and_custom_U():
    X, U, _, _ = make_synthetic()
    dmd = unDMDc(num_controls=2, num_iter=10, init="svd",
                 only_positive_U=False, svd_rank=-1, seed=0)
    dmd.fit(X, objective=None)
    assert len(dmd.all_U) == 10

    dmd = unDMDc(num_controls=2, num_iter=10, custom_U=U, svd_rank=-1)
    dmd.fit(X, objective=None)
    assert dmd.all_U[0] is not dmd._custom_U
    np.testing.assert_allclose(dmd.all_U[0], U, atol=1e-12)
    with pytest.raises(ValueError):
        unDMDc(num_controls=2, custom_U=np.zeros((2, 3))).fit(X)


def test_deterministic():
    X, _, _, _ = make_synthetic()
    run = lambda: unDMDc(num_controls=3, num_iter=6, svd_rank=-1,
                         seed=1).fit(X, objective=None)
    np.testing.assert_array_equal(run().all_U[0], run().all_U[0])


def test_inherited_dmdc_features():
    X, U, _, _ = make_synthetic()
    dmd = unDMDc(num_controls=2, num_iter=10, svd_rank=-1, seed=0)
    dmd.fit(X, objective="acf")
    mask = np.full(len(dmd.amplitudes), True, dtype=bool)
    mask[[0]] = False
    dmd.modes_activation_bitmask = mask
    assert dmd.modes_activation_bitmask[0] == False
    rec = dmd.reconstructed_data()
    assert rec.shape == X.shape


def test_lag_validation():
    X, _, _, _ = make_synthetic()
    dmd = unDMDc(num_controls=2, lag=0, num_iter=5)
    with pytest.raises(ValueError):
        dmd.fit(X)
