"""Tests for map_em / priors on a small dense problem.

Run from the repo root:   pixi run python -m pytest -q tests
"""
import numpy as np
import pytest
from scipy.optimize import minimize

from mcgpu_recon import (mlem, map_em, adjoint_test, binomial_thin,
                         uniform_weights, bowsher_weights, QuadraticPrior,
                         RDPrior, beta_scale)
from mcgpu_recon.priors import _all_offsets, _pair_slices

SHAPE = (5, 6, 7)                 # (Nz, Ny, Nx), 210 voxels
N = int(np.prod(SHAPE))


class DenseOp:
    """y = M x for an image of shape SHAPE; the stand-in for MCGPUProjector."""
    xp = np

    def __init__(self, M):
        self.M = M.astype(np.float32)
        self.in_shape = SHAPE
        self.out_shape = (M.shape[0],)

    def __call__(self, x):
        return (self.M @ np.asarray(x, np.float32).ravel()).astype(np.float32)

    def adjoint(self, y):
        return (self.M.T @ np.asarray(y, np.float32).ravel()).reshape(SHAPE).astype(np.float32)


def make_problem(seed=0, counts=2000.0):
    rng = np.random.default_rng(seed)
    m = 900
    M = rng.random((m, N)) * (rng.random((m, N)) < 0.08)
    A = DenseOp(M)
    x_true = np.ones(SHAPE, np.float32)
    x_true[1:4, 2:5, 2:6] = 4.0                       # hot block
    x_true[2, 1, 1] = 8.0                             # small hot spot
    lam = A(x_true)
    lam *= counts / lam.sum() * m / 50
    scale = counts / A(x_true).sum() * m / 50
    y = rng.poisson(lam).astype(np.float32)
    mr = (x_true > 1).astype(np.float32) + 0.05 * rng.standard_normal(SHAPE).astype(np.float32)
    return A, y, x_true * scale, mr


def dense_omega(weights):
    """Full N x N symmetric pair-weight matrix, built by brute force."""
    Om = np.zeros((N, N))
    idx = np.arange(N).reshape(SHAPE)
    for o, w in zip(weights.offsets, weights.W):
        sl_j, sl_k = _pair_slices(o, SHAPE)
        for j, k, v in zip(idx[sl_j].ravel(), idx[sl_k].ravel(), w[sl_j].ravel()):
            Om[j, k] += v; Om[k, j] += v
    return Om


def neg_objective(prior_kind, weights, A, y, beta, gamma=2.0):
    """-Phi(x) and its gradient, written independently from the package."""
    M = A.M.astype(np.float64)
    Om = dense_omega(weights)
    iu = np.triu_indices(N, 1)
    wj, jj, kk = Om[iu], iu[0], iu[1]
    sel = wj > 0
    wj, jj, kk = wj[sel], jj[sel], kk[sel]
    yy = y.astype(np.float64)

    def f(xv):
        yb = np.maximum(M @ xv, 1e-12)
        L = np.sum(yy * np.log(yb) - yb)
        gL = M.T @ (yy / yb - 1.0)
        d = xv[jj] - xv[kk]
        if prior_kind == "quad":
            R = 0.5 * np.sum(wj * d * d)
            gd = wj * d
            gR = np.zeros(N); np.add.at(gR, jj, gd); np.add.at(gR, kk, -gd)
        else:
            D = np.maximum(xv[jj] + xv[kk] + gamma * np.abs(d), 1e-12)
            R = np.sum(wj * d * d / D)
            gRj = wj * d * (xv[jj] + 3 * xv[kk] + gamma * np.abs(d)) / D**2
            gRk = -wj * d * (3 * xv[jj] + xv[kk] + gamma * np.abs(d)) / D**2
            gR = np.zeros(N); np.add.at(gR, jj, gRj); np.add.at(gR, kk, gRk)
        return -(L - beta * R), -(gL - beta * gR)
    return f


def reference_map(prior_kind, weights, A, y, beta, x0):
    f = neg_objective(prior_kind, weights, A, y, beta)
    res = minimize(f, x0.ravel().astype(np.float64), jac=True, method="L-BFGS-B",
                   bounds=[(0, None)] * N, options={"maxiter": 20000, "ftol": 1e-15, "gtol": 1e-10})
    return res.x.reshape(SHAPE), -res.fun


# ---------------------------------------------------------------------------

def test_adjoint_test_passes_for_exact_adjoint():
    A, *_ = make_problem()
    assert max(adjoint_test(A)) < 1e-5


def test_beta_zero_is_exactly_mlem():
    A, y, _, _ = make_problem()
    prior = QuadraticPrior(uniform_weights(SHAPE))
    x1 = mlem(A, y, n_iter=15, sens_floor_frac=0.0)
    x2 = map_em(A, y, prior=prior, beta=0.0, n_iter=15, sens_floor_frac=0.0)
    x3 = map_em(A, y, prior=None, n_iter=15, sens_floor_frac=0.0)
    assert np.array_equal(x1, x2) and np.array_equal(x1, x3)


def test_bowsher_B26_equals_uniform():
    mr = np.random.default_rng(1).random(SHAPE)
    wu = uniform_weights(SHAPE, distance_weighted=False)
    wb = bowsher_weights(mr, 26, distance_weighted=False)
    for a, b in zip(wu.W, wb.W):
        assert np.array_equal(a, b)


def test_bowsher_selection_brute_force():
    rng = np.random.default_rng(2)
    mr = rng.random(SHAPE)                     # distinct values: no ties
    B = 5
    wb = bowsher_weights(mr, B, distance_weighted=False)
    Om = dense_omega(wb)
    # brute force: directed choice, then average
    idx = np.arange(N).reshape(SHAPE)
    Wd = np.zeros((N, N))
    for z, yy, x in np.ndindex(*SHAPE):
        nb = []
        for o in _all_offsets():
            q = (z + o[0], yy + o[1], x + o[2])
            if all(0 <= q[i] < SHAPE[i] for i in range(3)):
                nb.append((abs(mr[z, yy, x] - mr[q]), idx[q]))
        for _, k in sorted(nb)[:B]:
            Wd[idx[z, yy, x], k] = 1
    assert np.allclose(Om, 0.5 * (Wd + Wd.T))


def test_bowsher_ties_are_isotropic_for_face_neighbours():
    # flat MR, B = 6: every interior voxel must pick exactly its 6 face neighbours
    wb = bowsher_weights(np.zeros(SHAPE), 6, distance_weighted=False)
    Om = dense_omega(wb)
    idx = np.arange(N).reshape(SHAPE)
    j = idx[2, 3, 3]
    faces = [idx[2 + a, 3 + b, 3 + c] for a, b, c in
             [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]]
    assert np.isclose(Om[j].sum(), 6.0)
    assert all(Om[j, k] == 1.0 for k in faces)


@pytest.mark.parametrize("kind", ["quad", "rdp"])
def test_prior_value_and_gradient(kind):
    rng = np.random.default_rng(3)
    w = bowsher_weights(rng.random(SHAPE), 8)
    prior = QuadraticPrior(w) if kind == "quad" else RDPrior(w, gamma=2.0)
    x = (rng.random(SHAPE) + 0.2).astype(np.float64)
    # value vs brute force
    f = neg_objective(kind, w, DenseOp(np.zeros((1, N))), np.zeros(1), beta=1.0)
    val_ref, _ = f(x.ravel())
    assert np.isclose(prior.value(x), val_ref, rtol=1e-6)     # -(0 - R) = R
    # gradient vs central finite differences
    g = prior.gradient(x)
    h = 1e-6
    for j in rng.choice(N, 20, replace=False):
        e = np.zeros(N); e[j] = h
        num = (prior.value((x.ravel() + e).reshape(SHAPE))
               - prior.value((x.ravel() - e).reshape(SHAPE))) / (2 * h)
        assert np.isclose(g.ravel()[j], num, rtol=1e-4, atol=1e-6)


@pytest.mark.parametrize("which", ["uniform", "bowsher"])
@pytest.mark.parametrize("beta_t", [0.1, 1.0, 20.0])
def test_depierro_monotone_and_converges_to_map(which, beta_t):
    A, y, x_true, mr = make_problem()
    w = uniform_weights(SHAPE) if which == "uniform" else bowsher_weights(mr, 8)
    prior = QuadraticPrior(w)
    x_ml = mlem(A, y, n_iter=20, sens_floor_frac=0.0)
    beta = beta_t * beta_scale(prior, A.adjoint(np.ones(A.out_shape)), x_ml)
    x, h = map_em(A, y, prior=prior, beta=beta, n_iter=3000,
                  sens_floor_frac=0.0, return_history=True)
    obj = np.array(h["objective"])
    assert x.min() >= 0
    # monotone up to float32 round-off
    assert np.all(np.diff(obj) >= -1e-6 * np.abs(obj[:-1]))
    x_ref, phi_ref = reference_map("quad", w, A, y, beta, x_ml)
    assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < 1e-3


@pytest.mark.parametrize("beta_t", [0.3, 3.0, 30.0])
def test_gradient_rdp_monotone_and_converges_to_map(beta_t):
    A, y, x_true, mr = make_problem()
    w = uniform_weights(SHAPE)
    prior = RDPrior(w, gamma=2.0)
    x_ml = mlem(A, y, n_iter=20, sens_floor_frac=0.0)
    beta = beta_t * beta_scale(prior, A.adjoint(np.ones(A.out_shape)), x_ml)
    x, h = map_em(A, y, prior=prior, beta=beta, n_iter=1500, x0=x_ml,
                  sens_floor_frac=0.0, return_history=True)       # auto -> gradient
    obj = np.array(h["objective"])
    assert x.min() >= 0
    assert np.all(np.diff(obj) >= -1e-6 * np.abs(obj[:-1]))
    x_ref, _ = reference_map("rdp", w, A, y, beta, x_ml)
    assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < 1e-2


def test_gradient_matches_depierro_for_quadratic():
    A, y, x_true, mr = make_problem()
    prior = QuadraticPrior(bowsher_weights(mr, 8))
    x_ml = mlem(A, y, n_iter=20, sens_floor_frac=0.0)
    beta = 1.0 * beta_scale(prior, A.adjoint(np.ones(A.out_shape)), x_ml)
    x1 = map_em(A, y, prior=prior, beta=beta, n_iter=2000, sens_floor_frac=0.0)
    x2 = map_em(A, y, prior=prior, beta=beta, n_iter=2000, sens_floor_frac=0.0,
                method="gradient", x0=x_ml)
    assert np.linalg.norm(x1 - x2) / np.linalg.norm(x1) < 1e-2


def test_osl_rdp_small_beta_reaches_map():
    A, y, x_true, mr = make_problem()
    prior = RDPrior(uniform_weights(SHAPE), gamma=2.0)
    x_ml = mlem(A, y, n_iter=20, sens_floor_frac=0.0)
    beta = 0.3 * beta_scale(prior, A.adjoint(np.ones(A.out_shape)), x_ml)
    x, h = map_em(A, y, prior=prior, beta=beta, n_iter=3000, method="osl",
                  sens_floor_frac=0.0, return_history=True)
    assert max(h["n_clamped"]) == 0
    x_ref, _ = reference_map("rdp", uniform_weights(SHAPE), A, y, beta, x_ml)
    assert np.linalg.norm(x - x_ref) / np.linalg.norm(x_ref) < 1e-2


@pytest.mark.parametrize("kind", ["quad", "rdp"])
def test_prior_finite_for_tiny_float32_values(kind):
    # MLEM drives background voxels towards ~1e-30; the penalty must stay finite
    rng = np.random.default_rng(4)
    w = uniform_weights(SHAPE)
    prior = QuadraticPrior(w) if kind == "quad" else RDPrior(w)
    x = (rng.random(SHAPE) * 1e-30).astype(np.float32)
    x[0, 0, 0] = 0.0
    x[2] = 5.0
    assert np.isfinite(prior.gradient(x)).all() and np.isfinite(prior.value(x))


def test_binomial_thin_mean():
    y = np.full(200000, 50.0, np.float32)
    t = binomial_thin(y, 0.1, seed=0)
    assert abs(t.mean() - 5.0) < 0.05 and abs(t.var() - 4.5) < 0.1
