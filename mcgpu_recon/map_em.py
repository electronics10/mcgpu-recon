"""
MAP-EM (penalized-likelihood EM) reconstruction.

Maximizes   Phi(x) = L(x) - beta * R(x),   x >= 0
with the Poisson log-likelihood
    L(x) = sum_i ( y_i log ybar_i - ybar_i ),   ybar = mult * (A x) + contamination
and a neighbourhood penalty R from priors.py.

Three update rules (see WORKING_PLAN_year2_v2.md, section 3.4):

* "depierro" (De Pierro 1995), for QUADRATIC penalties. An MM method: each
  (x_j - x_k)^2 is replaced by a separable upper bound, so every voxel gets
  its own 1D problem, solved exactly as the positive root of
        a x^2 + b x - c = 0,
        a = 2 beta Omega_j
        b = s_j - beta sum_k omega_jk (x_j^old + x_k^old)
        c = s_j * x_EM,j         (x_EM = the plain MLEM update)
  Phi never decreases, x stays >= 0, and beta = 0 gives exactly MLEM.

* "gradient", for any differentiable penalty (default for RDP):
  EM-preconditioned gradient ascent with an exact line search,
        p = (x / s) * grad Phi(x),     x <- x + alpha p,
  alpha maximizes Phi along p subject to x + alpha p >= 0. With beta = 0 and
  alpha = 1 this is exactly the MLEM step. Phi never decreases. Because
  A(x + alpha p) = A x + alpha A p, one extra forward projection per iteration
  makes the line search projection-free: ~1.5x the cost of an MLEM iteration.

* "osl" (Green 1990, one-step-late), kept for comparison:
  x_j <- s_j x_EM,j / (s_j + beta dR/dx_j(x_old)).
  No convergence guarantee; the denominator can approach 0 or go negative for
  large beta. It is clamped at osl_floor_frac * s_j, and the number of clamped
  voxels is reported. In tests, OSL + RDP diverged already at moderate beta.

Unlike MLEM, MAP is meant to be run close to CONVERGENCE (often 100-300
iterations); beta, not the iteration number, controls the smoothness.
"""

from __future__ import annotations

import numpy as np

from .mlem import sensitivity_and_support, expected_counts, em_backprojection


def map_em(A, y, prior=None, beta=0.0, n_iter=100, method="auto", x0=None,
           mult=None, contamination=None, sens_floor_frac=0.025,
           osl_floor_frac=0.05, eps=1e-8, verbose=False, callback=None,
           return_history=False):
    """MAP-EM reconstruction.

    Parameters
    ----------
    A : linear operator with __call__ (forward), .adjoint, .in_shape,
        .out_shape and .xp (e.g. MCGPUProjector).
    y : measured sinogram, shape A.out_shape, counts >= 0.
    prior : QuadraticPrior or RDPrior from mcgpu_recon.priors, or None (= MLEM).
    beta : penalty strength >= 0. Use priors.beta_scale() to find a sensible
        range (beta = beta~ * beta0 with beta~ ~ 0.01 ... 10).
    n_iter : number of iterations.
    method : "auto" (De Pierro if the prior is quadratic, else "gradient"),
        "depierro", "gradient", or "osl".
    x0 : initial image (default: ones on the support).
    mult, contamination : as in mlem() (attenuation/normalization factors;
        additive scatter/randoms expectation).
    sens_floor_frac : voxels with sensitivity below this fraction of the
        maximum are outside the support and held at 0 (as in mlem()). Pairs
        touching such voxels are removed from the penalty.
    osl_floor_frac : OSL only; lower clamp of the denominator, as a fraction of s_j.
    callback : optional f(k, x) after each iteration.
    return_history : if True, also return a dict with, per iteration k, the
        log-likelihood, penalty and objective Phi of the image BEFORE update k
        (computed at no extra projection cost), plus OSL clamp counts.

    Returns
    -------
    x, or (x, history) if return_history.
    """
    xp = getattr(A, "xp", np)
    y = xp.asarray(y, dtype=xp.float32)
    use_prior = prior is not None and beta > 0

    if method == "auto":
        method = "depierro" if (use_prior and prior.quadratic) else "gradient"
    if use_prior and method == "depierro" and not prior.quadratic:
        raise ValueError("De Pierro's update needs a quadratic prior; "
                         "use method='gradient'.")
    if method not in ("depierro", "gradient", "osl"):
        raise ValueError(f"unknown method {method!r}")

    # ---- sensitivity and support (shared with mlem) -----------------------
    mult_arr, sens, support, sens_safe = sensitivity_and_support(
        A, mult, sens_floor_frac, eps)
    m = None if mult is None else mult_arr

    if use_prior:
        prior = prior.restrict(support)

    x = xp.ones(A.in_shape, dtype=xp.float32) if x0 is None \
        else xp.asarray(x0, dtype=xp.float32)
    x = xp.where(support, x, 0.0)

    hist = {"loglik": [], "penalty": [], "objective": [], "n_clamped": []}

    for k in range(n_iter):
        # ---- E-step part: forward model and back-projected ratio ----------
        ybar = expected_counts(A, x, m, contamination)

        if return_history or verbose:
            ybar_safe = xp.maximum(ybar, eps)
            ll = float(xp.sum(xp.where(y > 0, y * xp.log(ybar_safe), 0.0) - ybar,
                              dtype=xp.float64))
            pen = prior.value(x) if use_prior else 0.0
            hist["loglik"].append(ll)
            hist["penalty"].append(pen)
            hist["objective"].append(ll - beta * pen)

        c = em_backprojection(A, x, y, ybar, m, eps)     # = s_j * x_EM,j

        # ---- M-step -------------------------------------------------------
        n_clamped = 0
        if not use_prior:
            x_new = c / sens_safe                                   # MLEM
        elif method == "depierro":
            Om, S = prior.depierro_terms(x)
            a = 2.0 * beta * Om
            b = sens_safe - beta * S
            disc = xp.sqrt(b * b + 4.0 * a * c)
            # positive root of a x^2 + b x - c = 0, in the numerically stable
            # form for each sign of b (a = 0 only if b = s_j > 0)
            x_pos_b = 2.0 * c / xp.maximum(b + disc, eps)
            x_neg_b = (disc - b) / xp.maximum(2.0 * a, eps)
            x_new = xp.where(b > 0, x_pos_b, x_neg_b)
        elif method == "gradient":
            x_new = _line_search_step(A, x, y, ybar, c, sens_safe, support,
                                      prior, beta, m, contamination, eps, xp)
        else:                                                       # OSL
            g = prior.gradient(x)
            den = sens_safe + beta * g
            floor = osl_floor_frac * sens_safe
            clamped = (den < floor) & support
            n_clamped = int(xp.sum(clamped))
            x_new = c / xp.maximum(den, floor)

        x = xp.where(support, x_new, 0.0)
        hist["n_clamped"].append(n_clamped)

        if verbose:
            msg = (f"  MAP-EM[{method if use_prior else 'mlem'}] iter {k+1:3d}/{n_iter}"
                   f"  Phi={hist['objective'][-1]:.8g}")
            if use_prior and method == "osl" and n_clamped:
                msg += f"  (OSL clamped {n_clamped} voxels)"
            print(msg)
        if callback is not None:
            callback(k, x)

    if use_prior and method == "osl" and max(hist["n_clamped"]) > 0:
        import warnings
        warnings.warn(f"OSL denominator was clamped (max {max(hist['n_clamped'])} "
                      "voxels in one iteration): beta is probably too large for OSL.")
    return (x, hist) if return_history else x


def _line_search_step(A, x, y, ybar, c, sens, support, prior, beta, mult,
                      contamination, eps, xp, n_search=30):
    """One step of EM-preconditioned gradient ascent with exact line search.

    grad Phi = A^T(mult * y/ybar) - s - beta grad R = c/x - s - beta grad R,
    so the direction p = (x/s) grad Phi = (c - s x - beta x grad R) / s.
    phi(alpha) = Phi(x + alpha p) is concave; we find its maximum on
    [0, alpha_max] (alpha_max keeps x >= 0) by bisection on phi'(alpha).
    """
    g = prior.gradient(x)
    p = xp.where(support, (c - sens * x - beta * x * g) / sens, 0.0)

    q = A(p)                                  # the one extra forward projection
    if mult is not None:
        q = q * mult
    yb0 = ybar                                # = mult*A x + contamination
    has_y = y > 0

    neg = p < 0
    if bool(xp.any(neg)):
        alpha_max = float(xp.min(xp.where(neg, -x / xp.where(neg, p, -1.0), xp.inf)))
    else:
        alpha_max = 1e6

    def dphi(alpha):
        yb = yb0 + alpha * q
        d_like = xp.sum(xp.where(has_y, y * q / xp.maximum(yb, eps), 0.0) - q,
                        dtype=xp.float64)
        d_pen = xp.sum(prior.gradient(xp.maximum(x + alpha * p, 0.0)) * p,
                       dtype=xp.float64)
        return float(d_like) - beta * float(d_pen)

    d0 = dphi(0.0)
    if d0 <= 0:                               # already stationary along p
        return x
    # bracket: grow from alpha = 1 (the MLEM step) until phi' < 0 or alpha_max
    lo, hi = 0.0, min(1.0, alpha_max)
    while dphi(hi) > 0:
        if hi >= alpha_max:
            return xp.maximum(x + alpha_max * p, 0.0)
        lo, hi = hi, min(2.0 * hi, alpha_max)
    for _ in range(n_search):                 # bisection on the sign of phi'
        mid = 0.5 * (lo + hi)
        if dphi(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4 * hi:
            break
    return xp.maximum(x + lo * p, 0.0)
