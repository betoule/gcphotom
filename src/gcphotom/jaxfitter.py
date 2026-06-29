"""Optimizers for growth-curve fitting."""

import time

import jax
import jax.numpy as jnp


def fit_adam(
    func, init_params, grad_func=None, learning_rate=5e-3, niter=200, tol=1e-3, **kwargs
):
    """simple Adam gradient descent using optax.adam

    Parameters
    ----------
    func: function
        function to minimize. Should return a float.

    learning_rate: float
        learning rate of the gradient descent.
        (careful, results can be sensitive to this parameter)

    init_param:
        entry parameter of the input func

    niter: int
        maximum number of iterations

    tol: None or float
        targeted func variations below which the iteration will stop

    **kwargs other func entries

    Returns
    -------
    list
        - best parameters
        - loss (array)
    """
    tstart = time.time()
    import optax

    # handle kwargs more easily
    func_ = lambda x: func(x, **kwargs)

    # Initialize the adam optimizer
    params = init_params
    optimizer = optax.adam(learning_rate)
    # Obtain the `opt_state` that contains statistics for the optimizer.
    opt_state = optimizer.init(params)

    if grad_func is None:
        grad_func = jax.jit(jax.grad(func_))  # get the derivative

    # and do the gradient descent
    losses = [func_(params)]

    timings = [0]
    for i in range(niter):
        current_grads = grad_func(params)
        updates, opt_state = optimizer.update(current_grads, opt_state)
        params = optax.apply_updates(params, updates)
        losses.append(func_(params))  # store the loss function
        timings.append(time.time() - tstart)
        if tol is not None and (i > 2 and ((losses[-2] - losses[-1]) < tol)):
            break
    timings = jnp.array(timings)
    return params, {"loss": jnp.array(losses), "timings": timings}


def tukey(c=4.685):
    """Tukey's bisquare (biweight) loss function.

    Parameters
    ----------
    c : float
        Scale parameter. Points with ``|x| > c`` contribute zero gradient.
        Default ``4.685`` provides 95% efficiency for Gaussian noise.

    Returns
    -------
    callable
        Loss function ``loss(x) -> per-element loss`` suitable for use as
        the ``loss`` argument of :meth:`gcphotom.Fitter.fit`.
    """

    def loss(x):
        mask = jnp.abs(x) <= c
        return jnp.where(mask, c**2 / 6 * (1 - (1 - (x / c) ** 2) ** 3), c**2 / 6)

    return loss


def pseudo_huber(c=1.0):
    """Pseudo-Huber loss function (smooth, C^\\infty robust loss).

    Parameters
    ----------
    c : float
        Scale parameter controlling the transition from quadratic to linear
        behaviour. Default ``1.0``.

    Returns
    -------
    callable
        Loss function ``loss(x) -> per-element loss``.
    """

    def loss(x):
        return c**2 * (jnp.sqrt(1 + (x / c) ** 2) - 1)

    return loss


def cauchy(c=1.0):
    """Cauchy (Lorentzian) loss function.

    Parameters
    ----------
    c : float
        Scale parameter. Default ``1.0``.

    Returns
    -------
    callable
        Loss function ``loss(x) -> per-element loss``.
    """

    def loss(x):
        return c**2 * jnp.log(1 + (x / c) ** 2)

    return loss


def nmad(arr):
    """Normalized Median Absolute Deviation, implemented in JAX.

    ``1.4826 * median(|arr - median(arr)|)``, a robust estimate of the
    standard deviation of *arr*.
    """
    med = jnp.median(arr)
    return 1.4826 * jnp.median(jnp.abs(arr - med))


def parameter_uncertainty(weighted_residuals_fn, params):
    """Parameter covariance via the Jacobian of weighted residuals.

    Uses
        Cov = (J_wr^T J_wr)^{-1}  scaled by nmad(wr)^2

    where J_wr = ∂wr/∂p is the Jacobian of the weighted residuals at the
    best-fit point.  For well-fitting models J_wr ≈ -diag(1/σ) · J, so
    J_wr^T J_wr ≈ J^T W J.  Scaling by the squared NMAD of the weighted
    residuals provides a robust alternative to the classical chi2/dof
    scaling that is less sensitive to outlier-contaminated fits.

    Parameters
    ----------
    weighted_residuals_fn : callable
        Function ``params -> 1-D array of weighted residuals (y - model)/σ``.
        Only the good (fitted) data points should be returned.
    params : pytree
        Best-fit parameters (any structure compatible with
        ``jax.flatten_util.ravel_pytree``).

    Returns
    -------
    cov : (n_params, n_params) ndarray
        Covariance matrix in flattened-parameter space.
    se : pytree
        Standard errors with the same structure as *params*.
    """
    from jax.flatten_util import ravel_pytree

    wr = weighted_residuals_fn(params)
    n_good = wr.shape[0]

    p0, unravel = ravel_pytree(params)
    n_params = p0.size

    if n_good <= n_params:
        se = jax.tree_util.tree_map(lambda x: jnp.full_like(x, jnp.nan), params)
        return jnp.full((n_params, n_params), jnp.nan), se

    def flat_fn(pf):
        return weighted_residuals_fn(unravel(pf))

    J = jax.jacfwd(flat_fn)(p0)
    JTJ = J.T @ J
    cov = jnp.linalg.inv(JTJ)
    cov = cov * nmad(wr) ** 2

    se_flat = jnp.sqrt(jnp.diag(cov))
    leaves, treedef = jax.tree_util.tree_flatten(params)
    se_leaves = []
    start = 0
    for leaf in leaves:
        a = jnp.asarray(leaf)
        size = a.size
        se_leaves.append(se_flat[start : start + size].reshape(a.shape))
        start += size
    se = jax.tree_util.tree_unflatten(treedef, se_leaves)

    return cov, se
