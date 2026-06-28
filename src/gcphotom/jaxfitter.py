"""The module provides two second order methods to solve non-linear
problems
"""

import time

import jax
import jax.numpy as jnp


def flatten_vector(v):
    """Transforms a vector with a pytree structure into a standard array"""
    return jnp.hstack([jnp.ravel(v[p]) for p in v])


def unflatten_vector(p, v):
    """Give a standard array v the exact same pytree structure as p"""
    st = {}
    i = 0
    for k in p:
        j = i + jnp.size(p[k])
        st[k] = jnp.reshape(v[i:j], jnp.shape(p[k]))
        i = j
    return st


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
