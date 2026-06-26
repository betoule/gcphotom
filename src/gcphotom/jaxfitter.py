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
