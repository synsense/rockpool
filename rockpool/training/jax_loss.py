"""
Jax functions useful for training networks using Jax Modules.

See Also:
    See :ref:`/in-depth/jax-training.ipynb` for an introduction to training networks using Jax-backed modules in Rockpool, including the functions in `.jax_loss`.
"""

import jax.numpy as np

import jax.tree_util as tu

from typing import Tuple

from copy import deepcopy

import jax.random as random
from jax.lax import stop_gradient
import jax.numpy as jnp
from jax import grad, value_and_grad

from typing import Callable, Tuple, Dict, List
from jax.lib import pytree
from jax.tree_util import tree_structure

def mse(output: np.array, target: np.array) -> float:
    """
    Compute the mean-squared error between output and target

    This function is designed to be used as a component in a loss function. It computes the mean-squared error

    .. math::

        \\textrm{mse}(y, \\hat{y}) = { E[{(y - \\hat{y})^2}] }

    where :math:`E[\\cdot]` is the expectation of the expression within the brackets.

    Args:
        output (np.ndarray): The network output to test, with shape ``(T, N)``
        target (np.ndarray): The target output, with shape ``(T, N)``

    Returns:
        float: The mean-squared-error cost
    """
    return np.mean((output - target) ** 2)


def sse(output: np.array, target: np.array) -> float:
    """
    Compute the sum-squared error between output and target

    This function is designed to be used as a component in a loss function. It computes the mean-squared error

    .. math::

        \\textrm{sse}(y, \\hat{y}) = \\Sigma {(y - \\hat{y})^2}

    Args:
        output (np.ndarray): The network output to test, with shape ``(T, N)``
        target (np.ndarray): The target output, with shape ``(T, N)``

    Returns:
        float: The sum-squared-error cost
    """
    return np.sum((output - target) ** 2)


def make_bounds(params: dict) -> Tuple[dict, dict]:
    """
    Convenience function to build a bounds template for a problem

    This function works hand-in-hand with :py:func:`.bounds_cost`, to enforce minimum and/or maximum parameter bounds. :py:func:`.make_bounds` accepts a set of parameters (e.g. as returned from the :py:meth:`Module.parameters` method), and returns a ready-made dictionary of bounds (with no restrictions by default).

    See Also:
        See :ref:`/in-depth/jax-training.ipynb` for examples for using :py:func:`.make_bounds` and :py:func:`.bounds_cost`.

    :py:func:`.make_bounds` returns two dictionaries, representing the lower and upper bounds respectively. Initially all entries will be set to ``-np.inf`` and ``np.inf``, indicating that no bounds should be enforced. You must edit these dictionaries to set the bounds.

    Args:
        params (dict): Dictionary of parameters defining an optimisation problem. This can be provided as the parameter dictionary returned by :py:meth:`Module.parameters`.

    Returns:
        (dict, dict): ``lower_bounds``, ``upper_bounds``. Each dictionary mimics the structure of ``params``, with initial bounds set to ``-np.inf`` and ``np.inf`` (i.e. no bounds enforced).
    """
    # - Make copies
    lower_bounds = deepcopy(params)
    upper_bounds = deepcopy(params)

    # - Reset to -inf and inf
    lower_bounds = tu.tree_map(lambda _: -np.inf, lower_bounds)
    upper_bounds = tu.tree_map(lambda _: np.inf, upper_bounds)

    return lower_bounds, upper_bounds


def bounds_cost(params: dict, lower_bounds: dict, upper_bounds: dict) -> float:
    """
    Impose a cost on parameters that violate bounds constraints

    This function works hand-in-hand with :py:func:`.make_bounds` to enforce greater-than and less-than constraints on parameter values. This is designed to be used as a component of a loss function, to ensure parameter values fall in a reasonable range.

    :py:func:`.bounds_cost` imposes a value of 1.0 for each parameter element that exceeds a bound infinitesimally, increasing exponentially as the bound is exceeded, or 0.0 for each parameter within the bounds. You will most likely want to scale this by a penalty factor within your cost function.

    Warnings:
        :py:func:`.bounds_cost` does **not** clip parameters to the bounds. It is possible for parameters to exceed the bounds during optimisation. If this must be prevented, you should clip the parameters explicitly.

    See Also:
        See :ref:`/in-depth/jax-training.ipynb` for examples for using :py:func:`.make_bounds` and :py:func:`.bounds_cost`.

    Args:
        params (dict): A dictionary of parameters over which to impose bounds
        lower_bounds (dict): A dictionary of lower bounds for parameters matching your model, modified from that returned by :py:func:`.make_bounds`
        upper_bounds (dict): A dictionary of upper bounds for parameters matching your model, modified from that returned by :py:func:`.make_bounds`

    Returns:
        float: The cost to include in the cost function.
    """
    # - Flatten all parameter dicts
    params, tree_def_params = tu.tree_flatten(params)
    lower_bounds, tree_def_minparams = tu.tree_flatten(lower_bounds)
    upper_bounds, tree_def_maxparams = tu.tree_flatten(upper_bounds)

    if len(params) != len(lower_bounds) != len(upper_bounds):
        raise KeyError(
            "`lower_bounds` and `upper_bounds` must have the same keys as `params`."
        )

    # - Define a bounds function
    def bound(p, lower, upper):
        lb_cost_all = np.exp(-(p - lower))
        ub_cost_all = np.exp(-(upper - p))

        lb_cost = np.nansum(np.where(p < lower, lb_cost_all, 0.0))
        ub_cost = np.nansum(np.where(p > upper, ub_cost_all, 0.0))

        return lb_cost + ub_cost

    # - Map bounds function over parameters and return
    return np.sum(np.array(list(map(bound, params, lower_bounds, upper_bounds))))


def l2sqr_norm(params: dict) -> float:
    """
    Compute the mean L2-squared-norm of the set of parameters

    This function computes the mean :math:`L_2^2` norm of each parameter. The gradient of :math:`L_2^2(x)` is defined everywhere, where the gradient of :math:`L_2(x)` is not defined at :math:`x = 0`.

    The function is given by

    .. math::

        L_2^2(x) = E[x^2]

    where :math:`E[\\cdot]` is the expecation of the expression within the brackets.

    Args:
        params (dict): A Rockpool parameter dictionary

    Returns:
        float: The mean L2-sqr-norm of all parameters, computed individually for each parameter
    """
    # - Compute the L2 norm of each parameter individually
    params, _ = tu.tree_flatten(params)
    l22_norms = np.array(list(map(lambda p: np.nanmean(p ** 2), params)))

    # - Return the mean of each L2-sqr norm
    return np.nanmean(l22_norms)


def l0_norm_approx(params: dict, sigma: float = 1e-4) -> float:
    """
    Compute a smooth differentiable approximation to the L0-norm

    The :math:`L_0` norm estimates the **sparsity** of a vector -- i.e. the number of non-zero elements. This function computes a smooth approximation to the :math:`L_0` norm, for use as a component in cost functions. Including this cost will encourage parameter sparsity, by penalising non-zero parameters.

    The approximation is given by

    .. math::

        L_0(x) = \\frac{x^4}{x^4 + \\sigma}

    where :math:`\\sigma`` is a small regularisation value (by default ``1e-4``).

    Args:
        params (dict): A parameter dictionary over which to compute the L_0 norm
        sigma (float): A small value to use as a regularisation parameter. Default: ``1e-4``.

    Returns:
        float: The estimated L_0 norm cost
    """
    params, _ = tu.tree_flatten(params)
    return np.nanmean(
        np.array(
            list(
                map(
                    lambda p: np.nanmean(np.atleast_2d(p ** 4 / (p ** 4 + sigma))),
                    params,
                )
            )
        )
    )


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Implements the softmax function

    .. math::

        S(x, \\tau) = \\exp(l / \\tau) / { \\Sigma { \\exp(l / \\tau)} }

        l = x - \\max(x)

    Args:
        x (np.ndarray): Input vector of scores
        temperature (float): Temperature :math:`\\tau` of the softmax. As :math:`\\tau \\rightarrow 0`, the function becomes a hard :math:`\\max` operation. Default: ``1.0``.

    Returns:
        np.ndarray: The output of the softmax.
    """
    logits = x - np.max(x)
    eta = np.exp(logits / temperature)
    return eta / np.sum(eta)


def logsoftmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Efficient implementation of the log softmax function

    .. math ::

        log S(x, \\tau) = (l / \\tau) - \\log \\Sigma { \\exp (l / \\tau) }

        l = x - \\max (x)

    Args:
        x (np.ndarray): Input vector of scores
        temperature (float): Temperature :math:`\\tau` of the softmax. As :math:`\\tau \\rightarrow 0`, the function becomes a hard :math:`\\max` operation. Default: ``1.0``.

    Returns:
        np.ndarray: The output of the logsoftmax.
    """
    logits = x - np.max(x)
    return (logits / temperature) - np.log(np.sum(np.exp(logits / temperature)))

def split_and_sample(key : np.ndarray, shape: Tuple) -> Tuple[np.ndarray,np.ndarray]:
    """
    Split the key and generate random data following a standard Gaussian distribution of given shape

    Args:
        key (np.ndarray): Array of two ints. A Jax random key
        shape (tuple): The shape that the random normal data should have
    
    Returns:
        (np.ndarray, np.ndarray): Tuple of `(key,data)`. `key` is the new key that can be used in subsequent computations and `data` is the Gaussian data
    """
    key, subkey = random.split(key)
    val = random.normal(subkey, shape=shape)
    return key, val

def robustness_loss(theta_star: List,
                    inputs: np.ndarray,
                    output_theta: np.ndarray,
                    net: Callable[[np.ndarray],np.ndarray],
                    tree_def_params: tree_structure,
                    boundary_loss: Callable[[np.ndarray,np.ndarray],float]) -> float:
    """
    Calculate the robustness loss of the adversarial attack given Theta*. This function resets the states of the network, unflattens the parameters `theta_star` and assigns them to the network `net`
    and then evaluates the network on the `inputs` using the adversarial weights. Following, the `boundary_loss` is evaluated using the outputs of the network using the original parameters `output_theta`
    and the newly generated outputs. The method returns the boundary loss. This method is used by the adversary in PGA (projected gradient ascent).

    Args:
        theta_star (List): Flattened pytree that was obtained using `jax.tree_util.tree_flatten`
        inputs (np.ndarray): Inputs that will be passed through the network
        output_theta (np.ndarray): Outputs of the network using the original weights
        net (Callable): A function (e.g. `Sequential` object) that takes an `np.ndarray` and generates another `np.ndarray`
        tree_def_params (tree_structure): Tree structure obtained by calling `jax.tree_util.tree_flatten` on `theta_star_unflattened`. Basically defining the shape of `theta`/`theta_star`
        boundary_loss (Callable): Boundary loss. Takes as input two `np.ndarray` s and returns a `float`. Example: KL divergence between softmaxed logits of the networks

    Returns:
        float: The `boundary_loss` evaluated on the outputs generated by the network using :math:`\Theta` and the outputs of the network using :math:`\Theta^*`
    """
    # - Reset the network
    net = net.reset_state()

    # - Set the network parameters to Theta*
    net = net.set_attributes(tu.tree_unflatten(tree_def_params, theta_star))

    # - Evolve the network using Theta*
    output_theta_star, _, _ = net(inputs)

    # - Return boundary loss
    return boundary_loss(output_theta, output_theta_star)

def pga_attack(params_flattened: List,
            net: Callable[[np.ndarray],np.ndarray],
            rand_key: np.ndarray,
            attack_steps: int,
            mismatch_level: float,
            initial_std: float,
            inputs: np.ndarray,
            output_theta: np.ndarray,
            tree_def_params: tree_structure,
            boundary_loss: Callable[[np.ndarray,np.ndarray],float]
            ) -> Tuple[List,Dict]:
    """
    Performs the PGA (projected gradient ascent) based attack on the parameters of the network given inputs.

    Args:
        params_flattened (List): Flattened pytree that was obtained using `jax.tree_util.tree_flatten` of the network parameters (obtained by `net.parameters()`)
        net (Callable): A function (e.g. `Sequential` object) that takes an `np.ndarray` and generates another `np.ndarray`
        rand_key (np.ndarray): Array of two `int` s. A Jax random key
        attack_steps (int): Number of PGA steps to be taken
        mismatch_level (float): Size by which the adversary can perturb the weights (:math:`\zeta`). Attack will be in :math:`[\Theta-\zeta \cdot |\Theta|,\Theta+\zeta \cdot |\Theta|]`
        initial_std (float): Initial perturbation (:math:`\zeta_{initial}`) of the parameters according to :math:`\Theta + \zeta_{initial} \cdot R \odot |\Theta| \; ; R \sim \mathcal{N}(0,\mathbf{I})`
        inputs (np.ndarray): Inputs that will be passed through the network
        output_theta (np.ndarray): Outputs of the network using the original weights
        tree_def_params (tree_structure): Tree structure obtained by calling `jax.tree_util.tree_flatten` on `theta_star_unflattened`. Basically defining the shape of `theta`/`theta_star`
        boundary_loss (Callable): Boundary loss. Takes as input two `np.ndarray` s and returns a `float`. Example: KL divergence between softmaxed logits of the networks
    
    Returns:
        Tuple[List,Dict]: Tuple comprising :math:`\Theta^*` in flattened form and a dictionary holding the `grads` and `losses` for every PGA iteration
    """
    # - Create verbose dict
    verbose = {
        "grads":[],
        "losses":[]
    }
    # - Initialize Theta*
    theta_star = []
    step_size =  []
    for p in params_flattened:
        rand_key, random_normal_var = split_and_sample(rand_key, p.shape)
        theta_star.append(p + jnp.abs(p) * initial_std*random_normal_var)
        step_size.append((mismatch_level * jnp.abs(p)) / attack_steps)

    # - Perform the attack on Theta using initialized Theta*
    for _ in range(attack_steps):
        loss, grads_theta_star = value_and_grad(robustness_loss)(theta_star, inputs, output_theta, net, tree_def_params, boundary_loss)
        verbose["losses"].append(loss)
        verbose["grads"].append(grads_theta_star)
        for idx in range(len(theta_star)):
            theta_star[idx] = theta_star[idx] + step_size[idx] * jnp.sign(grads_theta_star[idx])

    return theta_star, verbose

def adversarial_loss(parameters: pytree,
                    net: Callable[[np.ndarray],np.ndarray],
                    inputs: np.ndarray,
                    target: np.ndarray,
                    training_loss: Callable[[np.ndarray,np.ndarray],float],
                    boundary_loss: Callable[[np.ndarray,np.ndarray],float],
                    rand_key: np.ndarray,
                    noisy_forward_std: float,
                    initial_std: float,
                    mismatch_level: float,
                    beta_robustness: float,
                    attack_steps: int,
                    ):
    """
    Implement the loss of the form :math:`\mathcal{L} = \mathcal{L}_{nat}(f(X,\Theta),y) + \\beta_{rob} \cdot \mathcal{L}_{rob}(f(X,\Theta),f(X,\mathcal{A}(\Theta)))`
    where :math:`\mathcal{A}(\Theta)` is an PGA-based adversary and :math:`\Theta` are the weights of the input that are perturbed by Gaussian noise during the forward pass.

    Args:
        parameters (pytree): Parameters of the network (obtained by e.g. `net.parameters()`)
        net (Callable): A function (e.g. `Sequential` object) that takes an `np.ndarray` and generates another `np.ndarray`
        inputs (np.ndarray): Inputs that will be passed through the network
        target (np.ndarray): Targets for the network prediction. Can be anything as long as `training_loss` can cope with the type/shape
        training_loss (Callable): Training loss. Can be anything used for training a NN (e.g. cat. cross entropy). Expects `net(inputs),target` as inputs
        boundary_loss (Callable): Boundary loss. Takes as input two `np.ndarray` s and returns a `float`. Example: KL divergence between softmaxed logits of the networks
        rand_key (np.ndarray): Array of two `int` s. A Jax random key
        noisy_forward_std (float): Float (:math:`\zeta_{forward}`) determining the amound of noise added to the parameters in the forward pass of the network. Model: :math:`\Theta = \Theta + \zeta_{forward} \cdot R \odot |\Theta| \; ; R \sim \mathcal{N}(0,\mathbf{I})`
        initial_std (float): Initial perturbation (:math:`\zeta_{initial}`) of the parameters according to :math:`\Theta + \zeta_{initial} \cdot R \odot |\Theta| \; ; R \sim \mathcal{N}(0,\mathbf{I})`
        mismatch_level (float): Size by which the adversary can perturb the weights (:math:`\zeta`). Attack will be in :math:`[\Theta-\zeta \cdot |\Theta|,\Theta+\zeta \cdot |\Theta|]`
        beta_robustness (float): Tradeoff parameter for the adversarial regularizer. Setting to `0.0` trains without adversarial loss but is much slower and should not be done.
        attack_steps (int): Number of PGA steps to be taken

    Returns:
        float: The calculated loss
    """
    # - Handle the network state — randomise or reset
    net = net.reset_state()

    # - Add Gaussian noise to the parameters before evaluating
    params_flattened, tree_def_params = tu.tree_flatten(parameters)

    params_gaussian_flattened = []
    for p in params_flattened:
        rand_key, random_normal_var = split_and_sample(rand_key, p.shape)
        params_gaussian_flattened.append(p + stop_gradient(jnp.abs(p) * noisy_forward_std*random_normal_var))

    # - Reshape the new parameters
    params_gaussian = tu.tree_unflatten(tree_def_params, params_gaussian_flattened)
    net = net.set_attributes(params_gaussian)

    # - Evolve the network to get the ouput
    output, _, _ = net(inputs)

    loss_n = training_loss(output, target)

    # - Get output for normal Theta
    # - Reset network state
    net = net.reset_state()

    # - Set parameters to the original parameters
    net = net.set_attributes(parameters)

    # - Get the network output using the original parameters
    output_theta, _, _ = net(inputs)

    theta_star, _ = pga_attack(params_flattened=params_flattened,
                            net=net,
                            rand_key=rand_key,
                            attack_steps=attack_steps,
                            mismatch_level=mismatch_level,
                            initial_std=initial_std,
                            inputs=inputs,
                            output_theta=output_theta,
                            tree_def_params=tree_def_params,
                            boundary_loss=boundary_loss)

    # - Compute robustness loss using final Theta*
    loss_r = robustness_loss(theta_star, inputs, output_theta, net, tree_def_params, boundary_loss)
    
    # - Add the robustness loss as a regularizer
    return loss_n + beta_robustness * loss_r