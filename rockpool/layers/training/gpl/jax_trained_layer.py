#
# Implements a base class for Jax layers, that supports training
#

# - Import base classes
from rockpool.layers.layer import Layer
from abc import abstractmethod, ABC

# - Import and define types
from typing import Dict, Tuple, Any, Callable, Union, List

State = Any
Params = Union[Dict, Tuple, List]

__all__ = ['JaxTrainedLayer']

# - Import jax elements
from jax import numpy as np

class JaxTrainedLayer(Layer, ABC):
    """
    Base class for a trainable layer, with evolution functions based on Jax

    Derive from this base class to implement a new Jax-backed trainable layer.

    .. rubric:: How to train a layer based on this class

    This class defines a training method `~.train_output_target`, which performs one round of optimization based on a single trial, given an input and a target signal::

        lyr.train_output_target(input_ts, target_ts)

    .. rubric:: How to use this base class

    This class defines three abstract methods that you need to define for your layer::

        def _pack(self) -> Params:
            return {
                'param_a': self._param_a,
                'param_b': self._param_b,
                ...
            }

    The ``_pack()`` method must return a dictionary or tuple or other collection of strictly jax-compatible types, that completely define the modifiable parameters for this layer. For example: weights; biases; time constants. Included should be all parameters that one might want to perform gradient descent on. *Excluded* should be parameters that are fixed: for example ``dt``, and ``noise_std``.

    ::

        def _unpack(self, params: Params) -> None:
            (
                self._param_a,
                self._param_b,
                ...
            ) = (
                params['param_a'],
                params['param_b'],
                ...
            )

    The ``_unpack()`` method must accept a parameters definition as returned by ``_pack()``, and apply those parameters to the layer.

    ::

        @property
        def _evolve_functional(self) -> Callable[]:
            return evol_func

    The property ``_evolve_functional`` must return a *function* ``evol_func()`` with the following calling signature. This function must evolve the state of the layer, given an initial state, set of parameters and raw inputs, with *no side effects*. That means the function must not update the internal state of the layer, or update the `._t` attribute, etc. The function ``evol_func()`` must be compilable with `jax.jit`. An example property and function are shown here::

        @property
        def _evolve_functional(self) -> Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State]]:
            def evol_func(params: Params, state: State, input: np.ndarray) -> Tuple[np.ndarray, State]:
            '''
            :param Params params:       `params` contains the set of parameters needed to define this layer
            :param State state:         `state` contains the initial state of this layer
            :param np.ndarray input:    `input` is [TxI], T time steps by I input channels

            :return Tuple[np.ndarray, State]: (output, new_state)
                output:     A raw time series [TxO], T time steps by O output channels
                new_state:  The new state of the layer, after the evolution
            '''
                # - Perform evolution inner loop
                output, new_state = f(input, state)

                # - Return output and state
                return output, new_state

            return evol_func
    """
    @abstractmethod
    def _pack(self) -> Params:
        """
        Method returning a list or tuple or dict of Jax / numpy base classes, containing the tunable parameters of this layer

        You must override this abstract method when implementing your own concrete Jax layer class

        :return Params: params: list, tuple or dict of parameters
        """
        pass

    @abstractmethod
    def _unpack(self, params: Params) -> None:
        """
        Method that sets the internal parameters of this layer, given a set of parameters returned from :py:meth:`._pack`

        :param Params params:   list, tuple or dict of parameters
        """
        pass

    @property
    @abstractmethod
    def _evolve_functional(self) -> Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State]]:
        """
        Functional form of evolution for this layer

        This abstract property must return a function ``evol_func``, which evolves the dynamics of this layer given parameters, input and an initial state. The function must have the calling signature::

            def evol_func(params: Params, state: State, input: np.ndarray) -> Tuple[np.ndarray, State]:
                ...

        ``evol_func`` returns ``(outputs, new_state)``, and *must* be side-effect free. The goal is that ``evol_func`` can be compiled using `jax.jit`.

        :return Callable[[Params, State, np.ndarray)], Tuple[np.ndarray, State]]: evol_func: Evolution function
        """

        def evol_func(params: Params, state: State, input: np.ndarray) -> Tuple[np.ndarray, State]:
            pass

        return evol_func

