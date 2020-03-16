#
# stack_jax.py — Implement trainable stacks of jax layers
#

from rockpool import TimeSeries, TSContinuous
from rockpool.networks import Network
from rockpool.layers.training import JaxTrainedLayer
from typing import Tuple, List, Callable, Union, Dict, Sequence, Any

from jax import jit
from jax.experimental.optimizers import adam
import jax.numpy as np

Params = List
State = List


class JaxStack(Network, JaxTrainedLayer):
    """
    Build a network of Jax layers, supporting parameter optimisation



    """

    def __init__(self, layers: Sequence = None, dt=None, *args, **kwargs):
        """
        Encapsulate a stack of Jax layers in a single layer / network

        :param Sequence layers: A Sequence of layers to initialise the stack with
        :param float dt:        Unitary timestep to force on each of the sublayers
        """
        # - Check that the layers are subclasses of `JaxTrainedLayer`
        for layer in layers:
            assert isinstance(
                layer, JaxTrainedLayer
            ), "Each layer must inherit from the `JaxTrainedLayer` base class"

        # - Initialise super classes
        super().__init__(layers=layers, dt=dt, weights=[], *args, **kwargs)

        # - Initialise timestep
        self.__timestep: int = 0

        # - Get evolution functions
        self._all_evolve_funcs: List = [
            lyr._evolve_functional for lyr in self.evol_order
        ]

        # - Set sizes
        self._size_in: int = self.input_layer._size_in
        self._size_out: int = self.evol_order[-1]._size_out
        self._size: Any = []

    def evolve(
        self,
        ts_input: TSContinuous = None,
        duration: float = None,
        num_timesteps: int = None,
    ) -> TSContinuous:
        """

        :param ts_input:
        :param duration:
        :param num_timesteps:
        :return:
        """

        # - Prepare time base and inputs, using first layer
        time_base, ext_inps, num_timesteps = self.input_layer._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Evolve over layers
        new_states = []
        outputs = []
        inps = ext_inps
        for p, s, evol_func in zip(self._pack(), self.state, self._all_evolve_funcs):
            # - Evolve layer
            out, new_state = evol_func(p, s, inps)

            # - Record outputs and states
            outputs.append(out)
            new_states.append(new_state)

            # - Set up inputs for next layer
            inps = out

        # - Assign updated states
        self._states = new_states

        # - Update time stamps
        self._timestep += inps.shape[0]

        # - Wrap outputs as time series
        outputs_dict = {"external_input": TSContinuous(time_base, ext_inps)}
        for lyr, out in zip(self.evol_order, outputs):
            outputs_dict.update({lyr.name: TSContinuous(time_base, np.array(out))})

        # - Return a dictionary of outputs for all of the sublayers in this stack
        return outputs_dict

    def _pack(self) -> Params:
        """
        Return a set of parameters for all sublayers in this stack

        :return Params: params
        """
        # - Get lists of parameters
        return [lyr._pack() for lyr in self.evol_order]

    def _unpack(self, params: Params) -> None:
        """
        Apply a set of parameters to the sublayers in this stack

        :param Params params:
        """
        for layer, params in zip(self.evol_order, params):
            layer._unpack(params)

    def randomize_state(self) -> None:
        """
        Randomise the state of each sublayer
        """
        for lyr in self.evol_order:
            lyr.randomize_state()

    @property
    def _timestep(self) -> int:
        """(int) Current integer time step"""
        return self.__timestep

    @_timestep.setter
    def _timestep(self, timestep) -> None:
        self.__timestep = timestep

        # - Set the time step for each sublayer
        for lyr in self.evol_order:
            lyr._timestep = timestep

    @property
    def _evolve_functional(
        self,
    ) -> Callable[[Params, State, np.ndarray], Tuple[np.ndarray, State]]:
        """
        Return a functional form of the evolution function for this stack, with no side-effects

        :return Callable: evol_func
             evol_func(params: Params, all_states: State, ext_inputs: np.ndarray) -> Tuple[np.ndarray, State]:
        """

        def evol_func(
            params: Params, all_states: State, ext_inputs: np.ndarray,
        ) -> Tuple[np.ndarray, State]:
            # - Call the functional form of the evolution functions for each sublayer
            new_states = []
            inputs = ext_inputs
            out = np.array([])
            for p, s, evol_func in zip(params, all_states, self._all_evolve_funcs):
                # - Evolve layer
                out, new_state = evol_func(p, s, inputs)

                # - Record states
                new_states.append(new_state)

                # - Set up inputs for next layer
                inputs = out

            # - Return outputs and state
            return out, new_states

        return evol_func

    def train_output_target(
        self,
        ts_input: Union[TimeSeries, np.ndarray],
        ts_target: Union[TimeSeries, np.ndarray],
        is_first: bool = False,
        is_last: bool = False,
        loss_fcn: Callable[[Dict, np.ndarray, np.ndarray, Dict], float] = None,
        loss_params: Dict = {},
        optimizer: Callable = adam,
        opt_params: Dict = {"step_size": 1e-4},
    ) -> Tuple[Callable[[], float], Callable[[], float], Callable[[], np.ndarray]]:
        """
        Train this Jax stack, using a Jax optimiser

        See the documentation for :py:meth:`.TrainedJaxLayer.train_output_target` for details of how to train networks. This method provides a regularised MSE loss function that can cope with a Jax stack.

        :param Union[TimeSeries, np.ndarray] ts_input:  Input signal for this batch
        :param Union[TimeSeries, np.ndarray] ts_target: Target signal for this batch
        :param bool is_first:                           If ``True``, this is the first batch in the optimisation. Default: ``False``
        :param bool is_last:                            If ``True``, this is the last batch in the optimisation. Default: ``False``
        :param Callable loss_fcn:                       Loss function to optimise. Default: :py:func:`loss_mse_reg_stack`
        :param Dict loss_params:                        Dictionary of parameters to pass to :py:func:`loss_fcn`
        :param Callable optimizer:                      A Jax optimiser. See `jax.experimental.optimisers`. Default: :py:func:`jax.experimental.optimisers.adam`.
        :param Dict opt_params:                         Dictionary of parameters to pass to the optmiser. Default: {"step_size": 1e-4}

        :return Callable[[], float], Callable[[], float], Callable[[], np.ndarray]:
            loss_fcn:   A function that returns the loss for the current optimisation step
            grad_fcn:   A function that returns the gradients for the current optimisation step
            out_fcn:    A function that returns the output for the current optimisation step
        """
        # - Define a loss function that can deal with nested parameters
        @jit
        def loss_mse_reg_stack(
            params: List,
            output_batch_t: np.ndarray,
            target_batch_t: np.ndarray,
            min_tau: float,
            lambda_mse: float = 1.0,
            reg_tau: float = 10000.0,
            reg_l2_rec: float = 1.0,
        ) -> float:
            """
            Loss function for target versus output

            :param List params:                 List of packed parameters from each layer
            :param np.ndarray output_batch_t:   Output rasterised time series [TxO]
            :param np.ndarray target_batch_t:   Target rasterised time series [TxO]
            :param float min_tau:               Minimum time constant
            :param float lambda_mse:            Factor when combining loss, on mean-squared error term. Default: 1.0
            :param float reg_tau:               Factor when combining loss, on minimum time constant limit. Default: 1e5
            :param float reg_l2_rec:            Factor when combining loss, on L2-norm term of recurrent weights. Default: 1.

            :return float: Current loss value
            """
            # - Measure output-target loss
            mse = lambda_mse * np.mean((output_batch_t - target_batch_t) ** 2)

            # - Get loss for tau parameter constraints
            # - Measure recurrent L2 norms
            tau_loss = 0.0
            w_res_norm = 0.0
            for layer_params in params:
                tau_loss += reg_tau * np.mean(
                    np.where(
                        layer_params["tau"] < min_tau,
                        np.exp(-(layer_params["tau"] - min_tau)),
                        0,
                    )
                )
                w_res_norm += reg_l2_rec * np.mean(layer_params["w_recurrent"] ** 2)

            # - Loss: target/output squared error, time constant constraint, recurrent weights norm, activation penalty
            fLoss = mse + tau_loss + w_res_norm

            # - Return loss
            return fLoss

        # - Use provided loss function, if not overridden
        if loss_fcn is None:
            loss_fcn = loss_mse_reg_stack
            loss_params = {
                "min_tau": 11.0 * np.min([lyr._dt for lyr in self.evol_order])
            }

        # - Call super-class trainer
        return super().train_output_target(
            ts_input,
            ts_target,
            is_first,
            is_last,
            loss_fcn,
            loss_params,
            optimizer,
            opt_params,
        )

    @property
    def state(self) -> List:
        return [lyr.state for lyr in self.evol_order]

    @state.setter
    def state(self, new_states):
        for lyr, ns in zip(self.evol_order, new_states):
            lyr.state = ns

    @property
    def _state(self) -> List:
        # - Get states from all sublayers
        return [lyr._state for lyr in self.evol_order]

    @_state.setter
    def _state(self, new_states):
        for lyr, ns in zip(self.evol_order, new_states):
            lyr._state = ns

    def to_dict(self):
        raise NotImplementedError
