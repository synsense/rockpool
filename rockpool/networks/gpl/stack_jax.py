#
# stack_jax.py — Implement trainable stacks of jax layers
#

from ...timeseries import TSContinuous, TSEvent
from ..network import Network
from ...layers.training import JaxTrainer
from ...layers.layer import Layer

from typing import Tuple, List, Callable, Dict, Sequence, Optional, Any

from importlib import util

if (util.find_spec("jax") is None) or (util.find_spec("jaxlib") is None):
    raise ModuleNotFoundError(
        "'jax' and 'jaxlib' backends not found. Layers that rely on Jax will not be available."
    )

import jax.numpy as np
import json

Params = List
State = List

__all__ = ["JaxStack"]


def loss_mse_reg_stack(
    params: List,
    states_t: Dict[str, np.ndarray],
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


class JaxStack(Network, Layer, JaxTrainer):
    """
    Build a network of Jax layers, supporting parameter optimisation



    """

    def __init__(self, layers: Sequence = None, dt=None, *args, **kwargs):
        """
        Encapsulate a stack of Jax layers in a single layer / network

        :param Sequence layers: A Sequence of layers to initialise the stack with
        :param float dt:        Unitary timestep to force on each of the sublayers
        """
        # - Check that the layers are subclasses of `JaxTrainer`
        if layers is not None:
            for layer in layers:
                if not isinstance(layer, JaxTrainer):
                    raise TypeError(
                        "JaxStack: Each layer must inherit from the `JaxTrainer` mixin class"
                    )

        # - Initialise super classes
        super().__init__(layers=layers, dt=dt, weights=[], *args, **kwargs)

        # - Make sure every layer has same `dt`
        for lyr in self.evol_order:
            if lyr.dt != self.dt:
                raise ValueError("JacStack: All layers must have same `dt`.")

        # - Initialise timestep
        self.__timestep: int = 0

        self._size_in: Optional[int] = None
        self._size_out: Optional[int] = None
        self._size: Optional[int] = None

        # - Get evolution functions
        self._all_evolve_funcs: List = [
            lyr._evolve_functional for lyr in self.evol_order
        ]

        # - Set sizes
        if layers is not None:
            self._size_in = self.input_layer.size_in
            self._size_out = self.evol_order[-1].size_out

    def evolve(
        self,
        ts_input: TSContinuous = None,
        duration: float = None,
        num_timesteps: int = None,
        verbose: bool = False,
    ) -> Dict:
        """

        :param TSContinuous ts_input:
        :param float duration:
        :param int num_timesteps:
        :param bool verbose:
        :return:
        """

        # - Catch an empty stack
        if self.evol_order is None or len(self.evol_order) == 0:
            return {}

        # - Prepare time base and inputs, using first layer
        time_base_inp, ext_inps, num_timesteps = self.input_layer._prepare_input(
            ts_input, duration, num_timesteps
        )

        # - Evolve over layers
        outputs = []
        new_states = []
        inps = ext_inps
        for p, s, evol_func in zip(self._pack(), self.state, self._all_evolve_funcs):
            # - Evolve layer
            out, new_state, _ = evol_func(p, s, inps)

            # - Record outputs and states
            outputs.append(out)
            new_states.append(new_state)

            # - Set up inputs for next layer
            inps = out

        # - Wrap outputs as time series
        outputs_dict = {"external_input": ts_input}
        for lyr, out in zip(self.evol_order, outputs):
            if lyr.output_type == TSContinuous:
                ts_out = TSContinuous.from_clocked(
                    np.array(out), t_start=self.t, dt=self.dt, name=f"Output {lyr.name}"
                )
            else:
                ts_out = TSEvent.from_raster(
                    out,
                    t_start=self.t,
                    dt=self.dt,
                    periodic=False,
                    num_channels=lyr.size,
                    spikes_at_bin_start=False,
                    name=f"Spikes {lyr.name}",
                )
            outputs_dict.update({lyr.name: ts_out})

        # - Assign updated states
        self._states = new_states

        # - Update time stamps
        self._timestep += num_timesteps

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

    # - Replace the default loss function
    @property
    def _default_loss(self) -> Callable[[Any], float]:
        return loss_mse_reg_stack

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
    ) -> Callable[
        [Params, State, np.ndarray], Tuple[List[np.ndarray], State, List[np.ndarray]]
    ]:
        """
        Return a functional form of the evolution function for this stack, with no side-effects

        :return Callable: evol_func
             evol_func(params: Params, all_states: State, ext_inputs: np.ndarray) -> Tuple[List[np.ndarray], State, List[np.ndarray]]:
        """

        def evol_func(
            params: Params,
            all_states: State,
            ext_inputs: np.ndarray,
        ) -> Tuple[List[np.ndarray], State, List[np.ndarray]]:
            # - Call the functional form of the evolution functions for each sublayer
            new_states = []
            layer_states_t = []
            inputs = ext_inputs
            out = np.array([])
            for i_lyr, (p, s, evol_func) in enumerate(
                zip(params, all_states, self._all_evolve_funcs)
            ):
                # - Evolve layer
                out, new_state, states_t = evol_func(p, s, inputs)

                # - Record states
                new_states.append(new_state)
                layer_states_t.append(states_t)

                # - Set up inputs for next layer
                inputs = out

            # - Return outputs and state
            return out, new_states, layer_states_t

        return evol_func

    @property
    def state(self) -> List[State]:
        return [lyr.state for lyr in self.evol_order]

    @state.setter
    def state(self, new_states):
        for lyr, ns in zip(self.evol_order, new_states):
            lyr.state = ns

    @property
    def _state(self) -> List[State]:
        # - Get states from all sublayers
        return [lyr._state for lyr in self.evol_order]

    @_state.setter
    def _state(self, new_states):
        for lyr, ns in zip(self.evol_order, new_states):
            lyr._state = ns

    def to_dict(self):
        return Network.to_dict(self)

    @staticmethod
    def load(filename: str) -> "JaxStack":
        """
        Load a network from a JSON file

        :param str filename:    filename of a JSON file that contains a saved network
        :return JaxStack:        A JaxStack object with all the layers loaded from `filename`
        """
        # - Load dict holding the parameters
        with open(filename, "r") as f:
            loaddict: dict = json.load(f)
        net = Network.load_from_dict(loaddict)
        return JaxStack([l for l in net.evol_order])

    @property
    def input_type(self):
        return self.evol_order[0].input_type
