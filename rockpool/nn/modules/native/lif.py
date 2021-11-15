"""
Implements a leaky integrate-and-fire neuron module with a numpy backend
"""

from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter, State, SimulationParameter

import numpy as np

from .linear import kaiming

from typing import Optional, Tuple, Union, Dict, Callable, Any
from rockpool.typehints import FloatVector, P_float, P_tensor

__all__ = ["LIF"]

# - Surrogate function
def sigmoid(x: FloatVector) -> FloatVector:
    """
    Sigmoid function

    :param FloatVector x: Input value

    :return FloatVector: Output value
    """
    return np.tanh(x + 1) / 2 + 0.5


class LIF(Module):
    """
    A leaky integrate-and-fire spiking neuron model

    This module implements the dynamics:

    .. math ::

        \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

        I_{syn} += S_{in}(t)

        \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + b + \\sigma\\zeta(t)

    where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

    .. math ::

        V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

        I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

        V_{mem, j} = V_{mem, j} - 1

    Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``0.``).
    """

    def __init__(
        self,
        shape: tuple = None,
        tau_mem: Optional[FloatVector] = None,
        tau_syn: Optional[FloatVector] = None,
        bias: Optional[FloatVector] = None,
        threshold: Optional[FloatVector] = None,
        w_rec: Optional[FloatVector] = None,
        weight_init_func: Callable = kaiming,
        dt: float = 1e-3,
        noise_std: float = 0.0,
        spiking_input: bool = True,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ):
        """
        Instantiate an LIF module

        Args:
            shape (tuple): Either a single dimension ``N``, which defines a feed-forward layer of LIF neurons, or two dimensions ``(N, N)``, which defines a recurrent layer of LIF neurons.
            tau_mem (Optional[np.ndarray]): An optional array with concrete initialisation data for the membrane time constants. If not provided, 100ms will be used by default.
            tau_syn (Optional[np.ndarray]): An optional array with concrete initialisation data for the synaptic time constants. If not provided, 50ms will be used by default.
            bias (Optional[np.ndarray]): An optional array with concrete initialisation data for the neuron bias currents. If not provided, 0.0 will be used by default.
            w_rec (Optional[np.ndarray]): If the module is initialised in recurrent mode, you can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(N, N)``. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
            dt (float): The time step for the forward-Euler ODE solver. Default: 1ms
            noise_std (float): The std. dev. of the noise added to membrane state variables at each time-step. Default: 0.0
            rng_key (Optional[Any]): The Jax RNG seed to use on initialisation. By default, a new seed is generated.
        """
        # - Work out the shape of this module
        if shape is None:
            assert (
                tau_mem is not None
            ), "You must provide either `shape` or else specify parameters."
            shape = np.array(tau_mem).shape

        # - Call the superclass initialiser
        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        # - Should we be recurrent or FFwd?
        if len(self.shape) == 1:
            # - Feed-forward mode
            if w_rec is not None:
                raise ValueError(
                    "If `shape` is unidimensional, then `w_rec` may not be provided as an argument."
                )

        else:
            # - Recurrent mode
            # - Check that `shape` is correctly specified
            if len(self.shape) > 2:
                raise ValueError("`shape` may not specify more than two dimensions.")

            if self.size_out != self.size_in:
                raise ValueError(
                    "`shape[0]` and `shape[1]` must be equal for a recurrent module."
                )

            self.w_rec: Union[np.ndarray, Parameter] = Parameter(
                w_rec,
                family="weights",
                init_func=weight_init_func,
                shape=self.shape,
            )
            """ (np.ndarray) Recurrent weights ``(N, N)`` """

        # - Set parameters
        self.tau_mem: Union[np.ndarray, Parameter] = Parameter(
            tau_mem,
            "taus",
            init_func=lambda s: np.ones(s) * 100e-3,
            shape=(self.size_out,),
        )
        """ (np.ndarray) Membrane time constants for each neuron ``(N,)`` in seconds """

        self.tau_syn: P_tensor = Parameter(
            tau_syn,
            "taus",
            init_func=lambda s: np.ones(s) * 50e-3,
            shape=(self.size_out,),
        )
        """ (np.ndarray) Synaptic time constants for each neuron ``(N,)`` in seconds """

        self.bias: P_tensor = Parameter(
            bias,
            "bias",
            init_func=lambda s: np.zeros(s),
            shape=(self.size_out,),
        )
        """ (np.ndarray) Bias current for each neuron ``(N,)`` """

        self.threshold: P_tensor = Parameter(
            threshold,
            "threshold",
            init_func=lambda s: np.zeros(s),
            shape=(self.size_out,),
        )
        """ (np.ndarray) Threshold for each neuron ``(N,)`` """

        self.dt: P_float = SimulationParameter(dt)
        """ (float) Simulation time-step in seconds """

        self.noise_std: P_float = SimulationParameter(noise_std)
        """ (float) White noise std. dev. added at each time-step """

        # - Specify state
        self.spikes: P_tensor = State(shape=(self.size_out,), init_func=np.zeros)
        """ (np.ndarray) Spiking status of each neuron ``(N,)`` """

        self.Isyn: P_tensor = State(shape=(self.size_out,), init_func=np.zeros)
        """ (np.ndarray) Synaptic current of each neuron ``(N,)`` """

        self.Vmem: P_tensor = State(shape=(self.size_out,), init_func=np.zeros)
        """ (np.ndarray) Membrane potential of each neuron ``(N,)`` """

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        Evolution function for an LIF spiking layer

        This function implements the dynamics:

        .. math ::

            \\tau_{syn} \\dot{I}_{syn} + I_{syn} = 0

            I_{syn} += S_{in}(t)

            \\tau_{syn} \\dot{V}_{mem} + V_{mem} = I_{syn} + b + \\sigma\\zeta(t)

        where :math:`S_{in}(t)` is a vector containing ``1`` for each input channel that emits a spike at time :math:`t`; :math:`b` is a :math:`N` vector of bias currents for each neuron; :math:`\\sigma\\zeta(t)` is a white-noise process with standard deviation :math:`\\sigma` injected independently onto each neuron's membrane; and :math:`\\tau_{mem}` and :math:`\\tau_{syn}` are the membrane and synaptic time constants, respectively.

        :On spiking:

        When the membrane potential for neuron :math:`j`, :math:`V_{mem, j}` exceeds the threshold voltage :math:`V_{thr} = 0`, then the neuron emits a spike.

        .. math ::

            V_{mem, j} > V_{thr} \\rightarrow S_{rec,j} = 1

            I_{syn} = I_{syn} + S_{rec} \\cdot w_{rec}

            V_{mem, j} = V_{mem, j} - 1

        Neurons therefore share a common resting potential of ``0``, a firing threshold of ``0``, and a subtractive reset of ``-1``. Neurons each have an optional bias current `.bias` (default: ``-1``).

        Args:
            input_data (np.ndarray): Input array of shape ``(T, Nin)`` to evolve over
            record (bool): If ``True``,

        Returns:
            (np.ndarray, dict, dict): output, new_state, record_state
            ``output`` is an array with shape ``(T, Nout)`` containing the output data produced by this module. ``new_state`` is a dictionary containing the updated module state following evolution. ``record_state`` will be a dictionary containing the recorded state variables for this evolution, if the ``record`` argument is ``True``.
        """

        # - Get evolution constants
        alpha = self.dt / self.tau_mem
        beta = np.exp(-self.dt / self.tau_syn)

        # - Single-step LIF dynamics
        def forward(
            state: Tuple[Any, Any, Any], inputs_t: Tuple[np.ndarray, np.ndarray]
        ) -> (
            State,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ):
            """
            Single-step LIF dynamics for a recurrent LIF layer

            :param LayerState state:
            :param Tuple[np.ndarray, np.ndarray] inputs_t: (spike_inputs_ts, current_inputs_ts)

            :return: (state), (Irec_ts, spikes_ts, Vmem_ts, Isyn_ts)
                state:          (LayerState) Layer state at end of evolution
                Irec_ts:        (np.ndarray) Recurrent input received at each neuron over time [T, N]
                spikes_ts:      (np.ndarray) Logical spiking raster for each neuron [T, N]
                Vmem_ts:        (np.ndarray) Membrane voltage of each neuron over time [T, N]
                Isyn_ts:        (np.ndarray) Synaptic input current received by each neuron over time [T, N]
            """
            # - Unpack inputs
            (sp_in_t, I_in_t) = inputs_t
            sp_in_t = sp_in_t.reshape(-1)
            Iin = I_in_t.reshape(-1)

            spikes, Isyn, Vmem = state

            # - Synaptic input
            Irec = np.dot(spikes, self.w_rec)
            dIsyn = sp_in_t + Irec
            Isyn = beta * Isyn + dIsyn

            # - Apply subtractive reset
            Vmem = Vmem - spikes

            # - Membrane potentials
            dVmem = Isyn + Iin + self.bias - Vmem
            Vmem = Vmem + alpha * dVmem

            # - Detect next spikes
            spikes = Vmem > self.threshold

            # - Return state and outputs
            return (spikes, Isyn, Vmem), (Irec, spikes, Vmem, Isyn)

        # - Generate membrane noise trace
        num_timesteps = input_data.shape[0]
        noise_ts = self.noise_std * np.random.normal((num_timesteps, self.size_out))

        # - Evolve over spiking inputs
        Irec_ts = []
        spikes_ts = []
        Vmem_ts = []
        Isyn_ts = []
        for t in range(input_data.shape[0]):
            _, (this_irec, self.spikes, self.Vmem, self.Isyn) = forward(
                (self.spikes, self.Isyn, self.Vmem),
                (input_data[t, :], noise_ts[t, :]),
            )
            Irec_ts.append(this_irec)
            spikes_ts.append(self.spikes)
            Vmem_ts.append(self.Vmem)
            Isyn_ts.append(self.Isyn)

        # state, (Irec_ts, spikes_ts, Vmem_ts, Isyn_ts) = scan(
        #     forward,
        #     (self.spikes, self.Isyn, self.Vmem),
        #     (input_data, noise_ts),
        # )

        # - Generate output surrogate
        surrogate_ts = sigmoid(np.array(Vmem_ts) * 20.0)

        # - Generate return arguments
        outputs = np.array(spikes_ts)
        record_dict = (
            {
                "Irec": np.array(Irec_ts),
                "spikes": np.array(spikes_ts),
                "Isyn": np.array(Isyn_ts),
                "Vmem": np.array(Vmem_ts),
                "U": np.array(surrogate_ts),
            }
            if record
            else {}
        )

        # - Return outputs
        return outputs, np.state(), record_dict
