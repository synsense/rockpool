"""
Low level DynapSE1 simulator.
Solves the characteristic equations to simulate the circuit.
Trainable parameters

References:
[1] E. Chicca, F. Stefanini, C. Bartolozzi and G. Indiveri,
    "Neuromorphic Electronic Circuits for Building Autonomous Cognitive Systems,"
    in Proceedings of the IEEE, vol. 102, no. 9, pp. 1367-1388, Sept. 2014,
    doi: 10.1109/JPROC.2014.2313954.

[2] C. Bartolozzi and G. Indiveri, “Synaptic dynamics in analog vlsi,” Neural
    Comput., vol. 19, no. 10, p. 2581–2603, Oct. 2007. [Online]. Available:
    https://doi.org/10.1162/neco.2007.19.10.2581

[3] Dynap-SE1 Neuromorphic Chip Simulator for NICE Workshop 2021
    https://code.ini.uzh.ch/yigit/NICE-workshop-2021

[4] Course: Neurormophic Engineering 1
    Tobi Delbruck, Shih-Chii Liu, Giacomo Indiveri
    https://tube.switch.ch/channels/88df64b6

[5] Course: 21FS INI508 Neuromorphic Intelligence
    Giacomo Indiveri
    https://tube.switch.ch/switchcast/uzh.ch/series/5ee1d666-25d2-4c4d-aeb9-4b754b880345?order=newest-first


Acknowledgements:
    Dimitri Zendrikov for providing circuit diagrams and design feedback
    Michel Perez for spotting DPI simulation pitfalls
    Giacomo Indiveri

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
13/07/2021
"""

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter, State, SimulationParameter

import numpy as onp
import jax.random as rand
from jax.lax import scan

from jax import numpy as np
from typing import Optional, Protocol, Tuple, Union, Dict, Callable, Any

from rockpool.typehints import JP_ndarray, P_float


class DynapSE1NeuronSynapseJax(JaxModule):
    """
    Implements the chip dynamical equations for the DPI neuron and synapse models
    Receives configuration as bias currents
    As few HW restrictions as possible
    [] ATTENTION TODO: Now, implementation is only for one core
    [] TODO: TRY to present rng_key, dt, parameters, states as property, timeit!
    [] TODO: Arrange the silicon features in neat way. Think about dict, class, json

    DynapSE1 neuron and synapse circuit simulation
    This module implements the dynamics

    :Spike Frequency Adaptation:

    .. math ::
        \\tau_{ahp} \\dfrac{d}{dt} I_{ahp} + I_{ahp} = I_{ahp_{\\infty}} u(t)

    Where

    .. math ::
        I_{ahp_{\\infty}} = \\dfrac{I_{th_{ahp}}}{I_{\\tau_{ahp}}}I_{Ca}
        \\tau_{ahp} = \\dfrac{C_{ahp}U_{T}}{\\kappa I_{\\tau_{ahp}}}

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`I_{mem, j}` exceeds the threshold current :math:`I_{spkthr}`, then the neuron emits a spike.

    .. math ::

        I_{mem, j} > I_{spkthr} \\rightarrow S_{rec, j} = 1
        I_{ahp, j} += \\dfrac{1}{\\tau_{ahp}} \\cdot I_{ahp_{\\infty}}
        I_{mem, j} = I_{reset}

    [ ] TODO: FROM THE FIRST DESIGN MEETING
    [ ] TODO: What is the initial configuration of biases?
    [ ] TODO: How to convert from bias current parameters to high-level parameters and vice versa?
    [ ] TODO: Provides mismatch simulation (as second step)
    As a utility function that operates on a set of parameters?
    As a method on the class?
    """

    def __init__(
        self,
        shape: tuple = None,
        out_rate: float = 0.02,
        Itau_ahp: float = 1e-12,
        Ith_ahp: float = 1e-12,
        Io: float = 0.5e-12,
        Ica: float = 2e-12,
        kappa_n: float = 0.75,
        kappa_p: float = 0.66,
        Ut: float = 25e-3,
        Cahp: float = 1e-12,
        dt: float = 1e-3,
        rng_key: Optional[Any] = None,
        spiking_input: bool = True,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ):
        """
        __init__ Instantiate a DynapSE1NeuronSynapseJax Module

        :param shape: The number of neruons to employ, defaults to None
        :type shape: tuple, optional
        :param out_rate: the rate of output poisson spike generator(TEMPORARY PARAMETER), defaults to 0.02
        :type out_rate: float, optional
        :param Itau_ahp: Leakage current for spike-frequency adaptation, defaults to 1e-12
        :type Itau_ahp: float, optional
        :param Ith_ahp: Spike-frequency adaptation threshold current, defaults to 1e-12
        :type Ith_ahp: float, optional
        :param Io: Dark current, defaults to 0.5e-12
        :type Io: float, optional
        :param Ica: Calcium current, Spike-frequency adaptation weight current, defaults to 2e-12
        :type Ica: float, optional
        :param kappa_n: Subthreshold slope factor (n-type transistor), defaults to 0.75
        :type kappa_n: float, optional
        :param kappa_p: Subthreshold slope factor (p-type transistor), defaults to 0.66
        :type kappa_p: float, optional
        :param Ut: Thermal voltage, defaults to 25e-3
        :type Ut: float, optional
        :param Cahp: Spike-frequency adaptation capacitance, defaults to 1e-12
        :type Cahp: float, optional
        :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
        :type dt: float, optional
        :param rng_key: The Jax RNG seed to use on initialisation. By default, a new seed is generated, defaults to None
        :type rng_key: Optional[Any], optional
        :param spiking_input: Whether this module receives spiking input, defaults to True
        :type spiking_input: bool, optional
        :param spiking_output: Whether this module produces spiking output, defaults to True
        :type spiking_output: bool, optional
        :raises ValueError: When the user does not provide a valid shape
        """

        # [] TODO : Once attributes are stable, please add docstrings for each in `__init__()`

        if shape is None:
            raise ValueError("You must provide `shape`")

        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        # Parameters
        # [] TODO: all neurons cannot have different parameters ideally
        # however, they experience different parameters in practice
        # because of device mismatch

        self.Itau_ahp: JP_ndarray = Parameter(
            Itau_ahp * np.ones(self.size_out),
            "AHP",
            init_func=lambda s: np.ones(s) * Io,
            shape=(self.size_out,),
        )

        self.Ith_ahp: JP_ndarray = Parameter(
            Ith_ahp * np.ones(self.size_out),
            "AHP",
            init_func=lambda s: np.ones(s) * Io,
            shape=(self.size_out,),
        )

        # [] TODO : out_rate is just been using to validate the model
        self.out_rate = out_rate

        # States
        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))
        _, rng_key = rand.split(np.array(rng_key, dtype=np.uint32))

        self.rng_key: JP_ndarray = State(rng_key, init_func=lambda _: rng_key)

        self.spikes: JP_ndarray = State(shape=(self.size_out,), init_func=np.zeros)

        self.Iahp: JP_ndarray = State(
            shape=(self.size_out,), init_func=lambda s: np.ones(s) * Io
        )

        # Simulation Parameters
        self.dt: P_float = SimulationParameter(dt)
        ## Silicon Features
        self.Io: P_float = SimulationParameter(Io)
        self.Ica: P_float = SimulationParameter(Ica)
        self.kappa: P_float = SimulationParameter((kappa_n + kappa_p) / 2)
        self.Ut: P_float = SimulationParameter(Ut)
        self.Cahp: P_float = SimulationParameter(Cahp)

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        evolve Raw JAX evolution function for a DynapSE1NeuronSynapseJax module
        The function solves the dynamical equations introduced at
        ``DynapSE1NeuronSynapseJax`` module definition

        :param input_data: Input array of shape ``(T, Nin)`` to evolve over
        :type input_data: np.ndarray
        :param record: record the each timestep of evolution or not, defaults to False
        :type record: bool, optional
        :return: outputs, states, record_dict
            outputs: is an array with shape ``(T, Nout)`` containing the output data(spike raster) produced by this module.
            states: is a dictionary containing the updated module state following evolution.
            record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True``.
        :rtype: Tuple[np.ndarray, dict, dict]
        """

        def forward(
            state: State, spike_inputs_ts: np.ndarray
        ) -> Tuple[State, np.ndarray, np.ndarray]:
            """
            forward Single time-step neuron and synapse dynamics

            :param state: (spikes, Iahp, key)
                spikes: Logical spiking raster for each neuron [N]
                Iahp: Spike frequency adaptation current by each neuron [N]
                key: The Jax RNG seed to use for random spike generation
            :type state: State
            :param spike_inputs_ts: incoming spike raster to be used as an axis [T, M]
            :type spike_inputs_ts: np.ndarray
            :return: state, (spikes, Iahp)
                state: State at end of forward step
                spikes_ts: Logical spiking raster for each neuron over time [T, N]
                Iahp_ts: Spike frequency adaptation current by each neuron over time [T, N]
            :rtype: Tuple[State, np.ndarray, np.ndarray]
            """

            spikes, Iahp, key = state

            # Reset depending on spiking activity
            Iahp += spikes * self.Iahp_inf

            # Apply forward step
            Iahp = Iahp * np.exp(-self.dt / self.tau_ahp)
            # Iahp += self.del_Iahp()

            # Io clipping
            Iahp = np.clip(Iahp, self.Io)

            # [] TODO : Decide on upper bound
            # Iahp = np.clip(Iahp, self.Io, self.Iahp_inf)

            # [] TODO: REMOVE
            # Random spike generation
            key, subkey = rand.split(key)
            spikes = rand.poisson(subkey, self.out_rate, (self.size_out,)).astype(float)

            return (spikes, Iahp, key), (spikes, Iahp)

        # - Evolve over spiking inputs
        state, (spikes_ts, Iahp_ts) = scan(
            forward,
            (self.spikes, self.Iahp, self.rng_key),
            input_data,
        )

        # - Generate return arguments
        outputs = spikes_ts

        states = {
            "spikes": spikes_ts[-1],
            "Iahp": Iahp_ts[-1],
        }

        if record:
            record_dict = {
                "spikes": spikes_ts,
                "Iahp": Iahp_ts,
            }
        else:
            record_dict = {}

        # - Return outputs
        return outputs, states, record_dict

    @property
    def Iahp_inf(self):
        """
        Iahp_inf the reset value for Iahp
        ! Attention: Subject to change depended on I_tau_ahp

        :return: Ratio of currents through diffpair and adaptation block
        :rtype: np.ndarray
        """
        return (self.Ith_ahp / self.Itau_ahp) * self.Ica

    @property
    def tau_ahp(self):
        """
        tau_ahp adaptation time constant
        ! Attention: Subject to change depended on I_tau_ahp

        :return: depends on silicon properties and Itau_ahp together
        :rtype: np.ndarray
        """
        return (self.Cahp * self.Ut) / (self.kappa * self.Itau_ahp)
