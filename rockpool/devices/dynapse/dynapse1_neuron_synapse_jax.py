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
from dataclasses import dataclass

from typing import (
    Optional,
    Tuple,
    Callable,
    Any,
)

from rockpool.typehints import JP_ndarray, P_float


@dataclass
class DynapSE1Layout:
    def __init__(
        self,
        kappa_n: float = 0.75,
        kappa_p: float = 0.66,
        Ut: float = 25e-3,
        Cmem: float = 1.5e-12,
        Cahp: float = 1e-9,
        Cnmda: float = 1.5e-9,
        Campa: float = 1.5e-12,
        Cgaba_a: float = 1.5e-12,
        Cgaba_b: float = 1.5e-12,
        Io: float = 5e-13,
    ):

        self.kappa_n = kappa_n
        self.kappa_p = kappa_p
        self.Ut = Ut
        self.Cmem = Cmem
        self.Cahp = Cahp
        self.Cnmda = Cnmda
        self.Campa = Campa
        self.Cgaba_a = Cgaba_a
        self.Cgaba_b = Cgaba_b
        self.Io = Io
        self.kappa = (kappa_n + kappa_p) / 2


@dataclass
class DPIParameters:
    def __init__(
        self,
        Isyn: float = 5e-12,
        Itau: float = 1e-9,
        Ith: float = 1e-9,
        Iw: float = 1e-6,
    ):

        self.Isyn = Isyn
        self.Itau = Itau
        self.Ith = Ith
        self.Iw = Iw


@dataclass
class DynapSE1Parameters:
    def __init__(
        self,
        ahp: Optional[DPIParameters] = None,
        nmda: Optional[DPIParameters] = None,
    ):

        self.ahp = ahp
        self.nmda = nmda


class DynapSE1NeuronSynapseJax(JaxModule):
    """
    Implements the chip dynamical equations for the DPI neuron and synapse models
    Receives configuration as bias currents
    As few HW restrictions as possible
    [] ATTENTION TODO: Now, implementation is only for one core
    [] TODO: TRY to present rng_key, dt, parameters, states as property, timeit!
    [] TODO: Arrange the silicon features in neat way. Think about dict, class, json
    [] TODO: think about activating and deactivating certain circuit blocks. Might be function returning another function
    [] TODO: all neurons cannot have the same parameters ideally however, they experience different parameters in practice because of device mismatch

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
        dt: float = 1e-3,
        t_pulse: float = 1e-5,
        out_rate: float = 0.02,
        update_type: str = "dpi",
        params: Optional[DynapSE1Parameters] = None,
        layout: Optional[DynapSE1Layout] = None,
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
        :param Iw_ahp: Calcium current, Spike-frequency adaptation weight current, defaults to 2e-12
        :type Iw_ahp: float, optional
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
            raise ValueError("You must provide a `shape` tuple (N,)")

        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))

        if layout is None:
            layout = DynapSE1Layout()

        if params is None:
            params = DynapSE1Parameters(ahp=DPIParameters(), nmda=DPIParameters())

        _, rng_key = rand.split(np.array(rng_key, dtype=np.uint32))
        self._rng_key: JP_ndarray = State(rng_key, init_func=lambda _: rng_key)
        self._layout = layout
        self._params = params

        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        # [] TODO: Think about loop and __setattribute__

        # --- Parameters --- #
        self.Iahp, self.Itau_ahp, self.Ith_ahp, self.Iw_ahp = self._set_dpi_params(
            init=params.ahp,
            block_name="AHP",
        )

        self.Inmda, self.Itau_nmda, self.Ith_nmda, self.Iw_nmda = self._set_dpi_params(
            init=params.nmda,
            block_name="NMDA",
        )

        # [] TODO : out_rate is just been using to validate the model
        self.out_rate = out_rate
        self.update_type = update_type

        # --- States --- #
        self.spikes: JP_ndarray = State(shape=(self.size_out,), init_func=np.zeros)

        # --- Simulation Parameters --- #
        self.dt: P_float = SimulationParameter(dt)
        self.t_pulse: P_float = SimulationParameter(t_pulse)
        self.layout = SimulationParameter(layout)

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
        update_ahp = self._get_dpi_update_func(
            self.update_type, self.get_tau("ahp"), self.get_Iinf("ahp")
        )
        update_nmda = self._get_dpi_update_func(
            self.update_type, self.get_tau("nmda"), self.get_Iinf("nmda")
        )

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

            spikes, Iahp, Inmda, key = state

            # Apply forward step
            Iahp = update_ahp(Iahp, spikes)
            Inmda = update_nmda(Inmda, spike_inputs_ts)

            # [] TODO: REMOVE
            # Random spike generation
            key, subkey = rand.split(key)
            spikes = rand.poisson(subkey, self.out_rate, (self.size_out,)).astype(float)

            return (spikes, Iahp, Inmda, key), (spikes, Iahp, Inmda)

        # - Evolve over spiking inputs
        state, (spikes_ts, Iahp_ts, Inmda_ts) = scan(
            forward,
            (self.spikes, self.Iahp, self.Inmda, self._rng_key),
            input_data,
        )

        self.spikes, self.Iahp, self.Inmda, self._rng_key = state

        # - Generate return arguments
        outputs = spikes_ts

        states = {
            "spikes": self.spikes,
            "Iahp": self.Iahp,
            "Inmda": self.Inmda,
        }

        if record:
            record_dict = {
                "input_data": input_data,
                "spikes": spikes_ts,
                "Iahp": Iahp_ts,
                "Inmda": Inmda_ts,
            }
        else:
            record_dict = {}

        # - Return outputs
        return outputs, states, record_dict

    def _get_dpi_update_func(
        self, type: str, tau: Callable[[], float], Iss: Callable[[], float]
    ) -> Callable[[float], float]:
        """
        _get_dpi_update_func Returns the DPI update function given the type

        :param type: The update type : 'taylor', 'exp', or 'dpi'
        :type type: str
        :raises TypeError: If the update type given is neither 'taylor' nor 'exp' nor 'dpi'
        :return: a function calculating the value of the synaptic current in the next time step, given the instantaneous synaptic current
        :rtype: Callable[[float], float]
        """
        if type == "taylor":

            def _taylor_update(Isyn, spikes):
                Isyn += spikes * Iss()
                factor = -self.dt / tau()
                I_next = Isyn + Isyn * factor
                I_next = np.clip(I_next, self.layout.Io)
                return I_next

            return _taylor_update

        elif type == "exp":

            def _exponential_update(Isyn, spikes):
                Isyn += spikes * Iss()
                factor = np.exp(-self.dt / tau())
                I_next = Isyn * factor
                I_next = np.clip(I_next, self.layout.Io)
                return I_next

            return _exponential_update

        elif type == "dpi":

            def _dpi_update(Isyn, spikes):

                factor = np.exp(-self.dt / tau())

                # CHARGE PHASE
                charge = Iss() * (1.0 - factor) + Isyn * factor
                charge_vector = spikes * charge

                # DISCHARGE PHASE
                discharge = Isyn * factor
                discharge_vector = (1 - spikes) * discharge

                I_next = charge_vector + discharge_vector
                I_next = np.clip(I_next, self.layout.Io)

                return I_next

            return _dpi_update

        elif (
            type == "dpi_us"
        ):  # DPI Undersampled Simulation : only 1 spike allowed in 1ms

            def _dpi_us_update(Isyn, spikes):

                full_discharge = np.exp(-self.dt / tau())
                f_charge = np.exp(-self.t_pulse / tau())
                t_dis = (self.dt - self.t_pulse) / 2.0
                f_dis = np.exp(-t_dis / tau())

                # IF spikes
                # CHARGE PHASE -- UNDERSAMPLED -- dt >> t_pulse
                charge = (
                    Iss() * f_dis * (1.0 - f_charge) + Isyn * f_charge * f_dis * f_dis
                )
                charge_vector = spikes * charge

                # IF no spike at all
                # DISCHARGE PHASE
                discharge = Isyn * full_discharge
                discharge_vector = (1 - spikes) * discharge

                I_next = charge_vector + discharge_vector
                I_next = np.clip(I_next, self.layout.Io)

                return I_next

            return _dpi_us_update

        else:
            raise TypeError(
                f"{type} Update type undefined. Try one of 'taylor', 'exp', 'dpi'"
            )

    def _set_dpi_params(
        self,
        init: DPIParameters,
        block_name: str,
        Itau: Optional[np.ndarray] = None,
        Ith: Optional[np.ndarray] = None,
        Iw: Optional[np.ndarray] = None,
    ) -> Tuple[JP_ndarray, JP_ndarray, JP_ndarray, JP_ndarray]:
        if init is not None:
            Isyn: JP_ndarray = State(
                shape=(self.size_out,),
                family=block_name,
                init_func=lambda s: np.ones(s) * init.Isyn,
            )

            Itau: JP_ndarray = Parameter(
                data=Itau,
                family=block_name,
                init_func=lambda s: np.ones(s) * init.Itau,
                shape=(self.size_out,),
            )

            Ith: JP_ndarray = Parameter(
                data=Ith,
                family=block_name,
                init_func=lambda s: np.ones(s) * init.Ith,
                shape=(self.size_out,),
            )

            Iw: JP_ndarray = Parameter(
                data=Iw,
                family=block_name,
                init_func=lambda s: np.ones(s) * init.Iw,
                shape=(self.size_out,),
            )

            return Isyn, Itau, Ith, Iw

        else:
            raise ValueError("Initial DPI parameter values are not given!")

    def get_tau(self, name):
        capacitance = self.layout.__getattribute__(f"C{name}")
        Ut = self.layout.Ut
        kappa = self.layout.kappa

        def tau():
            Itau = self.__getattribute__(f"Itau_{name}")
            time_const = (capacitance * Ut) / (kappa * Itau)
            return time_const

        return tau

    def get_Iinf(self, name):
        def Iinf():
            Ith = self.__getattribute__(f"Ith_{name}")
            Itau = self.__getattribute__(f"Itau_{name}")
            Iw = self.__getattribute__(f"Iw_{name}")
            Iss = (Ith / Itau) * Iw
            return Iss

        return Iinf
