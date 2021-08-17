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

from utils import dpi_update_func


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
        Campa: float = 1.5e-10,
        Cgaba_a: float = 1.5e-11,
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

        # Calculated
        self.kappa = (kappa_n + kappa_p) / 2

        _f_tau = Ut / self.kappa

        self.f_tau_mem = _f_tau * Cmem

        self.f_tau_ahp = _f_tau * Cahp
        self.f_tau_nmda = _f_tau * Cnmda
        self.f_tau_ampa = _f_tau * Campa
        self.f_tau_gaba_a = _f_tau * Cgaba_a
        self.f_tau_gaba_b = _f_tau * Cgaba_b
        self.f_tau_syn = np.array(
            [
                self.f_tau_ahp,
                self.f_tau_nmda,
                self.f_tau_ampa,
                self.f_tau_gaba_a,
                self.f_tau_gaba_b,
            ]
        )


@dataclass
class DPIParameters:
    def __init__(
        self,
        Isyn: float = 5e-13,
        Itau: float = 1e-9,
        Ith: float = 1e-9,
        Iw: float = 1e-6,
    ):

        self.Isyn = Isyn
        self.Itau = Itau
        self.Ith = Ith
        self.Iw = Iw


@dataclass
class FeedbackParameters:
    def __init__(
        self,
        Igain: float = 5e-11,
        Ith: float = 5e-10,
        Inorm: float = 1e-11,
    ):
        self.Igain = Igain
        self.Ith = Ith
        self.Inorm = Inorm


@dataclass
class MembraneParameters:
    def __init__(
        self,
        Imem: float = 5e-13,
        Itau: float = 1e-9,
        Ith: float = 1e-9,
        feedback: Optional[FeedbackParameters] = None,
    ):
        self.Imem = Imem
        self.Itau = Itau
        self.Ith = Ith
        if feedback is None:
            feedback = FeedbackParameters()
        self.feedback = feedback


@dataclass
class DynapSE1Parameters:
    def __init__(
        self,
        mem: Optional[MembraneParameters] = None,
        ahp: Optional[DPIParameters] = None,
        nmda: Optional[DPIParameters] = None,
        ampa: Optional[DPIParameters] = None,
        gaba_a: Optional[DPIParameters] = None,
        gaba_b: Optional[DPIParameters] = None,
    ):

        self.mem = mem
        self.ahp = ahp
        self.nmda = nmda
        self.ampa = ampa
        self.gaba_a = gaba_a
        self.gaba_b = gaba_b


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
        t_pulse_rec: float = 1e-6,
        Idc: float = 5e-13,
        out_rate: float = 0.02,
        update_type: str = "dpi_us3",
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
            params = DynapSE1Parameters(
                mem=MembraneParameters(),
                ahp=DPIParameters(),
                nmda=DPIParameters(),
                ampa=DPIParameters(),
                gaba_a=DPIParameters(),
                gaba_b=DPIParameters(),
            )

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
        self.Imem, self.Itau_mem, self.Ith_mem, self._feedback = self._set_mem_params(
            init=params.mem,
            block_name="MEM",
        )

        self.Iahp, self.Itau_ahp, self.Ith_ahp, self.Iw_ahp = self._set_dpi_params(
            init=params.ahp,
            block_name="AHP",
        )

        self.Inmda, self.Itau_nmda, self.Ith_nmda, self.Iw_nmda = self._set_dpi_params(
            init=params.nmda,
            block_name="NMDA",
        )

        self.Iampa, self.Itau_ampa, self.Ith_ampa, self.Iw_ampa = self._set_dpi_params(
            init=params.ampa,
            block_name="AMPA",
        )

        (
            self.Igaba_a,
            self.Itau_gaba_a,
            self.Ith_gaba_a,
            self.Iw_gaba_a,
        ) = self._set_dpi_params(init=params.gaba_a, block_name="GABA_A")

        (
            self.Igaba_b,
            self.Itau_gaba_b,
            self.Ith_gaba_b,
            self.Iw_gaba_b,
        ) = self._set_dpi_params(init=params.gaba_b, block_name="GABA_B")

        # [] TODO : out_rate is just been using to validate the model
        self.out_rate = out_rate
        self.update_type = update_type

        # --- States --- #
        self.spikes: JP_ndarray = State(shape=(self.size_out,), init_func=np.zeros)

        # --- Simulation Parameters --- #
        self.dt: P_float = SimulationParameter(dt)
        self.t_pulse: P_float = SimulationParameter(t_pulse)
        self.t_pulse_rec: P_float = SimulationParameter(t_pulse_rec)
        self.Idc: P_float = SimulationParameter(Idc)
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

        :param input_data: Input array of shape ``(T, Nin)`` to evolve over. Represents number of spikes at that timebin
        :type input_data: np.ndarray
        :param record: record the each timestep of evolution or not, defaults to False
        :type record: bool, optional
        :return: outputs, states, record_dict
            outputs: is an array with shape ``(T, Nout)`` containing the output data(spike raster) produced by this module.
            states: is a dictionary containing the updated module state following evolution.
            record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True``.
        :rtype: Tuple[np.ndarray, dict, dict]
        """
        f_tau_syn = self.layout.f_tau_syn
        Io = self.layout.Io

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
            spikes, Imem, Iahp, Inmda, Iampa, Igaba_a, Igaba_b, key = state

            # --- Utilities --- #  # 5xNin
            # Combine synaptic currents and related parameters together for the sake of parallel computation
            _Isyn = np.array([Iahp, Inmda, Iampa, Igaba_a, Igaba_b])

            _Itau_syn = np.array(
                [
                    self.Itau_ahp,
                    self.Itau_nmda,
                    self.Itau_ampa,
                    self.Itau_gaba_a,
                    self.Itau_gaba_b,
                ]
            )

            _Ith_syn = np.array(
                [
                    self.Ith_ahp,
                    self.Ith_nmda,
                    self.Ith_ampa,
                    self.Ith_gaba_a,
                    self.Ith_gaba_b,
                ]
            )

            _Iw_syn = np.array(
                [
                    self.Iw_ahp,
                    self.Iw_nmda,
                    self.Iw_ampa,
                    self.Iw_gaba_a,
                    self.Iw_gaba_b,
                ]
            )

            _tau_syn = (f_tau_syn / _Itau_syn.T).T  # 5xNin
            _Iss_syn = (_Ith_syn / _Itau_syn) * _Iw_syn  # 5xNin

            # --- Forward step --- #

            ## combine spike output and spiking inputs for 4 synapses
            spike_inputs = np.tile(spike_inputs_ts, (4, 1))

            ## Calculate the pulse width with a linear increase
            t_pw_in = self.t_pulse * spike_inputs  # 4xNin NMDA, AMPA, GABA_A, GABA_B
            t_pw_out = self.t_pulse_rec * spikes  # 1xNin AHP

            # 5xNin [AHP, NMDA, AMPA, GABA_A, GABA_B]
            t_pw = np.vstack((t_pw_out, t_pw_in))

            ## Exponential charge and discharge factor vectors
            f_charge = 1 - np.exp(-t_pw / _tau_syn)  # 5xNin
            f_discharge = np.exp(-self.dt / _tau_syn)  # 5xNin

            # DISCHARGE IN ANY CASE
            _Isyn = f_discharge * _Isyn

            # CHARGE PHASE -- UNDERSAMPLED -- dt >> t_pulse
            # f_charge array = 0 where there is no spike
            _Isyn += f_charge * _Iss_syn

            # Make sure that synaptic currents are lower bounded by Io
            _Isyn = np.clip(_Isyn, Io)  # 5xNin

            # --- MEMRANE DYNAMICS --- #
            # [] TODO : Change I_gaba_b dynamics. It's the shunt current

            tau_mem = self.layout.f_tau_mem / self.Itau_mem
            Iin = Inmda + Iampa - Igaba_a - Igaba_b + self.Idc
            Ia = self._feedback.Igain / (
                1 + np.exp(-(Imem + self._feedback.Ith) / (self._feedback.Inorm))
            )
            f_Imem = ((Ia) / (self.Itau_mem)) * (Imem + self.Ith_mem)
            Imem_inf = ((self.Ith_mem) / (self.Itau_mem)) * (Iin - Iahp - self.Itau_mem)

            ## MEMRANE Forward Euler ##
            del_Imem = (
                (Imem / (tau_mem * (self.Ith_mem + Imem)))
                * (Imem_inf + f_Imem - Imem * (1 + (Iahp / self.Itau_mem)))
                * self.dt
            )
            Imem = Imem + del_Imem

            Iahp, Inmda, Iampa, Igaba_a, Igaba_b = _Isyn  # 5xNin

            # [] TODO: REMOVE
            # Random spike generation
            key, subkey = rand.split(key)
            spikes = rand.poisson(subkey, self.out_rate, (self.size_out,)).astype(float)

            state = (spikes, Imem, Iahp, Inmda, Iampa, Igaba_a, Igaba_b, key)
            return state, (spikes, Imem, Iahp, Inmda, Iampa, Igaba_a, Igaba_b)

        # - Evolve over spiking inputs
        state, (
            spikes_ts,
            Imem_ts,
            Iahp_ts,
            Inmda_ts,
            Iampa_ts,
            Igaba_a_ts,
            Igaba_b_ts,
        ) = scan(
            forward,
            (
                self.spikes,
                self.Imem,
                self.Iahp,
                self.Inmda,
                self.Iampa,
                self.Igaba_a,
                self.Igaba_b,
                self._rng_key,
            ),
            input_data,
        )

        (
            self.spikes,
            self.Imem,
            self.Iahp,
            self.Inmda,
            self.Iampa,
            self.Igaba_a,
            self.Igaba_b,
            self._rng_key,
        ) = state

        # - Generate return arguments
        outputs = spikes_ts

        states = {
            "spikes": self.spikes,
            "Imem": self.Imem,
            "Iahp": self.Iahp,
            "Inmda": self.Inmda,
            "Iampa": self.Iampa,
            "Igaba_a": self.Igaba_a,
            "Igaba_b": self.Igaba_b,
        }

        if record:
            record_dict = {
                "input_data": input_data,
                "spikes": spikes_ts,
                "Imem": Imem_ts,
                "Iahp": Iahp_ts,
                "Inmda": Inmda_ts,
                "Iampa": Iampa_ts,
                "Igaba_a": Igaba_a_ts,
                "Igaba_b": Igaba_b_ts,
            }
        else:
            record_dict = {}

        # - Return outputs
        return outputs, states, record_dict

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

    def _set_mem_params(
        self,
        init: MembraneParameters,
        block_name: str,
        Itau: Optional[np.ndarray] = None,
        Ith: Optional[np.ndarray] = None,
    ) -> Tuple[JP_ndarray, JP_ndarray, JP_ndarray, FeedbackParameters]:
        if init is not None:
            Imem: JP_ndarray = State(
                shape=(self.size_out,),
                family=block_name,
                init_func=lambda s: np.ones(s) * init.Imem,
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

            feedback: FeedbackParameters = init.feedback

            return Imem, Itau, Ith, feedback

        else:
            raise ValueError("Initial membrane parameter values are not given!")

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
