"""
Low level DynapSE1 simulator.
Solves the characteristic equations to simulate the circuits.
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
    Any,
)

from rockpool.typehints import JP_ndarray, P_float


@dataclass
class DynapSE1Layout:
    """
    DynapSE1Layout contains the constant values used in simulation that are related to the exact silicon layout of a chip.

    :param kappa_n: Subthreshold slope factor (n-type transistor), defaults to 0.75
    :type kappa_n: float, optional
    :param kappa_p: Subthreshold slope factor (n-type transistor), defaults to 0.66
    :type kappa_p: float, optional
    :param Ut: Thermal voltage in Volts, defaults to 25e-3
    :type Ut: float, optional
    :param Cmem: Membrane capacitance in Farads, fixed at layout time, defaults to 3.2e-12
    :type Cmem: float, optional
    :param Cahp: Spike-frequency adaptation capacitance in Farads, defaults to 40e-12
    :type Cahp: float, optional
    :param Cnmda: NMDA DPI synaptic capacitance in Farads, fixed at layout time, defaults to 28e-12
    :type Cnmda: float, optional
    :param Campa: AMPA DPI synaptic capacitance in Farads, fixed at layout time, defaults to 28e-12
    :type Campa: float, optional
    :param Cgaba_a: GABA_A DPI synaptic capacitance in Farads, fixed at layout time,, defaults to 27e-12
    :type Cgaba_a: float, optional
    :param Cgaba_b: GABA_B(SHUNT) DPI synaptic capacitance in Farads, fixed at layout time, defaults to 27e-12
    :type Cgaba_b: float, optional
    :param Io: Dark current in Amperes that flows through the transistors even at the idle state, defaults to 5e-13
    :type Io: float, optional

    :Instance Variables:

    :ivar kappa: Mean kappa
    :type kappa: float
    :ivar f_tau_ahp: Tau factor for AHP. tau = f_tau/I_tau
    :type f_tau_ahp: float
    :ivar f_tau_nmda: Tau factor for NMDA. tau = f_tau/I_tau
    :type f_tau_nmda: float
    :ivar f_tau_ampa: Tau factor for AMPA. tau = f_tau/I_tau
    :type f_tau_ampa: float
    :ivar f_tau_gaba_a: Tau factor for GABA_A. tau = f_tau/I_tau
    :type f_tau_gaba_a: float
    :ivar f_tau_gaba_b: Tau factor for GABA_B. tau = f_tau/I_tau
    :type f_tau_gaba_b: float
    :ivar f_tau_syn: A vector of tau factors in the following order: [AHP, NMDA, AMPA, GABA_A, GABA_B]
    :type f_tau_syn: np.ndarray
    """

    kappa_n: float = 0.75
    kappa_p: float = 0.66
    Ut: float = 25e-3
    Cmem: float = 3.2e-12
    Cahp: float = 40e-12
    Cnmda: float = 28e-12
    Campa: float = 28e-12
    Cgaba_a: float = 27e-12
    Cgaba_b: float = 27e-12
    Io: float = 5e-13

    def __post_init__(self):
        """
        __post_init__ runs after __init__ and sets the variables depending on the initial values
        """
        self.kappa = (self.kappa_n + self.kappa_p) / 2.0
        self._set_tau_factors()

    def _set_tau_factors(self):
        """
        _set_tau_factors sets individual tau factors as well as a vector of factors as instance variables.
        Tau can then be obtained by tau = f_tau/I_tau
        """
        # Block independend factor
        _f_tau = self.Ut / self.kappa

        # Membrane
        self.f_tau_mem = _f_tau * self.Cmem

        # DPI Circuits
        self.f_tau_ahp = _f_tau * self.Cahp
        self.f_tau_nmda = _f_tau * self.Cnmda
        self.f_tau_ampa = _f_tau * self.Campa
        self.f_tau_gaba_a = _f_tau * self.Cgaba_a
        self.f_tau_gaba_b = _f_tau * self.Cgaba_b

        # All DPI synapses together
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
    """
    DPIParameters contains DPI specific parameter and state variables

    :param Isyn: DPI output current in Amperes, defaults to 5e-13
    :type Isyn: float, optional
    :param Itau: Synaptic time constant current in Amperes, that is inversely proportional to time constant tau, defaults to 10e-12
    :type Itau: float, optional
    :param Ith: DPI's threshold / gain current in Amperes, scaling factor for the synaptic weight (typically x2, x4 of I_tau), defaults to 40e-12
    :type Ith: float, optional
    :param Iw: Synaptic weight current in Amperes, determines how strong the response is in terms of amplitude, defaults to 1e-6
    :type Iw: float, optional
    """

    Isyn: float = 5e-13
    Itau: float = 10e-12
    Ith: float = 40e-12
    Iw: float = 1e-6


@dataclass
class FeedbackParameters:
    """
    FeedbackParameters contains the positive feedback circuit block parameters of Dynap-SE1 membrane.
    Parameters are used to calculate positive feedback current with respect to the formula below.

    .. math ::
        I_{a} = \\dfrac{I_{a_{gain}}}{1+ exp\\left(-\\dfrac{I_{mem}+I_{a_{th}}}{I_{a_{norm}}}\\right)}

    :param Igain: Feedback gain current, heuristic parameter, defaults to 5e-11
    :type Igain: float, optional
    :param Ith: Feedback threshold current, typically a fraction of Ispk_th, defaults to 5e-10
    :type Ith: float, optional
    :param Inorm: Feedback normalization current, heuristic parameter, defaults to 1e-11
    :type Inorm: float, optional
    """

    Igain: float = 5e-11
    Ith: float = 5e-10
    Inorm: float = 1e-11


@dataclass
class MembraneParameters:
    """
    MembraneParameters contains membrane specific parameters and state variables

    :param Imem: The sub-threshold current that represents the real neuron’s membrane potential variable, defaults to 5e-13
    :type Imem: float, optional
    :param Itau: Membrane time constant current in Amperes, that is inversely proportional to time constant tau, defaults to 10e-12
    :type Itau: float, optional
    :param Ith: Membrane's threshold / gain current in Amperes, scaling factor for the membrane current (typically x2, x4 of I_tau), defaults to 40e-12
    :type Ith: float, optional
    :param feedback: positive feedback circuit block parameters:Ia_gain, Ia_th, and Ia_norm, defaults to None
    :type feedback: Optional[FeedbackParameters], optional
    """

    Imem: float = 5e-13
    Itau: float = 10e-12
    Ith: float = 40e-12
    feedback: Optional[FeedbackParameters] = None

    def __post_init__(self):
        """
        __post_init__ runs after __init__ and initializes the feedback block with default values in the case that it's not specified.
        """
        if self.feedback is None:
            self.feedback = FeedbackParameters()


@dataclass
class DynapSE1Parameters:
    """
    DynapSE1Parameters encapsulates the DynapSE1 circuit block parameters and provides an easy access.

    :param mem: Membrane block parameters (Imem, Itau, Ith, feedback(Igain, Ith, Inorm)), defaults to None
    :type mem: Optional[MembraneParameters], optional
    :param ahp: Spike frequency adaptation block parameters (Isyn, Itau, Ith, Iw), defaults to None
    :type ahp: Optional[DPIParameters], optional
    :param nmda: NMDA synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type nmda: Optional[DPIParameters], optional
    :param ampa: AMPA synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type ampa: Optional[DPIParameters], optional
    :param gaba_a: GABA_A synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type gaba_a: Optional[DPIParameters], optional
    :param gaba_b: GABA_B (shunt) synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
    :type gaba_b: Optional[DPIParameters], optional
    """

    mem: Optional[MembraneParameters] = None
    ahp: Optional[DPIParameters] = None
    nmda: Optional[DPIParameters] = None
    ampa: Optional[DPIParameters] = None
    gaba_a: Optional[DPIParameters] = None
    gaba_b: Optional[DPIParameters] = None

    def __post_init__(self):
        """
        __post_init__ runs after __init__ and initializes the DPI and membrane blocks with default values in the case that they are not specified.
        """

        if self.mem is None:
            self.mem = MembraneParameters()
        if self.ahp is None:
            self.ahp = DPIParameters()
        if self.nmda is None:
            self.nmda = DPIParameters()
        if self.ampa is None:
            self.ampa = DPIParameters()
        if self.gaba_a is None:
            self.gaba_a = DPIParameters()
        if self.gaba_b is None:
            self.gaba_b = DPIParameters()


class DynapSE1NeuronSynapseJax(JaxModule):
    """
    Implements the chip dynamical equations for the DPI neuron and synapse models
    Receives configuration as bias currents
    As few HW restrictions as possible
    [] ATTENTION TODO: Now, the implementation is only for one core
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
        layout: Optional[DynapSE1Layout] = None,
        params: Optional[DynapSE1Parameters] = None,
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
            params = DynapSE1Parameters()

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

        # [AHP, NMDA, AMPA, GABA_A, GABA_B]
        self.Isyn, self.Itau_syn, self.Ith_syn, self.Iw = self._set_syn_params(
            init=params
        )

        # [] TODO : out_rate is just been using to validate the model
        self.out_rate = out_rate

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
            spikes, Imem, Isyn, key = state

            # --- Utilities --- #  # 5xNin

            Iahp, Inmda, Iampa, Igaba_a, Igaba_b = Isyn
            _tau_syn = (f_tau_syn / self.Itau_syn.T).T  # 5xNin
            _Iss_syn = (self.Ith_syn / self.Itau_syn) * self.Iw  # 5xNin

            # --- Forward step --- #

            ## spiking inputs for 4 synapses NMDA, AMPA, GABA_A, GABA_B
            ## spiking output for 1 synapse AHP
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
            Isyn = f_discharge * Isyn

            # CHARGE PHASE -- UNDERSAMPLED -- dt >> t_pulse
            # f_charge array = 0 where there is no spike
            Isyn += f_charge * _Iss_syn

            # Make sure that synaptic currents are lower bounded by Io
            Isyn = np.clip(Isyn, Io)  # 5xNin

            # --- MEMBRANE DYNAMICS --- #
            # [] TODO : Change I_gaba_b dynamics. It's the shunt current

            tau_mem = self.layout.f_tau_mem / self.Itau_mem
            Iin = Inmda + Iampa - Igaba_a - Igaba_b + self.Idc
            Ia = self._feedback.Igain / (
                1 + np.exp(-(Imem + self._feedback.Ith) / (self._feedback.Inorm))
            )
            f_Imem = ((Ia) / (self.Itau_mem)) * (Imem + self.Ith_mem)
            Imem_inf = ((self.Ith_mem) / (self.Itau_mem)) * (Iin - Iahp - self.Itau_mem)

            ## MEMBRANE Forward Euler ##
            del_Imem = (
                (Imem / (tau_mem * (self.Ith_mem + Imem)))
                * (Imem_inf + f_Imem - (Imem * (1 + (Iahp / self.Itau_mem))))
                * self.dt
            )
            Imem = Imem + del_Imem

            # [] TODO: REMOVE
            # Random spike generation
            key, subkey = rand.split(key)
            spikes = rand.poisson(subkey, self.out_rate, (self.size_out,)).astype(float)

            state = (spikes, Imem, Isyn, key)
            return state, (spikes, Imem, Isyn)

        # - Evolve over spiking inputs
        state, (spikes_ts, Imem_ts, Isyn_ts) = scan(
            forward,
            (self.spikes, self.Imem, self.Isyn, self._rng_key),
            input_data,
        )

        (self.spikes, self.Imem, self.Isyn, self._rng_key) = state

        # - Generate return arguments
        outputs = spikes_ts

        states = {
            "spikes": self.spikes,
            "Imem": self.Imem,
            "Iahp": self.Isyn[0],
            "Inmda": self.Isyn[1],
            "Iampa": self.Isyn[2],
            "Igaba_a": self.Isyn[3],
            "Igaba_b": self.Isyn[4],
        }

        if record:
            record_dict = {
                "input_data": input_data,
                "spikes": spikes_ts,
                "Imem": Imem_ts,
                "Iahp": Isyn_ts[:, 0, :],
                "Inmda": Isyn_ts[:, 1, :],
                "Iampa": Isyn_ts[:, 2, :],
                "Igaba_a": Isyn_ts[:, 3, :],
                "Igaba_b": Isyn_ts[:, 4, :],
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

    def _set_syn_params(self, init: DynapSE1Parameters):

        _Isyn = np.array(
            [
                init.ahp.Isyn,
                init.nmda.Isyn,
                init.ampa.Isyn,
                init.gaba_a.Isyn,
                init.gaba_b.Isyn,
            ]
        )

        _Itau = np.array(
            [
                init.ahp.Itau,
                init.nmda.Itau,
                init.ampa.Itau,
                init.gaba_a.Itau,
                init.gaba_b.Itau,
            ]
        )

        _Ith = np.array(
            [
                init.ahp.Ith,
                init.nmda.Ith,
                init.ampa.Ith,
                init.gaba_a.Ith,
                init.gaba_b.Ith,
            ]
        )

        _Iw = np.array(
            [
                init.ahp.Iw,
                init.nmda.Iw,
                init.ampa.Iw,
                init.gaba_a.Iw,
                init.gaba_b.Iw,
            ]
        )

        # 5xNin

        Isyn: JP_ndarray = State(
            shape=(len(_Isyn), self.size_out),
            family="synapse",
            init_func=lambda s: (np.ones(s).T * _Isyn).T,
        )

        Itau: JP_ndarray = Parameter(
            shape=(len(_Itau), self.size_out),
            data=None,
            family="leak",
            init_func=lambda s: (np.ones(s).T * _Itau).T,
        )

        Ith: JP_ndarray = Parameter(
            shape=(len(_Ith), self.size_out),
            data=None,
            family="gain",
            init_func=lambda s: (np.ones(s).T * _Ith).T,
        )

        Iw: JP_ndarray = Parameter(
            shape=(len(_Iw), self.size_out),
            data=None,
            family="weight",
            init_func=lambda s: (np.ones(s).T * _Iw).T,
        )

        return Isyn, Itau, Ith, Iw

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
