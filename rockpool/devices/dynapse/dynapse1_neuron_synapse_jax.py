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

import numpy as onp

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter, State, SimulationParameter


import jax
import jax.random as rand
from jax.lax import scan
from jax import numpy as np

from dataclasses import dataclass

from typing import (
    Optional,
    Tuple,
    Any,
    Callable,
)

from rockpool.typehints import JP_ndarray, P_float, FloatVector, Value

from rockpool.devices.dynapse.utils import (
    get_param_vector,
    set_param,
)

from rockpool.devices.dynapse.dynapse1_parameters import (
    DynapSE1Parameters,
    SynapseParameters,
    MembraneParameters,
    FeedbackParameters,
)

DynapSE1State = Tuple[JP_ndarray, JP_ndarray, JP_ndarray, Optional[Any]]


@jax.custom_gradient
def step_pwl(
    Imem: FloatVector, Ispkthr: FloatVector, Ireset: FloatVector
) -> Tuple[FloatVector, Callable[[FloatVector], FloatVector]]:
    """
    step_pwl implements heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param x: Input current to be compared for firing
    :type x: FloatVector
    :param Ispkthr: Spiking threshold current in Amperes
    :type Ispkthr: FloatVector
    :param Ireset: Reset current after spike generation in Amperes
    :type Ireset: FloatVector
    :return: spike output value and gradient function
    :rtype: Tuple[FloatVector, Callable[[FloatVector], FloatVector]]
    """

    spikes = np.clip(np.floor(Imem - Ispkthr) + 1.0, 0.0)
    grad_func = lambda g: (g * (Imem > Ireset) * (Ispkthr - Ireset), 0.0, 0.0)
    return spikes, grad_func


class DynapSE1NeuronSynapseJax(JaxModule):
    """
    Solves the chip dynamical equations for the DPI neuron and synapse models.
    Receives configuration as bias currents and solves membrane and synapse dynamics using ``jax`` backend.
    One block has
        - 4 synapses receiving spikes from the other circuits,
        - 1 recurrent synapse for spike frequency adaptation,
        - 1 membrane evaluating the state and deciding fire or not

    For all the synapses, the ``DPI Synapse`` update equations below are solved in parallel.

    :DPI Synapse:

    .. math ::

        I_{syn}(t_1) = \\begin{cases} I_{syn}(t_0) \\cdot exp \\left( \\dfrac{-dt}{\\tau} \\right) &\\text{in any case} \\\\ \\\\ I_{syn}(t_1) + \\dfrac{I_{th} I_{w}}{I_{\\tau}} \\cdot \\left( 1 - exp \\left( \\dfrac{-t_{pulse}}{\\tau} \\right) \\right) &\\text{if a spike arrives} \\end{cases}

    Where

    .. math ::

        \\tau = \\dfrac{C U_{T}}{\\kappa I_{\\tau}}

    For the membrane update, the forward Euler solution below is applied.

    :Membrane:

    .. math ::

        dI_{mem} = \dfrac{I_{mem}}{\tau \left( I_{mem} + I_{th} \right) } \cdot \left( I_{mem_{\infty}} + f(I_{mem}) - I_{mem} \left( 1 + \dfrac{I_{ahp}}{I_{\tau}} \right) \right) \cdot dt

        I_{mem}(t_1) = I_{mem}(t_0) + dI_{mem}

    Where

    .. math ::

        I_{mem_{\\infty}} = \\dfrac{I_{th}}{I_{\\tau}} \\left( I_{in} - I_{ahp} - I_{\\tau}\\right)

        f(I_{mem}) = \\dfrac{I_{a}}{I_{\\tau}} \\left(I_{mem} + I_{th} \\right )

        I_{a} = \\dfrac{I_{a_{gain}}}{1+ exp\\left(-\\dfrac{I_{mem}+I_{a_{th}}}{I_{a_{norm}}}\\right)}

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`I_{mem, j}` exceeds the threshold current :math:`I_{spkthr}`, then the neuron emits a spike.

    .. math ::

        I_{mem, j} > I_{spkthr} \\rightarrow S_{j} = 1

        I_{mem, j} = I_{reset}

    :Parameters:

    :param shape: The number of neruons to employ, defaults to None
    :type shape: tuple, optional
    :param layout: constant values that are related to the exact silicon layout of a chip, defaults to None
    :type layout: Optional[DynapSE1Layout], optional
    :param config: Dynap-SE1 bias currents and configuration parameters, defaults to None
    :type config: Optional[DynapSE1Parameters], optional
    :param dt: The time step for the forward-Euler ODE solve, defaults to 1e-3
    :type dt: float, optional
    :param rng_key: The Jax RNG seed to use on initialisation. By default, a new seed is generated, defaults to None
    :type rng_key: Optional[Any], optional
    :param spiking_input: Whether this module receives spiking input, defaults to True
    :type spiking_input: bool, optional
    :param spiking_output: Whether this module produces spiking output, defaults to True
    :type spiking_output: bool, optional
    :raises ValueError: When the user does not provide a valid shape

    :Instance Variables:

    :ivar Imem: Array of membrane currents of the neurons with shape = (Nin,)
    :type Imem: JP_ndarray
    :ivar Itau_mem: Array of membrane leakage currents of the neurons with shape = (Nin,)
    :type Itau_mem: JP_ndarray
    :ivar Ith_mem: Array of membrane gain currents of the neurons with shape = (Nin,)
    :type Ith_mem: JP_ndarray
    :ivar mem_fb: positive feedback circuit heuristic parameters:Ia_gain, Ia_th, and Ia_norm
    :type mem_fb: FeedbackParameters
    :ivar Isyn: 2D array of synapse currents of the neurons in the order of [AHP, NMDA, AMPA, GABA_A, GABA_B] with shape = (5,Nin)
    :type Isyn: JP_ndarray
    :ivar Itau_syn: 2D array of synapse leakage currents of the neurons in the order of [AHP, NMDA, AMPA, GABA_A, GABA_B] with shape = (5,Nin)
    :type Itau_syn: JP_ndarray
    :ivar Ith_syn: 2D array of synapse gain currents of the neurons in the order of [AHP, NMDA, AMPA, GABA_A, GABA_B] with shape = (5,Nin)
    :type Ith_syn: JP_ndarray
    :ivar Iw: 2D array of synapse weight currents of the neurons in the order of [AHP, NMDA, AMPA, GABA_A, GABA_B] with shape = (5,Nin)
    :type Iw: JP_ndarray
    :ivar spikes: Logical spiking raster for each neuron at the last simulation time-step with shape (Nin,)
    :type spikes: JP_ndarray
    :ivar Ispkthr: Spiking threshold current in with shape (Nin,)
    :type Ispkthr: JP_ndarray
    :ivar Ireset: Reset current after spike generation with shape (Nin,)
    :type Ireset: JP_ndarray

    [] TODO: ATTENTION Now, the implementation is only for one core, extend it for multiple cores
    [] TODO: think about activating and deactivating certain circuit blocks.
    [] TODO: all neurons cannot have the same parameters ideally however, they experience different parameters in practice because of device mismatch
    [] TODO: What is the initial configuration of biases?
    [] TODO: How to convert from bias current parameters to high-level parameters and vice versa?
    [] TODO: Provides mismatch simulation (as second step)
        -As a utility function that operates on a set of parameters?
        -As a method on the class?
    """

    def __init__(
        self,
        shape: tuple = None,
        config: Optional[DynapSE1Parameters] = None,
        dt: float = 1e-3,
        rng_key: Optional[Any] = None,
        spiking_input: bool = True,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ Initialize ``DynapSE1NeuronSynapseJax`` module. Parameters are explained in the class docstring.
        """

        # Check the parameters and initialize to default if necessary
        if shape is None:
            raise ValueError("You must provide a ``shape`` tuple (N,)")

        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))

        if config is None:
            config = DynapSE1Parameters()

        _, rng_key = rand.split(np.array(rng_key, dtype=np.uint32))
        self._rng_key: JP_ndarray = State(rng_key, init_func=lambda _: rng_key)

        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        # --- Parameters & States --- #
        self.Imem, self.Itau_mem, self.Ith_mem, self.mem_fb = self._set_mem_params(
            init=config.mem,
        )

        ## Synapses parameters are combined in the order of [AHP, NMDA, AMPA, GABA_A, GABA_B]
        self.Isyn, self.Itau_syn, self.Ith_syn, self.Iw = self._set_syn_params(
            ahp=config.ahp,
            nmda=config.nmda,
            ampa=config.ampa,
            gaba_a=config.gaba_a,
            gaba_b=config.gaba_b,
        )
        self.spikes: JP_ndarray = State(shape=(self.size_out,), init_func=np.zeros)

        # --- Simulation Parameters --- #
        self.dt: P_float = SimulationParameter(dt)

        ## Layout Params
        self.Io = SimulationParameter(config.layout.Io)

        ## Configuration Parameters
        self.f_tau_mem = SimulationParameter(config.f_tau_mem)
        self.f_tau_syn = SimulationParameter(config.f_tau_syn)
        self.t_pulse = SimulationParameter(config.t_pulse)
        self.t_pulse_ahp = SimulationParameter(config.t_pulse_ahp)
        self.Idc = SimulationParameter(config.Idc)
        self.If_nmda = SimulationParameter(config.If_nmda)
        self.t_ref = SimulationParameter(config.t_ref)

        ### Policy currents
        self.Ireset: JP_ndarray = SimulationParameter(
            shape=(self.size_out,),
            family="simulation",
            init_func=lambda s: np.ones(s) * config.Ireset,
        )
        self.Ispkthr: JP_ndarray = SimulationParameter(
            shape=(self.size_out,),
            family="simulation",
            init_func=lambda s: np.ones(s) * config.Ispkthr,
        )

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        evolve implements raw JAX evolution function for a DynapSE1NeuronSynapseJax module.
        The function solves the dynamical equations introduced at the ``DynapSE1NeuronSynapseJax`` module definition

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

        def forward(
            state: DynapSE1State, spike_inputs_ts: np.ndarray
        ) -> Tuple[DynapSE1State, Tuple[JP_ndarray, JP_ndarray, JP_ndarray]]:
            """
            forward implements single time-step neuron and synapse dynamics

            :param state: (spikes, Imem, Isyn, key)
                spikes: Logical spike raster for each neuron [Nin]
                Imem: Membrane currents of each neuron [Nin]
                Isyn: Synapse currents of each synapses[AHP, NMDA, AMPA, GABA_A, GABA_B] of each neuron [5xNin]
                key: The Jax RNG seed to be used for mismatch simulation
            :type state: DynapSE1State
            :param spike_inputs_ts: incoming spike raster to be used as an axis [T, Nin]
            :type spike_inputs_ts: np.ndarray
            :return: state, (spikes, Imem, Isyn)
                state: Updated state at end of the forward steps
                spikes: Logical spiking raster for each neuron over time [Nin]
                Imem: Updated membrane membrane currents of each neuron [Nin]
                Isyn: Updated synapse currents of each synapses[AHP, NMDA, AMPA, GABA_A, GABA_B] of each neuron [5xNin]
            :rtype: Tuple[DynapSE1State, Tuple[JP_ndarray, JP_ndarray, JP_ndarray]]
            """
            # [] TODO : Change I_gaba_b dynamics. It's the shunt current
            # [] TODO : Implement NMDA gating mechanism
            # [] TODO : Would you allow currents to go below Io or not?!!!!

            spikes, Imem, Isyn, key = state

            # Reset Imem depending on spiking activity
            Imem = (1 - spikes) * Imem + spikes * self.Ireset

            ## ATTENTION : Optimization can make Itau_mem and I_tau_syn < Io
            # We might have division by 0 if we allow this to happen!
            Itau_mem_clip = np.clip(self.Itau_mem, self.Io)
            Ith_mem_clip = np.clip(self.Ith_mem, self.Io)
            Itau_syn_clip = np.clip(self.Itau_syn, self.Io)
            Ith_syn_clip = np.clip(self.Ith_syn, self.Io)

            # --- Implicit parameters  --- #  # 5xNin
            tau_mem = self.f_tau_mem / Itau_mem_clip
            tau_syn = (self.f_tau_syn / Itau_syn_clip.T).T
            Isyn_inf = (Ith_syn_clip / Itau_syn_clip) * self.Iw

            # --- Forward step: DPI SYNAPSES --- #

            ## spike input for 4 synapses: NMDA, AMPA, GABA_A, GABA_B; spike output for 1 synapse: AHP
            spike_inputs = np.tile(spike_inputs_ts, (4, 1))

            ## Calculate the effective pulse width with a linear increase
            t_pw_in = self.t_pulse * spike_inputs  # 4xNin [NMDA, AMPA, GABA_A, GABA_B]
            t_pw_out = self.t_pulse_ahp * spikes  # 1xNin [AHP]
            t_pw = np.vstack((t_pw_out, t_pw_in))

            ## Exponential charge and discharge factor arrays
            f_charge = 1 - np.exp(-t_pw / tau_syn)  # 5xNin
            f_discharge = np.exp(-self.dt / tau_syn)  # 5xNin

            ## DISCHARGE in any case
            Isyn = f_discharge * Isyn

            ## CHARGE if spike occurs -- UNDERSAMPLED -- dt >> t_pulse
            Isyn += f_charge * Isyn_inf
            Isyn = np.clip(Isyn, self.Io)  # 5xNin

            # --- Forward step: MEMBRANE --- #

            ## Decouple synaptic currents and calculate membrane input
            Iahp, Inmda, Iampa, Igaba_a, Igaba_b = Isyn
            # Ishunt = np.clip(Igaba_b, self.layout.Io, Imem) # Not sure how to use
            # Inmda = 0 if Vmem < Vth_nmda else Inmda
            Iin = Inmda + Iampa - Igaba_a - Igaba_b + self.Idc
            Iin = np.clip(Iin, self.Io)

            ## Steady state current
            Imem_inf = ((Ith_mem_clip) / (Itau_mem_clip)) * (Iin - Iahp - Itau_mem_clip)

            ## Positive feedback
            Ia = self.mem_fb.Igain / (
                1 + np.exp(-(Imem - self.mem_fb.Ith) / self.mem_fb.Inorm)
            )
            Ia = np.clip(Ia, self.Io)
            f_Imem = ((Ia) / (Itau_mem_clip)) * (Imem + Ith_mem_clip)

            ## Forward Euler Update
            del_Imem = (Imem / (tau_mem * (Ith_mem_clip + Imem))) * (
                Imem_inf + f_Imem - (Imem * (1 + (Iahp / Itau_mem_clip)))
            )
            Imem = Imem + del_Imem * self.dt
            Imem = np.clip(Imem, self.Io)

            # --- Spike Generation Logic --- #
            ## Detect next spikes (with custom gradient)
            spikes = step_pwl(Imem, self.Ispkthr, self.Ireset)

            state = (spikes, Imem, Isyn, key)
            return state, (spikes, Imem, Isyn)

        # --- Evolve over spiking inputs --- #
        state, (spikes_ts, Imem_ts, Isyn_ts) = scan(
            forward, (self.spikes, self.Imem, self.Isyn, self._rng_key), input_data
        )

        new_spikes, new_Imem, new_Isyn, new_rng_key = state

        # --- RETURN ARGUMENTS --- #
        outputs = spikes_ts

        ## the state returned should be in the same shape with the state dictionary given
        states = {
            "_rng_key": new_rng_key,
            "Imem": new_Imem,
            "Isyn": new_Isyn,
            "spikes": new_spikes,
        }

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

        return outputs, states, record_dict

    def _set_syn_params(
        self,
        ahp: Optional[SynapseParameters] = None,
        nmda: Optional[SynapseParameters] = None,
        ampa: Optional[SynapseParameters] = None,
        gaba_a: Optional[SynapseParameters] = None,
        gaba_b: Optional[SynapseParameters] = None,
    ) -> Tuple[JP_ndarray, JP_ndarray, JP_ndarray, JP_ndarray]:
        """
        _set_syn_params helps constructing and initiating synapse parameters and states for ["AHP", "NMDA", "AMPA", "GABA_A", "GABA_B"]

        :param ahp: Spike frequency adaptation block parameters (Isyn, Itau, Ith, Iw), defaults to None
        :type ahp: Optional[SynapseParameters], optional
        :param nmda: NMDA synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
        :type nmda: Optional[SynapseParameters], optional
        :param ampa: AMPA synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
        :type ampa: Optional[SynapseParameters], optional
        :param gaba_a: GABA_A synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
        :type gaba_a: Optional[SynapseParameters], optional
        :param gaba_b: GABA_B (shunt) synapse paramters (Isyn, Itau, Ith, Iw), defaults to None
        :type gaba_b: Optional[SynapseParameters], optional
        :return: Isyn, Itau, Ith, Iw : states and parameters in the order of [AHP, NMDA, AMPA, GABA_A, GABA_B] with shape = (5,Nin)
            Isyn: 2D array of synapse currents (State)
            Itau: 2D array of synapse leakage currents (Parameter)
            Ith: 2D array of synapse gain currents (Parameter)
            Iw: 2D array of synapse weight currents (Parameter)
        :rtype: Tuple[JP_ndarray, JP_ndarray, JP_ndarray, JP_ndarray]
        """

        dpi_list = (ahp, nmda, ampa, gaba_a, gaba_b)

        def get_dpi_parameter(
            target: str, family: str, object: Optional[str] = "parameter"
        ) -> JP_ndarray:
            """
            get_dpi_parameter encapsulates required data management to set a synaptic parameter

            :param target: target parameter to be extracted from the DPIParameters object: Isyn, Itau, Ith, or Iw
            :type target: str
            :param family: the parameter family name
            :type family: str
            :param object: the object type to be constructed. It can be "state", "parameter" or "simulation"
            :type object: Optional[str], optional
            :return: constructed parameter or the state variable
            :rtype: JP_ndarray
            """
            _Iparam = get_param_vector(dpi_list, target)
            shape = (len(_Iparam), self.size_out)
            init_func = lambda s: (np.ones(s).T * _Iparam).T
            Iparam = set_param(shape, family, init_func, object)

            return Iparam

        # Construct the parameter objects
        Isyn = get_dpi_parameter("Isyn", "synapse", object="state")
        Itau = get_dpi_parameter("Itau", "leak")
        Ith = get_dpi_parameter("Ith", "gain")
        Iw = get_dpi_parameter("Iw", "weight")

        return Isyn, Itau, Ith, Iw

    def _set_mem_params(
        self, init: MembraneParameters, family: Optional[str] = "membrane"
    ) -> Tuple[JP_ndarray, JP_ndarray, JP_ndarray, FeedbackParameters]:
        """
        _set_mem_params constructs and initiates membrane parameters and states

        :param init: Inital membrane block parameters (Imem, Itau, Ith, feedback(Igain, Ith, Inorm))
        :type init: MembraneParameters
        :param family: the parameter family name, defaults to "membrane"
        :type family: Optional[str], optional
        :return: Imem, Itau, Ith, feedback
            Imem: Array of membrane currents with shape = (Nin,) (State)
            Itau: Array of membrane leakage currents with shape = (Nin,) (Parameter)
            Ith: Array of membrane gain currents with shape = (Nin,) (Parameter)
            feedback: positive feedback circuit heuristic parameters: Ia_gain, Ia_th, and Ia_norm
        :rtype: Tuple[JP_ndarray, JP_ndarray, JP_ndarray, FeedbackParameters]
        """

        def get_mem_parameter(
            target: str, object: Optional[str] = "parameter"
        ) -> JP_ndarray:
            """
            get_mem_parameter encapsulates required data management for setting a membrane parameter

            :param target: target parameter to be extracted from the MembraneParameters object: Imem, Itau, or Ith
            :type target: str
            :param object: the object type to be constructed. It can be "state", "parameter" or "simulation"
            :type object: Optional[str], optional
            :return: constructed parameter or the state variable
            :rtype: JP_ndarray
            """

            shape = (self.size_out,)
            init_func = lambda s: np.ones(s) * init.__getattribute__(target)
            Iparam = set_param(shape, family, init_func, object)
            return Iparam

        Imem = get_mem_parameter("Imem", object="state")
        Itau = get_mem_parameter("Itau")
        Ith = get_mem_parameter("Ith")

        feedback: FeedbackParameters = init.feedback

        return Imem, Itau, Ith, feedback