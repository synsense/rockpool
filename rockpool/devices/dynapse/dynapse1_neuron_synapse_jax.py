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

from typing import (
    Union,
    Optional,
    Tuple,
    Any,
    Callable,
)

from rockpool.typehints import (
    JP_ndarray,
    FloatVector,
)

from rockpool.devices.dynapse.dynapse1_simconfig import DynapSE1SimCore


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
    DynapSE1NeuronSynapseJax Solves the chip dynamical equations for the DPI neuron and synapse models.
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

    :Attributes:

    :attr biases: name list of all the low level biases
    :type biases: List[str]

    :Parameters:

    :param shape: The number of neruons to employ, defaults to None
    :type shape: tuple, optional
    :param sim_config: Dynap-SE1 bias currents and simulation configuration parameters, defaults to None
    :type sim_config: Optional[DynapSE1SimCore], optional
    :param w_in: Initial input weights defining the connections from virtual FPGA neurons to real device neurons. It must be a rectangular matrix with shape ``(Nin, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types.
    :type w_in: Optional[FloatVector], optional

        #  Gb Ga N  A
         [[0, 0, 0, 1], # pre = 0 (virtual) post = 0 (device)
          [0, 0, 0, 1],  #                  post = 1 (device)
          [0, 0, 0, 0],  #                  post = 2 (device)
          [0, 0, 0, 0],  #                  post = 3 (device)
          [0, 0, 0, 1]], #                  post = 4 (device)

         [[0, 0, 0, 0], # pre = 1 (virtual)
          [0, 0, 0, 0],
          [0, 0, 1, 0],
          [0, 0, 1, 0],
          [0, 0, 0, 1]],

         [[0, 0, 0, 0], # pre = 3 (virtual)
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 1, 0, 0]]],

        Virtual(External Input)

            AMPA : 1 from n5 to n0, 1 from n5 to n1 1 from n5 to n4
            NMDA : 1 from n6 to n2, 1 from n6 to n3 1 from n6 to n4
            GABA_A: 1 from n7 to n4

    :param w_rec: If the module is initialised in recurrent mode, one can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(Nrec, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
    :type w_rec: Optional[FloatVector], optional

        Let's say 5 device neuron and 3 virtual FPGA neurons initiated with w_rec

        #  Gb Ga N  A
        [[[0, 0, 0, 0],  # pre = 0 (device) post = 0 (device)
          [0, 0, 0, 1],  #                  post = 1 (device)
          [0, 0, 0, 0],  #                  post = 2 (device)
          [0, 0, 0, 0],  #                  post = 3 (device)
          [0, 1, 0, 0]], #                  post = 4 (device)

         [[0, 0, 0, 0], # pre = 1 (device)
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0]],

         [[2, 0, 0, 0], # pre = 2 (device)
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0]],

         [[0, 0, 0, 0], # pre = 3 (device)
          [0, 0, 0, 0],
          [0, 0, 0, 1],
          [0, 0, 0, 0],
          [0, 0, 0, 0]],

         [[0, 0, 0, 0], # pre = 4 (device)
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 0, 0],
          [0, 0, 1, 0]],

        Real
            AMPA : 1 from n0 to n1, 1 from n2 to n3, 1 from n3 to n2
            NMDA : 1 from n4 to n4
            GABA_A: 1 from n0 to n4
            GABA_B: 2 from n2 to n0

    :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
    :type dt: float, optional
    :param rng_key: The Jax RNG seed to use on initialisation. By default, a new seed is generated, defaults to None
    :type rng_key: Optional[Any], optional
    :param spiking_input: Whether this module receives spiking input, defaults to True
    :type spiking_input: bool, optional
    :param spiking_output: Whether this module produces spiking output, defaults to True
    :type spiking_output: bool, optional
    :raises ValueError: When the user does not provide a valid shape

    :Instance Variables:

    :ivar SYN: A dictionary storing default indexes(order) of the synapse types
    :type SYN: Dict[str, int]
    :ivar Isyn: 2D array of synapse currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape (5,Nrec)
    :type Isyn: JP_ndarray
    :ivar Imem: Array of membrane currents of the neurons with shape (Nrec,)
    :type Imem: JP_ndarray
    :ivar spikes: Logical spiking raster for each neuron at the last simulation time-step with shape (Nrec,)
    :type spikes: JP_ndarray
    :ivar timer_ref: timer to keep the time from the spike generation until the refractory period ends
    :type timer_ref: int
    :ivar Itau_syn: 2D array of synapse leakage currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape (5,Nrec)
    :type Itau_syn: JP_ndarray
    :ivar f_gain_syn: 2D array of synapse gain parameters of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape (5,Nrec)
    :type f_gain_syn: JP_ndarray
    :ivar Iw: 2D array of synapse weight currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape (5,Nrec)
    :type Iw: JP_ndarray
    :ivar Itau_mem: Array of membrane leakage currents of the neurons with shape (Nrec,)
    :type Itau_mem: JP_ndarray
    :ivar f_gain_mem: Array of membrane gain parameter of the neurons with shape (Nrec,)
    :type f_gain_mem: JP_ndarray
    :ivar Idc: Array of constant DC current in Amperes, injected to membrane with shape (Nrec,)
    :type Idc: JP_ndarray
    :ivar If_nmda: Array of the NMDA gate current in Amperes setting the NMDA gating voltage. If :math:`V_{mem} > V_{nmda}` : The :math:`I_{syn_{NMDA}}` current is added up to the input current, else it cannot with shape (Nrec,)
    :type If_nmda: JP_ndarray
    :ivar Io: Array of Dark current in Amperes that flows through the transistors even at the idle state with shape (Nrec,)
    :type Io: JP_ndarray
    :ivar Ip_gain: Array of positive feedback gain current, heuristic parameter with shape (Nrec,)
    :type Ip_gain: JP_ndarray
    :ivar Ip_th: Array of positive feedback threshold current, typically a fraction of Ispk_th with shape (Nrec,)
    :type Ip_th: JP_ndarray
    :ivar Ip_norm: Array of positive feedback normalization current, heuristic parameter with shape (Nrec,)
    :type Ip_norm: JP_ndarray
    :ivar f_tau_mem: Array of tau factor for membrane circuit. :math:`f_{\\tau} = \\dfrac{U_T}{\\kappa \\cdot C}`, :math:`f_{\\tau} = I_{\\tau} \\cdot \\tau` with shape (Nrec,)
    :type f_tau_mem: JP_ndarray
    :ivar f_tau_syn: Array of tau factors in the following order: [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape (5,Nrec)
    :type f_tau_syn: np.ndarray
    :ivar t_pulse: Array of the width of the pulse in seconds produced by virtue of a spike with shape (Nrec,)
    :type t_pulse: JP_ndarray
    :ivar t_pulse_ahp: Array of reduced pulse width also look at ``t_pulse`` and ``fpulse_ahp`` with shape (Nrec,)
    :type t_pulse_ahp: JP_ndarray
    :ivar t_ref: Array of refractory periods in seconds, limits maximum firing rate with shape (Nrec,)
    :type t_ref: JP_ndarray
    :ivar Ispkthr: Array of spiking threshold current in with shape (Nrec,)
    :type Ispkthr: JP_ndarray
    :ivar Ireset: Array of reset current after spike generation with shape (Nrec,)
    :type Ireset: JP_ndarray


    [] TODO: ATTENTION Now, the implementation is only for one core, extend it for multiple cores
    [] TODO: all neurons cannot have the same parameters ideally however, they experience different parameters in practice because of device mismatch
    [] TODO: Provides mismatch simulation (as second step)
        -As a utility function that operates on a set of parameters?
        -As a method on the class?
    """

    __doc__ += "\nJaxModule" + JaxModule.__doc__

    syn_types = ["GABA_B", "GABA_A", "NMDA", "AMPA", "AHP"]
    SYN = dict(zip(syn_types, range(len(syn_types))))

    def __init__(
        self,
        shape: tuple = None,
        sim_config: Optional[DynapSE1SimCore] = None,
        w_in: Optional[FloatVector] = None,
        w_rec: Optional[FloatVector] = None,
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
            raise ValueError("You must provide a ``shape`` tuple (N,) or (Nin,Nrec)")

        if rng_key is None:
            rng_key = rand.PRNGKey(onp.random.randint(0, 2 ** 63))

        _, rng_key = rand.split(np.array(rng_key, dtype=np.uint32))
        self._rng_key: JP_ndarray = State(rng_key, init_func=lambda _: rng_key)

        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        if sim_config is None:
            sim_config = DynapSE1SimCore(size=self.size_out)

        else:  # be sure that the size matches
            sim_config.size = self.size_out

        # Check the network size and initialize the input and recurrent weight vector accordingly
        self.w_in, self.w_rec = self._init_weights(w_in, w_rec)

        # --- Parameters & States --- #

        # --- States --- #
        self.Isyn = State(sim_config.Isyn, shape=(5, self.size_out))
        self.Imem = State(sim_config.Imem, shape=(self.size_out,))
        self.spikes = State(shape=(self.size_out,), init_func=np.zeros)
        self.timer_ref = State(shape=(self.size_out,), init_func=np.zeros)

        # --- Parameters --- #
        ## Synapse
        self.Itau_syn = Parameter(sim_config.Itau_syn, shape=(5, self.size_out))
        self.f_gain_syn = Parameter(sim_config.f_gain_syn, shape=(5, self.size_out))
        self.Iw = Parameter(sim_config.Iw, shape=(5, self.size_out))

        ## Membrane
        self.Itau_mem = Parameter(sim_config.Itau_mem, shape=(self.size_out,))
        self.f_gain_mem = Parameter(sim_config.f_gain_mem, shape=(self.size_out,))
        self.Idc = Parameter(sim_config.Idc, shape=(self.size_out,))
        self.If_nmda = Parameter(sim_config.If_nmda, shape=(self.size_out,))

        # --- Simulation Parameters --- #
        self.dt = SimulationParameter(dt, shape=())
        self.Io = SimulationParameter(sim_config.Io, shape=(self.size_out,))

        ## Positive Feedback
        self.Ip_gain = SimulationParameter(sim_config.Ip_gain, shape=(self.size_out,))
        self.Ip_th = SimulationParameter(sim_config.Ip_th, shape=(self.size_out,))
        self.Ip_norm = SimulationParameter(sim_config.Ip_norm, shape=(self.size_out,))

        ## Time -> Current conversion
        self.f_tau_mem = SimulationParameter(
            sim_config.f_tau_mem, shape=(self.size_out,)
        )
        self.f_tau_syn = SimulationParameter(
            sim_config.f_tau_syn, shape=(5, self.size_out)
        )

        # Pulse width
        self.t_pulse = SimulationParameter(sim_config.t_pulse, shape=(self.size_out,))
        self.t_pulse_ahp = SimulationParameter(
            sim_config.t_pulse_ahp, shape=(self.size_out,)
        )

        ## Refractory Period
        self.t_ref = SimulationParameter(sim_config.t_ref, shape=(self.size_out,))

        ## Policy
        self.Ispkthr = SimulationParameter(sim_config.Ispkthr, shape=(self.size_out,))
        self.Ireset = SimulationParameter(sim_config.Ireset, shape=(self.size_out,))

    def evolve(
        self, input_data: np.ndarray, record: bool = True
    ) -> Tuple[np.ndarray, dict, dict]:
        """
        evolve implements raw JAX evolution function for a DynapSE1NeuronSynapseJax module.
        The function solves the dynamical equations introduced at the ``DynapSE1NeuronSynapseJax`` module definition

        :param input_data: Input array of shape ``(T, Nrec)`` to evolve over. Represents number of spikes at that timebin
        :type input_data: np.ndarray
        :param record: record the each timestep of evolution or not, defaults to False
        :type record: bool, optional
        :return: outputs, states, record_dict
            :outputs: is an array with shape ``(T, Nout)`` containing the output data(spike raster) produced by this module.
            :states: is a dictionary containing the updated module state following evolution.
            :record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True``.
        :rtype: Tuple[np.ndarray, dict, dict]
        """

        def forward(
            state: DynapSE1State, spike_inputs_ts: np.ndarray
        ) -> Tuple[DynapSE1State, Tuple[JP_ndarray, JP_ndarray, JP_ndarray]]:
            """
            forward implements single time-step neuron and synapse dynamics

            :param state: (spikes, Imem, Isyn, key)
                spikes: Logical spike raster for each neuron [Nrec]
                Imem: Membrane currents of each neuron [Nrec]
                Isyn: Synapse currents of each synapses[GABA_B, GABA_A, NMDA, AMPA, AHP] of each neuron [5xNrec]
                key: The Jax RNG seed to be used for mismatch simulation
            :type state: DynapSE1State
            :param spike_inputs_ts: incoming spike raster to be used as an axis [T, Nrec]
            :type spike_inputs_ts: np.ndarray
            :return: state, (spikes, Imem, Isyn)
                state: Updated state at end of the forward steps
                spikes: Logical spiking raster for each neuron over time [Nrec]
                Imem: Updated membrane membrane currents of each neuron [Nrec]
                Isyn: Updated synapse currents of each synapses[GABA_B, GABA_A, NMDA, AMPA, AHP] of each neuron [5xNrec]
            :rtype: Tuple[DynapSE1State, Tuple[JP_ndarray, JP_ndarray, JP_ndarray]]
            """
            # [] TODO : Would you allow currents to go below Io or not?!!!!

            spikes, Imem, Isyn, key, timer_ref = state

            # Reset Imem depending on spiking activity
            Imem = (1 - spikes) * Imem + spikes * self.Ireset

            # Set the refractrory timer
            timer_ref -= self.dt
            timer_ref = np.clip(timer_ref, 0)
            timer_ref = (1 - spikes) * timer_ref + spikes * self.t_ref

            ## ATTENTION : Optimization can make Itau_mem and I_tau_syn < Io
            # We might have division by 0 if we allow this to happen!
            Itau_mem_clip = np.clip(self.Itau_mem, self.Io)
            Itau_syn_clip = np.clip(self.Itau_syn, self.Io)

            # --- Implicit parameters  --- #  # 5xNrec
            tau_mem = self.f_tau_mem / Itau_mem_clip
            tau_syn = self.f_tau_syn / Itau_syn_clip
            Isyn_inf = self.f_gain_syn * self.Iw

            # --- Forward step: DPI SYNAPSES --- #
            ## spike input for 4 synapses: GABA_B, GABA_A, NMDA, AMPA; spike output for 1 synapse: AHP
            ## w_in.shape = NinxNrecx4 [pre,post,syn]
            ## w_rec.shape = NrecxNrecx4 [pre,post,syn]

            spikes_external = np.dot(self.w_in.T, spike_inputs_ts)
            spikes_internal = np.dot(self.w_rec.T, spikes)
            spike_inputs = np.add(spikes_external, spikes_internal) + self.Io

            ## Calculate the effective pulse width with a linear increase
            t_pw_in = self.t_pulse * spike_inputs  # 4xNrec [GABA_B, GABA_A, NMDA, AMPA]
            t_pw_out = self.t_pulse_ahp * spikes  # 1xNrec [AHP]
            t_pw = np.vstack((t_pw_in, t_pw_out))

            ## Exponential charge and discharge factor arrays
            f_charge = 1 - np.exp(-t_pw / tau_syn)  # 5xNrec
            f_discharge = np.exp(-self.dt / tau_syn)  # 5xNrec

            ## DISCHARGE in any case
            Isyn = f_discharge * Isyn

            ## CHARGE if spike occurs -- UNDERSAMPLED -- dt >> t_pulse
            Isyn += f_charge * Isyn_inf
            Isyn = np.clip(Isyn, self.Io)  # 5xNrec

            # --- Forward step: MEMBRANE --- #

            ## Decouple synaptic currents and calculate membrane input
            Igaba_b, Igaba_a, Inmda, Iampa, Iahp = Isyn

            # Inmda = 0 if Vmem < Vth_nmda else Inmda
            I_nmda_dp = Inmda / (1 + self.If_nmda / Imem)

            # Iin = 0 if the neuron is in the refractory period
            Iin = I_nmda_dp + Iampa - Igaba_b + self.Idc
            Iin *= np.logical_not(timer_ref.astype(bool))
            Iin = np.clip(Iin, self.Io)

            ## Steady state current
            Imem_inf = self.f_gain_mem * (Iin - (Iahp + Igaba_a) - Itau_mem_clip)
            Ith_mem_clip = self.f_gain_mem * Itau_mem_clip

            ## Positive feedback
            Ia = self.Ip_gain / (1 + np.exp(-(Imem - self.Ip_th) / self.Ip_norm))
            Ia = np.clip(Ia, self.Io)
            f_Imem = ((Ia) / (Itau_mem_clip)) * (Imem + Ith_mem_clip)

            ## Forward Euler Update
            del_Imem = (Imem / (tau_mem * (Ith_mem_clip + Imem))) * (
                Imem_inf + f_Imem - (Imem * (1 + ((Iahp + Igaba_a) / Itau_mem_clip)))
            )
            Imem = Imem + del_Imem * self.dt
            Imem = np.clip(Imem, self.Io)

            # --- Spike Generation Logic --- #
            ## Detect next spikes (with custom gradient)
            spikes = step_pwl(Imem, self.Ispkthr, self.Ireset)

            state = (spikes, Imem, Isyn, key, timer_ref)
            return state, (spikes, Imem, Isyn)

        # --- Evolve over spiking inputs --- #
        state, (spikes_ts, Imem_ts, Isyn_ts) = scan(
            forward,
            (self.spikes, self.Imem, self.Isyn, self._rng_key, self.timer_ref),
            input_data,
        )

        new_spikes, new_Imem, new_Isyn, new_rng_key, new_timer_ref = state

        # --- RETURN ARGUMENTS --- #
        outputs = spikes_ts

        ## the state returned should be in the same shape with the state dictionary given
        states = {
            "_rng_key": new_rng_key,
            "Imem": new_Imem,
            "Isyn": new_Isyn,
            "spikes": new_spikes,
            "timer_ref": new_timer_ref,
        }

        record_dict = {
            "input_data": input_data,
            "spikes": spikes_ts,
            "Imem": Imem_ts,
            "Igaba_b": Isyn_ts[:, self.SYN["GABA_B"], :],
            "Igaba_a": Isyn_ts[:, self.SYN["GABA_A"], :],
            "Inmda": Isyn_ts[:, self.SYN["NMDA"], :],
            "Iampa": Isyn_ts[:, self.SYN["AMPA"], :],
            "Iahp": Isyn_ts[:, self.SYN["AHP"], :],
        }

        return outputs, states, record_dict

    def _init_weights(
        self,
        w_in: Optional[FloatVector] = None,
        w_rec: Optional[FloatVector] = None,
    ) -> Union[JP_ndarray, float]:
        """
        _init_w_rec Intialize a recurrent weight matrix parameter given the network shape.

        :param w_in: Initial input weights defining the connections from virtual FPGA neurons to real device neurons. It must be a rectangular matrix with shape ``(Nin, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types.
        :type w_in: Optional[FloatVector], optional
        :param w_rec: If the module is initialised in recurrent mode, one can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(Nrec, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
        :type w_rec: Optional[FloatVector], optional
        :raises ValueError: If `shape` is unidimensional, then `w_rec` may not be provided as an argument.
        :raises ValueError: `shape` may not specify more than two dimensions (Nin, Nrec).
        :return: Recurrent weight matrix parameter initialized randomly or depending on an initial weight vector.
        :rtype: Union[JP_ndarray, float]
        """

        def get_weight_matrix(
            weight_matrix: Optional[FloatVector], shape: Tuple[int]
        ) -> JP_ndarray:
            """
            get_weight_matrix Create a weight matrix parameter for w_in or w_rec given a shape.

            :param weight_matrix: initial matrix values. init_func runs to fill the matrix if None.
            :type weight_matrix: Optional[FloatVector]
            :param shape: A tuple (or list) specifying the permitted shape of the attribute. If not provided, the shape of the concrete initialisation data will be used as the attribute shape.
            :type shape: Tuple[int]
            :return: a trainable weight matrix
            :rtype: JP_ndarray

            [] TODO: More realistic, sparse weight matrix initialization. Make multiple connections possible in the initialization with non-uniform selection
            """
            weight_init = lambda s: rand.randint(
                rand.split(self._rng_key)[0],
                shape=shape,
                minval=0,
                maxval=2,
            )

            # Values between 0,64
            weight_matrix: JP_ndarray = Parameter(
                weight_matrix,
                family="weight",
                init_func=weight_init,
                shape=shape,
            )

            return weight_matrix

        if w_in is not None:
            w_in = np.array(w_in, dtype=np.float32)

        # Feed forward Mode
        if len(self.shape) == 1:
            # - Feed-forward mode
            if w_rec is not None:
                raise ValueError(
                    "If `shape` is unidimensional, then `w_rec` may not be provided as an argument."
                )

            w_rec = np.zeros((self.size_out, self.size_out, 4))

        # Recurrent mode
        else:
            if len(self.shape) > 2:
                raise ValueError(
                    "`shape` can not specify more than two dimensions (Nin, Nrec)."
                )

            if w_rec is not None:
                w_rec = np.array(w_rec, dtype=np.float32)

            w_rec = get_weight_matrix(w_rec, (self.size_out, self.size_out, 4))

        w_in = get_weight_matrix(w_in, (self.size_in, self.size_out, 4))

        return w_in, w_rec

    ## --- HIGH LEVEL TIME CONSTANTS -- ##

    @property
    def tau_mem(self) -> JP_ndarray:
        """
        tau_mem holds an array of time constants in seconds for neurons with shape = (Nrec,)
        """
        return self.f_tau_mem / self.Itau_mem

    @property
    def tau_syn(self) -> JP_ndarray:
        """
        tau_syn holds an array of time constants in seconds for each synapse of the neurons with shape = (Nrec,5)
        There are tau_ahp, tau_nmda, tau_ampa, tau_gaba_a, and tau_gaba_b methods as well to fetch the time constants of the exact synapse
        """
        return self.f_tau_syn / self.Itau_syn

    @property
    def tau_gaba_b(self) -> JP_ndarray:
        """
        tau_gaba_b holds an array of time constants in seconds for GABA_B synapse of the neurons with shape = (Nrec,)
        """
        return self.f_tau_syn[self.SYN["GABA_B"]] / self.Itau_syn[self.SYN["GABA_B"]]

    @property
    def tau_gaba_a(self) -> JP_ndarray:
        """
        tau_gaba_a holds an array of time constants in seconds for GABA_A synapse of the neurons with shape = (Nrec,)
        """
        return self.f_tau_syn[self.SYN["GABA_A"]] / self.Itau_syn[self.SYN["GABA_A"]]

    @property
    def tau_nmda(self) -> JP_ndarray:
        """
        tau_nmda holds an array of time constants in seconds for NMDA synapse of the neurons with shape = (Nrec,)
        """
        return self.f_tau_syn[self.SYN["NMDA"]] / self.Itau_syn[self.SYN["NMDA"]]

    @property
    def tau_ampa(self) -> JP_ndarray:
        """
        tau_ampa holds an array of time constants in seconds for AMPA synapse of the neurons with shape = (Nrec,)
        """
        return self.f_tau_syn[self.SYN["AMPA"]] / self.Itau_syn[self.SYN["AMPA"]]

    @property
    def tau_ahp(self) -> JP_ndarray:
        """
        tau_ahp holds an array of time constants in seconds for AHP synapse of the neurons with shape = (Nrec,)
        """
        return self.f_tau_syn[self.SYN["AHP"]] / self.Itau_syn[self.SYN["AHP"]]

    ## --- MID-LEVEL HIDDEN BIAS CURRENTS (JAX) -- ##

    @property
    def Ith_mem(self) -> JP_ndarray:
        """
        Ith_mem create an array of membrane threshold(a.k.a gain) currents with shape = (Nrec,)
        """
        return self.Itau_mem * self.f_gain_mem

    @property
    def Ith_syn(self) -> JP_ndarray:
        """
        Ith_syn create an array of synaptic threshold(a.k.a gain) currents in the order of [GABA_B, GABA_A, NMDA, AMPA, AHP] with shape = (5,Nrec)
        """
        return self.Itau_syn * self.f_gain_syn
