"""
Low level DynapSE simulator.
Solves the characteristic equations to simulate the circuits.
Trainable parameters

renamed: dynapse1_neuron_synapse_jax.py -> adexplif_jax.py @ 211206
renamed: adexplif_jax.py -> dynapsim.py @ 220502

References:
[1] E. Chicca, F. Stefanini, C. Bartolozzi and G. Indiveri,
    "Neuromorphic Electronic Circuits for Building Autonomous Cognitive Systems,"
    in Proceedings of the IEEE, vol. 102, no. 9, pp. 1367-1388, Sept. 2014,
    doi: 10.1109/JPROC.2014.2313954.

[2] C. Bartolozzi and G. Indiveri, “Synaptic dynamics in analog vlsi,” Neural
    Comput., vol. 19, no. 10, p. 2581–2603, Oct. 2007. [Online]. Available:
    https://doi.org/10.1162/neco.2007.19.10.2581

[3] P. Livi and G. Indiveri, “A current-mode conductance-based silicon neuron for
    address-event neuromorphic systems,” in 2009 IEEE International Symposium on
    Circuits and Systems, May 2009, pp. 2898–2901

[4] Dynap-SE1 Neuromorphic Chip Simulator for NICE Workshop 2021
    https://code.ini.uzh.ch/yigit/NICE-workshop-2021

[5] Course: Neurormophic Engineering 1
    Tobi Delbruck, Shih-Chii Liu, Giacomo Indiveri
    https://tube.switch.ch/channels/88df64b6

[6] Course: 21FS INI508 Neuromorphic Intelligence
    Giacomo Indiveri
    https://tube.switch.ch/switchcast/uzh.ch/series/5ee1d666-25d2-4c4d-aeb9-4b754b880345?order=newest-first


Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
13/07/2021
[] TODO : Optional mismatch
[] TODO : Deal with SYN
[] TODO : from specifications
[] TODO : Current initialization
[] TODO : Remove simconfig
[] TODO : Length check simconfig
[] TODO : Check LIFjax dfor core.Tracer and all the other things
[] TODO : max spikes per dt
[] TODO : bias : DynapSimCurrents
[] TODO : handle device arrays
"""

from __future__ import annotations
import logging

from typing import Any, Callable, Dict, Optional, Tuple, Union

from jax import random as rand
from jax import custom_gradient
from jax.lax import scan
from jax.tree_util import Partial

from jax import numpy as jnp
import numpy as np
from rockpool.devices.dynapse.config.board import DynapSimConfig
from rockpool.devices.dynapse.config.simcore import (
    DynapSimCurrents,
    DynapSimLayout,
    DynapSimTime,
)

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter, State, SimulationParameter

from rockpool.devices.dynapse.infrastructure.mismatch import MismatchDevice
from rockpool.devices.dynapse.config.simconfig import DynapSE1SimBoard
from rockpool.devices.dynapse.base import DynapSE, DynapSERecord, DynapSEState

Dynapse1Configuration = Any
Dynapse2Configuration = Any


@custom_gradient
def step_pwl(
    imem: jnp.DeviceArray, Ispkthr: jnp.DeviceArray, Ireset: jnp.DeviceArray
) -> Tuple[jnp.DeviceArray, Callable[[jnp.DeviceArray], jnp.DeviceArray]]:
    """
    step_pwl implements heaviside step function with piece-wise linear derivative to use as spike-generation surrogate

    :param imem: Input current to be compared for firing
    :type imem: jnp.DeviceArray
    :param Ispkthr: Spiking threshold current in Amperes
    :type Ispkthr: jnp.DeviceArray
    :param Ireset: Reset current after spike generation in Amperes
    :type Ireset: jnp.DeviceArray
    :return: spikes, grad_func
        spike: generated spike output values
        grad_func:gradient function
    :rtype: Tuple[jnp.DeviceArray, Callable[[jnp.DeviceArray], jnp.DeviceArray]]
    """

    spikes = jnp.clip(jnp.floor(imem - Ispkthr) + 1.0, 0.0)
    grad_func = lambda g: (g * (imem > Ireset) * (Ispkthr - Ireset), 0.0, 0.0)
    return spikes, grad_func


class DynapSim(JaxModule, DynapSE):
    """
    DynapSim solves dynamical chip equations for the DPI neuron and synapse models.
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

    :param shape: Either a single dimension ``N``, which defines a feed-forward layer of DynapSE AdExpIF neurons, or two dimensions ``(N, N)``, which defines a recurrent layer of DynapSE AdExpIF neurons.
    :type shape: Optional[Tuple[int]], optional
    :param sim_config: Dynap-SE1 bias currents and simulation configuration parameters, defaults to None
    :type sim_config: Optional[Union[DynapSE1SimCore, DynapSE1SimBoard]], optional
    :param has_rec: When ``True`` the module provides a trainable recurrent weight matrix. ``False``, module is feed-forward, defaults to True
    :type has_rec: bool, optional
    :param w_rec: If the module is initialised in recurrent mode, one can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(Nrec, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
    :type w_rec: Optional[jnp.DeviceArray], optional

        Let's say 5 neurons allocated

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
            GABA_B: 2 from n2 to n0
            GABA_A: 1 from n0 to n4
            NMDA : 1 from n4 to n4
            AMPA : 1 from n0 to n1, 1 from n2 to n3, 1 from n3 to n2

    :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
    :type dt: float, optional
    :param rng_key: The Jax RNG seed to use on initialisation. By default, a new seed is generated, defaults to None
    :type rng_key: Optional[Any], optional
    :param spiking_input: Whether this module receives spiking input, defaults to True
    :type spiking_input: bool, optional
    :param spiking_output: Whether this module produces spiking output, defaults to True
    :type spiking_output: bool, optional

    :Instance Variables:

    :ivar SYN: A dictionary storing default indexes(order) of the synapse types
    :type SYN: Dict[str, int]
    :ivar isyn: 2D array of synapse currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA] with shape (4,Nrec)
    :type isyn: jnp.DeviceArray
    :ivar iahp: Array of spike frequency adaptation currents of the neurons with shape (Nrec,)
    :type iahp: jnp.DeviceArray
    :ivar imem: Array of membrane currents of the neurons with shape (Nrec,)
    :type imem: jnp.DeviceArray
    :ivar spikes: Logical spiking raster for each neuron at the last simulation time-step with shape (Nrec,)
    :type spikes: jnp.DeviceArray
    :ivar timer_ref: timer to keep the time from the spike generation until the refractory period ends
    :type timer_ref: int
    :ivar Itau_syn: 2D array of synapse leakage currents of the neurons in the order of [GABA_B, GABA_A, NMDA, AMPA] with shape (4,Nrec)
    :type Itau_syn: jnp.DeviceArray
    :ivar Itau_ahp: Array of spike frequency adaptation leakage currents of the neurons with shape (Nrec,)
    :type Itau_ahp: jnp.DeviceArray
    :ivar Itau_soma: Array of membrane leakage currents of the neurons with shape (Nrec,)
    :type Itau_soma: jnp.DeviceArray
    :ivar Iw_ahp: Array of spike frequency adaptation weight currents of the neurons with shape (Nrec,)
    :type Iw_ahp: jnp.DeviceArray
    :ivar Idc: Array of constant DC current in Amperes, injected to membrane with shape (Nrec,)
    :type Idc: jnp.DeviceArray
    :ivar If_nmda: Array of the NMDA gate current in Amperes setting the NMDA gating voltage. If :math:`V_{mem} > V_{nmda}` : The :math:`I_{syn_{NMDA}}` current is added up to the input current, else it cannot with shape (Nrec,)
    :type If_nmda: jnp.DeviceArray
    :ivar Iref: Array of the bias current setting the refractory period `t_ref` with shape (Nrec,)
    :type Iref: jnp.DeviceArray
    :ivar Ipulse: Array of  the bias current setting the pulse width `t_pulse` with shape (Nrec,)
    :type Ipulse: jnp.DeviceArray
    :ivar Ispkthr: Array of spiking threshold current with shape (Nrec,)
    :type Ispkthr: jnp.DeviceArray
    :ivar kappa: Array of mean subthreshold slope factor of the transistors with shape (Nrec,)
    :type kappa: jnp.DeviceArray
    :ivar Ut: Array of thermal voltage in Volts with shape (Nrec,)
    :type Ut: jnp.DeviceArray
    :ivar Io: Array of Dark current in Amperes that flows through the transistors even at the idle state with shape (Nrec,)
    :type Io: jnp.DeviceArray
    :ivar _attr_list: A list of names of attributes imported from simulation configuration object
    :type _attr_list: List[str]


    [] TODO: Find a better parameteric way of weight initialization
    [] TODO: parametric fill rate, different initialization methods
    [] TODO: all neurons cannot have the same parameters ideally however, they experience different parameters in practice because of device mismatch
    [] TODO: Provides mismatch simulation (as second step)
        -As a utility function that operates on a set of parameters?
        -As a method on the class?
    """

    __doc__ += "\nJaxModule" + JaxModule.__doc__

    def __init__(
        self,
        shape: Optional[Tuple[int]] = None,
        Idc: Optional[np.ndarray] = None,
        If_nmda: Optional[np.ndarray] = None,
        Igain_ahp: Optional[np.ndarray] = None,
        Igain_ampa: Optional[np.ndarray] = None,
        Igain_gaba: Optional[np.ndarray] = None,
        Igain_nmda: Optional[np.ndarray] = None,
        Igain_shunt: Optional[np.ndarray] = None,
        Igain_soma: Optional[np.ndarray] = None,
        Ipulse_ahp: Optional[np.ndarray] = None,
        Ipulse: Optional[np.ndarray] = None,
        Iref: Optional[np.ndarray] = None,
        Ispkthr: Optional[np.ndarray] = None,
        Itau_ahp: Optional[np.ndarray] = None,
        Itau_ampa: Optional[np.ndarray] = None,
        Itau_gaba: Optional[np.ndarray] = None,
        Itau_nmda: Optional[np.ndarray] = None,
        Itau_shunt: Optional[np.ndarray] = None,
        Itau_soma: Optional[np.ndarray] = None,
        Iw_ahp: Optional[np.ndarray] = None,
        C_ahp: Optional[np.ndarray] = None,
        C_ampa: Optional[np.ndarray] = None,
        C_gaba: Optional[np.ndarray] = None,
        C_nmda: Optional[np.ndarray] = None,
        C_pulse_ahp: Optional[np.ndarray] = None,
        C_pulse: Optional[np.ndarray] = None,
        C_ref: Optional[np.ndarray] = None,
        C_shunt: Optional[np.ndarray] = None,
        C_soma: Optional[np.ndarray] = None,
        Io: Optional[np.ndarray] = None,
        kappa_n: Optional[np.ndarray] = None,
        kappa_p: Optional[np.ndarray] = None,
        Ut: Optional[np.ndarray] = None,
        Vth: Optional[np.ndarray] = None,
        w_rec: Optional[jnp.DeviceArray] = None,
        has_rec: bool = True,
        weight_init_func: Optional[Callable[[Tuple], np.ndarray]] = None,
        dt: float = 1e-3,
        rng_key: Optional[Any] = None,
        spiking_input: bool = False,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ Initialize ``DynapSim`` module. Parameters are explained in the class docstring.
        """
        self._shape_check(shape)

        super(DynapSim, self).__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(np.random.randint(0, 2 ** 63))

        self.SYN = dict(zip(self.syn_types, range(len(self.syn_types))))

        # --- Parameters & States --- #
        init_current = lambda s: jnp.full(tuple(reversed(s)), Io).T
        # [] TODO : parametrize!
        init_weight = lambda s: w_rec

        # --- States --- #
        self.iahp = State(shape=(self.size_out,), init_func=init_current)
        self.iampa = State(shape=(self.size_out,), init_func=init_current)
        self.igaba = State(shape=(self.size_out,), init_func=init_current)
        self.imem = State(shape=(self.size_out,), init_func=init_current)
        self.inmda = State(shape=(self.size_out,), init_func=init_current)
        self.ishunt = State(shape=(self.size_out,), init_func=init_current)
        self.rng_key = State(rng_key, init_func=lambda _: rng_key)
        self.spikes = State(shape=(self.size_out,), init_func=jnp.zeros)
        self.timer_ref = State(shape=(self.size_out,), init_func=jnp.zeros)
        self.vmem = State(shape=(self.size_out,), init_func=jnp.zeros)

        # --- Parameters --- #
        # [] TODO : Change
        ## Weights
        self.__weight_init(has_rec, init_weight, w_rec)

        self.Idc = Parameter(Idc, shape=(self.size_out,))
        self.If_nmda = Parameter(If_nmda, shape=(self.size_out,))
        self.Igain_ahp = Parameter(Igain_ahp, shape=(self.size_out,))
        self.Igain_ampa = Parameter(Igain_ampa, shape=(self.size_out,))
        self.Igain_gaba = Parameter(Igain_gaba, shape=(self.size_out,))
        self.Igain_nmda = Parameter(Igain_nmda, shape=(self.size_out,))
        self.Igain_shunt = Parameter(Igain_shunt, shape=(self.size_out,))
        self.Igain_soma = Parameter(Igain_soma, shape=(self.size_out,))
        self.Ipulse_ahp = Parameter(Ipulse_ahp, shape=(self.size_out,))
        self.Ipulse = Parameter(Ipulse, shape=(self.size_out,))
        self.Iref = Parameter(Iref, shape=(self.size_out,))
        self.Ispkthr = Parameter(Ispkthr, shape=(self.size_out,))
        self.Itau_ahp = Parameter(Itau_ahp, shape=(self.size_out,))
        self.Itau_ampa = Parameter(Itau_ampa, shape=(self.size_out,))
        self.Itau_gaba = Parameter(Itau_gaba, shape=(self.size_out,))
        self.Itau_nmda = Parameter(Itau_nmda, shape=(self.size_out,))
        self.Itau_shunt = Parameter(Itau_shunt, shape=(self.size_out,))
        self.Itau_soma = Parameter(Itau_soma, shape=(self.size_out,))
        self.Iw_ahp = Parameter(Iw_ahp, shape=(self.size_out,))

        # --- Simulation Parameters --- #
        self.dt = SimulationParameter(dt, shape=())
        self.C_ahp = SimulationParameter(C_ahp, shape=(self.size_out,))
        self.C_ampa = SimulationParameter(C_ampa, shape=(self.size_out,))
        self.C_gaba = SimulationParameter(C_gaba, shape=(self.size_out,))
        self.C_nmda = SimulationParameter(C_nmda, shape=(self.size_out,))
        self.C_pulse_ahp = SimulationParameter(C_pulse_ahp, shape=(self.size_out,))
        self.C_pulse = SimulationParameter(C_pulse, shape=(self.size_out,))
        self.C_ref = SimulationParameter(C_ref, shape=(self.size_out,))
        self.C_shunt = SimulationParameter(C_shunt, shape=(self.size_out,))
        self.C_soma = SimulationParameter(C_soma, shape=(self.size_out,))
        self.Io = SimulationParameter(Io, shape=(self.size_out,))
        self.kappa_n = SimulationParameter(kappa_n, shape=(self.size_out,))
        self.kappa_p = SimulationParameter(kappa_p, shape=(self.size_out,))
        self.Ut = SimulationParameter(Ut, shape=(self.size_out,))
        self.Vth = SimulationParameter(Vth, shape=(self.size_out,))

        self._attr_list = (
            list(DynapSimCurrents.__annotations__.keys())
            + list(DynapSimLayout.__annotations__.keys())
            + [
                "igaba",
                "ishunt",
                "inmda",
                "iampa",
                "imem",
                "iahp",
                "w_rec",
            ]
        )
        self._attr_list.remove("Iw_0")
        self._attr_list.remove("Iw_1")
        self._attr_list.remove("Iw_2")
        self._attr_list.remove("Iw_3")
        attr = {name: self.__getattribute__(name) for name in self._attr_list}
        self.md = MismatchDevice(self.rng_key, **attr)

        # Performance : Use device arrays in calculations
        self._zero = jnp.array(0.0)
        self._one = jnp.array(1.0)
        self._two = jnp.array(2.0)

        # - Define additional arguments required during initialisation
        self._init_args = {
            "has_rec": has_rec,
        }

    @classmethod
    def from_specification(
        cls,
        shape: Optional[Tuple[int]],
        has_rec: bool = True,
        weight_init_func: Optional[Callable[[Tuple], np.ndarray]] = None,
        dt: float = 1e-3,
        rng_key: Optional[Any] = None,
        spiking_input: bool = False,
        spiking_output: bool = True,
        **kwargs,
    ) -> DynapSim:
        simconfig = DynapSimConfig.from_specification(shape[-1], **kwargs)
        _mod = cls.from_DynapSimConfig(
            shape,
            simconfig,
            has_rec,
            weight_init_func,
            dt,
            rng_key,
            spiking_input,
            spiking_output,
        )
        return _mod

    @classmethod
    def from_DynapSimConfig(
        cls,
        shape: Optional[Tuple[int]],
        simconfig: DynapSimConfig,
        has_rec: bool = True,
        weight_init_func: Optional[Callable[[Tuple], np.ndarray]] = None,
        dt: float = 1e-3,
        rng_key: Optional[Any] = None,
        spiking_input: bool = False,
        spiking_output: bool = True,
        **kwargs,
    ) -> DynapSim:

        __constructor = dict.fromkeys(DynapSimLayout.__annotations__.keys())
        __constructor.update(dict.fromkeys(DynapSimCurrents.__annotations__.keys()))
        for key in ["w_rec", "Iw_0", "Iw_1", "Iw_2", "Iw_3"]:
            __constructor.pop(key, None)
        for key in __constructor:
            __constructor[key] = simconfig.__getattribute__(key)

        _mod = cls(
            shape=shape,
            has_rec=has_rec,
            weight_init_func=weight_init_func,
            dt=dt,
            rng_key=rng_key,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            w_rec=simconfig.w_rec,
            **__constructor,
            **kwargs,
        )
        return _mod

    @classmethod
    def from_Dynapse1Configuration(cls, config: Dynapse1Configuration) -> DynapSim:
        # Parameters
        # Weights
        None

    @classmethod
    def from_Dynapse2Configuration(cls, config: Dynapse2Configuration) -> DynapSim:
        None

    def _shape_check(self, shape: Tuple[int]) -> None:
        """
        _shape_check Controls the shape of module and complains if not appropriate.

        :param shape: Either a single dimension ``N``, which defines a feed-forward layer of DynapSE AdExpIF neurons, or two dimensions ``(N, N)``, which defines a recurrent layer of DynapSE AdExpIF neurons.
        :type shape: Tuple[int]
        :raises ValueError: ``shape`` should be defined (N*4,N,)!
        :raises ValueError: The simulation configuration object size and number of device neruons does not match!
        :raises ValueError: `shape[0]` should be `shape[1]`*4 in the recurrent mode
        """
        # Check the parameters and initialize to default if necessary
        if shape is None or len(shape) != 2:
            raise ValueError(f"shape should be defined (N*4,N,)! shape={shape}")

        # Check the network size and the recurrent weight vector accordingly
        syn_size_check = lambda s: s == (s // 4) * 4  # 4 synapse per neuron for sure

        # Check if input dimension meets the 4 synapse per neuron criteria
        if not syn_size_check(shape[0]):
            raise ValueError(
                f"Input dimension ({shape[0]},..) should have been multiples of 4! (Go for {shape[0]//4}, {(shape[0]+4)//4}, or {shape[0]*4}) \n"
                f"Each neuron holds 4 synaptic state, which means 4 input gates per neuron!\n"
                f"i.e. ({(shape[0]//4)*4},..) means {shape[0]//4} neurons with 4 synapses\n"
                f"i.e. ({((shape[0]+4)//4)*4},..) means {(shape[0]+4)//4} neurons with 4 synapses\n"
                f"i.e. ({shape[0]*4},..) means {shape[0]} neurons with 4 synapses\n"
            )

        # Check if output dimension meets the 4 synapse per neuron criteria
        if shape[1] != shape[0] // 4:
            raise ValueError("`shape[0]` should be `shape[1]`*4")

    def __weight_init(
        self,
        has_rec: bool,
        init_func: Callable[[Tuple], jnp.DeviceArray],
        w_rec: Optional[jnp.DeviceArray] = None,
    ) -> None:
        """
        __weight_init initialize the weight matrix of the module as a `Parameter` object

        :param has_rec: When ``True`` the module provides a trainable recurrent weight matrix. ``False``, module is feed-forward, defaults to True
        :type has_rec: bool, optional
        :param init_func: a function to initilize the weights in the case that w_rec is not explicitly provided
        :type init_func: Callable[[Tuple], jnp.DeviceArray]
        :param w_rec: If the module is initialised in recurrent mode, one can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(Nrec, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``.
        :type w_rec: Optional[jnp.DeviceArray], optional
        :raises ValueError: If ``has_rec`` is False, then `w_rec` may not be provided as an argument or initialized by the module
        """
        # Feed forward Mode
        if not has_rec:
            if w_rec is not None:
                raise ValueError(
                    "If ``has_rec`` is False, then `w_rec` may not be provided as an argument or initialized by the module."
                )

            # In order not to make the jax complain about w_rec
            self.w_rec = jnp.zeros((self.size_out, self.size_in // 4, 4))
            logging.info(
                f"Feed forward module allocates {self.size_out} neurons with 4 synapses"
            )

        # Recurrent mode
        else:
            if w_rec is not None:
                w_rec = jnp.array(w_rec, dtype=jnp.float32)

            # Values between 0,64
            self.w_rec = Parameter(
                w_rec,
                family="weights",
                init_func=init_func,
                shape=(self.size_out, self.size_in // 4, 4),
            )
            logging.info(
                f"Recurrent module allocates {self.size_out} neurons with 4 synapses"
            )

    def evolve(
        self, input_data: jnp.DeviceArray, record: bool = True
    ) -> Tuple[jnp.DeviceArray, Dict[str, jnp.DeviceArray], Dict[str, jnp.DeviceArray]]:
        """
        evolve implements raw rockpool JAX evolution function for a DynapSim module.
        The function solves the dynamical equations introduced at the ``DynapSim`` module definition

        :param input_data: Input array of shape ``(T, Nrec, 4)`` to evolve over. Represents number of spikes at that timebin for different synaptic gates
        :type input_data: jnp.DeviceArray
        :param record: record the each timestep of evolution or not, defaults to False
        :type record: bool, optional
        :return: spikes_ts, states, record_dict
            :spikes_ts: is an array with shape ``(T, Nrec)`` containing the output data(spike raster) produced by the module.
            :states: is a dictionary containing the updated module state following evolution.
            :record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True`` else empty dictionary {}
        :rtype: Tuple[jnp.DeviceArray, Dict[str, jnp.DeviceArray], Dict[str, jnp.DeviceArray]]
        """

        kappa = (self.md.kappa_n + self.md.kappa_p) / 2

        # --- Time constant computation utils --- #
        __pw = lambda ipw, C: (self.md.Vth * C) / ipw
        __tau = lambda itau, C: ((self.md.Ut / kappa) * C) / itau

        tau_ahp = lambda itau: __tau(itau, self.md.C_ahp)
        tau_soma = lambda itau: __tau(itau, self.md.C_soma)
        tau_syn = lambda itau: __tau(itau, self.__syn_stack(self.md, "C"))

        # --- Stateless Parameters --- #
        t_ref = __pw(self.md.Iref, self.md.C_ref)
        t_pulse = __pw(self.md.Ipulse, self.md.C_pulse)
        t_pulse_ahp = __pw(self.md.Ipulse_ahp, self.md.C_pulse_ahp)

        ## --- Synapse Stack --- ## Nrec, 4
        Itau_syn_clip = jnp.clip(self.__syn_stack(self.md, "Itau").T, self.md.Io).T
        Igain_syn_clip = jnp.clip(self.__syn_stack(self.md, "Igain").T, self.md.Io).T

        ## --- Spike frequency adaptation --- ## Nrec
        Itau_ahp_clip = jnp.clip(self.md.Itau_ahp, self.md.Io)
        Igain_ahp_clip = jnp.clip(self.md.Igain_ahp, self.md.Io)

        ## -- Membrane -- ## Nrec
        Itau_soma_clip = jnp.clip(self.md.Itau_soma, self.md.Io)
        Igain_soma_clip = jnp.clip(self.md.Igain_soma, self.md.Io)

        # Both (T, Nrecx4), and (T, Nrec, 4) shaped inputs are accepted
        input_data = jnp.reshape(input_data, (input_data.shape[0], -1, 4))

        def forward(
            state: DynapSEState, Iw_input: jnp.DeviceArray
        ) -> Tuple[DynapSEState, DynapSERecord]:
            """
            forward implements single time-step neuron and synapse dynamics

            :param state: (igaba, ishunt, inmda, iampa, iahp, imem, vmem, spikes, timer_ref, key)
                igaba: GABA synapse currents of each neuron [Nrec]
                ishunt: SHUNT synapse currents of each neuron [Nrec]
                inmda: NMDA synapse currents of each neuron [Nrec]
                iampa: AMPA synapse currents of each neuron [Nrec]
                iahp: Spike frequency adaptation currents of each neuron [Nrec]
                imem: Membrane currents of each neuron [Nrec]
                vmem: Membrane voltages of each neuron [Nrec]
                spikes: Logical spike raster for each neuron [Nrec]
                timer_ref: Refractory timer of each neruon [Nrec]
                key: The Jax RNG seed to be used for mismatch simulation
            :type state: DynapSEState
            :param Iw_input: external weighted current matrix generated via input spikes [Nrec, 4]
            :type Iw_input: jnp.DeviceArray
            :return: state, record
                state: Updated state at end of the forward steps
                record: Updated record instance to including spikes, igaba, ishunt, inmda, iampa, iahp, imem, and vmem states
            :rtype: Tuple[DynapSEState, DynapSERecord]
            """
            # [] TODO : Would you allow currents to go below Io or not?!!!!

            (
                igaba,
                ishunt,
                inmda,
                iampa,
                iahp,
                imem,
                vmem,
                spikes,
                timer_ref,
                rng_key,
            ) = state

            isyn = jnp.stack([igaba, ishunt, inmda, iampa]).T
            # ---------------------------------- #
            # --- Forward step: DPI SYNAPSES --- #
            # ---------------------------------- #

            ## Real time weight is 0 if no spike, w_rec if spike event occurs
            Iws_internal = jnp.dot(self.md.w_rec.T, spikes).T
            Iws = Iws_internal + Iw_input

            # isyn_inf is the current that a synapse current would reach with a sufficiently long pulse
            isyn_inf = ((Igain_syn_clip / Itau_syn_clip) * Iws) - Igain_syn_clip
            isyn_inf = jnp.clip(isyn_inf, self._zero)

            # synaptic time constant is practically much more longer than expected when isyn << Igain
            tau_syn_prime = tau_syn(Itau_syn_clip) * (1 + (Igain_syn_clip / isyn))

            ## Exponential charge, discharge positive feedback factor arrays
            f_charge = self._one - jnp.exp(-t_pulse / tau_syn_prime.T).T  # Nrecx4
            f_discharge = jnp.exp(-self.dt / tau_syn_prime)  # Nrecx4

            ## DISCHARGE in any case
            isyn = f_discharge * isyn

            ## CHARGE if spike occurs -- UNDERSAMPLED -- dt >> t_pulse
            isyn += f_charge * isyn_inf
            isyn = jnp.clip(isyn.T, self.md.Io).T  # Nrecx4

            # ---------------------------------- #
            # ---------------------------------- #
            # ---------------------------------- #

            # ------------------------------------------------------ #
            # --- Forward step: AHP : Spike Frequency Adaptation --- #
            # ------------------------------------------------------ #

            Iws_ahp = self.md.Iw_ahp * spikes  # 0 if no spike, Iw_ahp if spike
            iahp_inf = ((Igain_ahp_clip / Itau_ahp_clip) * Iws_ahp) - Igain_ahp_clip

            tau_ahp_prime = tau_ahp(Itau_ahp_clip) * (
                self._one + (Igain_ahp_clip / iahp)
            )

            # Calculate charge and discharge factors
            f_charge_ahp = self._one - jnp.exp(-t_pulse_ahp / tau_ahp_prime)  # Nrec
            f_discharge_ahp = jnp.exp(-self.dt / tau_ahp_prime)  # Nrec

            ## DISCHARGE in any case
            iahp = f_discharge_ahp * iahp

            ## CHARGE if spike occurs -- UNDERSAMPLED -- dt >> t_pulse
            iahp += f_charge_ahp * iahp_inf
            iahp = jnp.clip(iahp, self.md.Io)  # Nrec

            # ------------------------------------------------------ #
            # ------------------------------------------------------ #
            # ------------------------------------------------------ #

            # ------------------------------ #
            # --- Forward step: MEMBRANE --- #
            # ------------------------------ #
            _kappa_2 = jnp.power(kappa, self._two)
            _kappa_prime = _kappa_2 / (kappa + self._one)
            f_feedback = jnp.exp(_kappa_prime * (vmem / self.md.Ut))  # 4xNrec

            ## Decouple synaptic currents and calculate membrane input
            igaba, ishunt, inmda, iampa = isyn.T

            # inmda = 0 if vmem < Vth_nmda else inmda
            I_nmda_dp = inmda / (self._one + self.md.If_nmda / imem)

            # Iin = 0 if the neuron is in the refractory period
            Iin = I_nmda_dp + iampa - igaba + self.md.Idc
            Iin *= jnp.logical_not(timer_ref.astype(bool)).astype(jnp.float32)
            Iin = jnp.clip(Iin, self.md.Io)

            # ishunt (shunting) contributes to the membrane leak instead of subtracting from Iin
            Ileak = Itau_soma_clip + ishunt
            tau_mem_prime = tau_soma(Ileak) * (self._one + (Igain_soma_clip / imem))

            ## Steady state current
            imem_inf = (Igain_soma_clip / Itau_soma_clip) * (Iin - iahp - Ileak)

            ## Positive feedback
            Ifb = self.md.Io * f_feedback
            f_imem = ((Ifb) / (Ileak)) * (imem + Igain_soma_clip)

            ## Forward Euler Update
            del_imem = (self._one / tau_mem_prime) * (
                imem_inf + f_imem - (imem * (self._one + (iahp / Ileak)))
            )
            imem = imem + del_imem * self.dt
            imem = jnp.clip(imem, self.md.Io)

            ## Membrane Potential
            vmem = (self.md.Ut / kappa) * jnp.log(imem / self.md.Io)

            # ------------------------------ #
            # ------------------------------ #
            # ------------------------------ #

            # ------------------------------ #
            # --- Spike Generation Logic --- #
            # ------------------------------ #

            ## Detect next spikes (with custom gradient)
            spikes = step_pwl(imem, self.md.Ispkthr, self.md.Io)

            ## Reset imem depending on spiking activity
            imem = (self._one - spikes) * imem + spikes * self.md.Io

            ## Set the refractrory timer
            timer_ref -= self.dt
            timer_ref = jnp.clip(timer_ref, self._zero)
            timer_ref = (self._one - spikes) * timer_ref + spikes * t_ref

            # ------------------------------ #
            # ------------------------------ #
            # ------------------------------ #

            # igaba, ishunt, inmda, iampa = isyn

            state = (
                igaba,
                ishunt,
                inmda,
                iampa,
                iahp,
                imem,
                vmem,
                spikes,
                timer_ref,
                rng_key,
            )
            record = (
                igaba,
                ishunt,
                inmda,
                iampa,
                iahp,
                imem,
                vmem,
                spikes,
            )
            return state, record

        # --- Evolve over spiking inputs --- #
        state, (
            igaba_ts,
            ishunt_ts,
            inmda_ts,
            iampa_ts,
            iahp_ts,
            imem_ts,
            vmem_ts,
            spikes_ts,
        ) = scan(
            forward,
            (
                self.md.igaba,
                self.md.ishunt,
                self.md.inmda,
                self.md.iampa,
                self.md.iahp,
                self.md.imem,
                self.vmem,
                self.spikes,
                self.timer_ref,
                self.rng_key,
            ),
            input_data,
        )

        # --- RETURN ARGUMENTS --- #

        ## the state returned should be in the same shape with the state dictionary given
        states = {
            "igaba": state[0],
            "ishunt": state[1],
            "inmda": state[2],
            "iampa": state[3],
            "iahp": state[4],
            "imem": state[5],
            "vmem": state[6],
            "spikes": state[7],
            "timer_ref": state[8],
            "rng_key": state[9],
        }

        record_dict = {}
        if record:
            record_dict = {
                "igaba": igaba_ts,
                "ishunt": ishunt_ts,
                "inmda": inmda_ts,
                "iampa": iampa_ts,
                "iahp": iahp_ts,
                "imem": imem_ts,
                "vmem": vmem_ts,
                "spikes": spikes_ts,
            }

        return spikes_ts, states, record_dict

    def export_Dynapse1Configuration(self) -> Dynapse1Configuration:
        None

    def export_Dynapse2Configuration(self) -> Dynapse2Configuration:
        None

    @staticmethod
    def __syn_stack(obj: object, name: str) -> jnp.DeviceArray:
        """
        __syn_stack fetches attributes from the synaptic gate subcircuits in [GABA, SHUNT, NMDA, AMPA] order and create a property array covering all the neurons allocated

        :param obj: the object to fectch the currents parameters
        :type obj: object
        :param name: the base name of the parameter
        :type name: str
        :return: an array full of the values of the target attributes of all 4 synapses (`size`, 4)
        :rtype: jnp.DeviceArray
        """
        return jnp.stack(
            [
                obj.__getattribute__(f"{name}_gaba"),
                obj.__getattribute__(f"{name}_shunt"),
                obj.__getattribute__(f"{name}_nmda"),
                obj.__getattribute__(f"{name}_ampa"),
            ]
        ).T
