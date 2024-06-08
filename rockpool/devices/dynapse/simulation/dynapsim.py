"""
Low level DynapSE-2 simulator neuron model implementation
Solves the characteristic equations to simulate the circuits with ptimizable parameters

References

[1] E. Chicca, F. Stefanini, C. Bartolozzi and G. Indiveri,
    "Neuromorphic Electronic Circuits for Building Autonomous Cognitive Systems,"
    in Proceedings of the IEEE, vol. 102, no. 9, pp. 1367-1388, Sept. 2014,
    doi: 10.1109/JPROC.2014.2313954.

[2] C. Bartolozzi and G. Indiveri, “Synaptic dynamics in analog vlsi,” Neural
    Comput., vol. 19, no. 10, p. 2581-2603, Oct. 2007. [Online]. Available:
    https://doi.org/10.1162/neco.2007.19.10.2581

[3] P. Livi and G. Indiveri, “A current-mode conductance-based silicon neuron for
    address-event neuromorphic systems,” in 2009 IEEE International Symposium on
    Circuits and Systems, May 2009, pp. 2898-2901

[4] Dynap-SE1 Neuromorphic Chip Simulator for NICE Workshop 2021
    https://code.ini.uzh.ch/yigit/NICE-workshop-2021

[5] Course: Neurormophic Engineering 1
    Tobi Delbruck, Shih-Chii Liu, Giacomo Indiveri
    https://tube.switch.ch/channels/88df64b6

[6] Course: 21FS INI508 Neuromorphic Intelligence
    Giacomo Indiveri
    https://tube.switch.ch/switchcast/uzh.ch/series/5ee1d666-25d2-4c4d-aeb9-4b754b880345?order=newest-first
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import jax
from jax import random as rand
from jax.lax import scan
from jax.tree_util import Partial
from jax import numpy as jnp

import sys
import numpy as np

from rockpool.devices.dynapse.lookup import (
    default_layout,
    default_weights,
    default_currents,
)
from rockpool.devices.dynapse.typehints import DynapSimRecord, DynapSimState
from rockpool.devices.dynapse.mapping import DynapseNeurons

from rockpool.typehints import FloatVector
from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.nn.modules.native.linear import kaiming
from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool.graph import GraphHolder, LinearWeights, as_GraphHolder
from rockpool.transform.mismatch import mismatch_generator

from .surrogate import step_pwl
from .mismatch_prototype import frozen_mismatch_prototype

__all__ = ["DynapSim"]


class DynapSim(JaxModule):
    """
    DynapSim solves dynamical chip equations for the DPI neuron and synapse models.
    Receives configuration as bias currents and solves membrane and synapse dynamics using ``jax`` backend.
    One block has

    * 1 synapse receiving spikes from the other circuits
    * 1 recurrent synapse for spike frequency adaptation (**AHP**)
    * 1 membrane evaluating the state and deciding fire or not

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

        dI_{mem} &= \\dfrac{I_{mem}}{\\tau \\left( I_{mem} + I_{th} \\right) } \\cdot \\left( I_{mem_{\\infty}} + f(I_{mem}) - I_{mem} \\left( 1 + \\dfrac{I_{ahp}}{I_{\\tau}} \\right) \\right) \\cdot dt \\\\\\\\
        I_{mem}(t_1) &= I_{mem}(t_0) + dI_{mem}

    Where

    .. math ::

        I_{mem_{\\infty}} &= \\dfrac{I_{th}}{I_{\\tau}} \\left( I_{in} - I_{ahp} - I_{\\tau}\\right) \\\\\\\\
        f(I_{mem}) &= \\dfrac{I_{a}}{I_{\\tau}} \\left(I_{mem} + I_{th} \\right ) \\\\\\\\
        I_{a} &= \\dfrac{I_{a_{gain}}}{1+ exp\\left(-\\dfrac{I_{mem}+I_{a_{th}}}{I_{a_{norm}}}\\right)} \\\\\\\\

    :On spiking:

    When the membrane potential for neuron :math:`j`, :math:`I_{mem, j}` exceeds the threshold current :math:`I_{spkthr}`, then the neuron emits a spike.

    .. math ::

        I_{mem, j} > I_{spkthr} \\rightarrow S_{j} &= 1 \\\\
        I_{mem, j} &= I_{reset} \\\\

    .. seealso ::
        For detailed explanations of the equations and the usage

        :ref:`/devices/DynapSE/neuron-model.ipynb` 

    """

    def __init__(
        self,
        shape: Union[Tuple[int], int],
        Idc: FloatVector = default_currents["Idc"],
        If_nmda: FloatVector = default_currents["If_nmda"],
        Igain_ahp: FloatVector = default_currents["Igain_ahp"],
        Igain_mem: FloatVector = default_currents["Igain_mem"],
        Igain_syn: FloatVector = default_currents["Igain_ampa"],
        Ipulse_ahp: FloatVector = default_currents["Ipulse_ahp"],
        Ipulse: FloatVector = default_currents["Ipulse"],
        Iref: FloatVector = default_currents["Iref"],
        Ispkthr: FloatVector = default_currents["Ispkthr"],
        Itau_ahp: FloatVector = default_currents["Itau_ahp"],
        Itau_mem: FloatVector = default_currents["Itau_mem"],
        Itau_syn: FloatVector = default_currents["Itau_ampa"],
        Iw_ahp: FloatVector = default_currents["Iw_ahp"],
        C_ahp: FloatVector = default_layout["C_ahp"],
        C_syn: FloatVector = default_layout["C_ampa"],
        C_pulse_ahp: FloatVector = default_layout["C_pulse_ahp"],
        C_pulse: FloatVector = default_layout["C_pulse"],
        C_ref: FloatVector = default_layout["C_ref"],
        C_mem: FloatVector = default_layout["C_mem"],
        Io: FloatVector = default_layout["Io"],
        kappa_n: FloatVector = default_layout["kappa_n"],
        kappa_p: FloatVector = default_layout["kappa_p"],
        Ut: FloatVector = default_layout["Ut"],
        Vth: FloatVector = default_layout["Vth"],
        Iscale: FloatVector = default_weights["Iscale"],
        w_rec: Optional[FloatVector] = None,
        has_rec: bool = False,
        weight_init_func: Optional[Callable[[Tuple], FloatVector]] = kaiming,
        dt: float = 1e-3,
        percent_mismatch: Optional[float] = None,
        rng_key: Optional[FloatVector] = None,
        spiking_input: bool = False,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ) -> None:
        """
        __init__ constructs a DynapSim object

        :param shape: Either a single dimension ``N``, which defines a feed-forward layer of DynapSE AdExpIF neurons, or two dimensions ``(N, N)``, which defines a recurrent layer of DynapSE AdExpIF neurons.
        :type shape: Tuple[int]
        :param Idc: Constant DC current injected to membrane in Amperes with shape
        :type Idc: FloatVector, optinoal
        :param If_nmda: NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes with shape (Nrec,)
        :type If_nmda: FloatVector, optinoal
        :param Igain_ahp: gain bias current of the spike frequency adaptation block in Amperes with shape (Nrec,)
        :type Igain_ahp: FloatVector, optinoal
        :param Igain_mem: gain bias current for neuron membrane in Amperes with shape (Nrec,)
        :type Igain_mem: FloatVector, optinoal
        :param Igain_syn: gain bias current of synaptic gates (AMPA, GABA, NMDA, SHUNT) combined in Amperes with shape (Nrec,)
        :type Igain_syn: FloatVector, optinoal
        :param Ipulse_ahp: bias current setting the pulse width for spike frequency adaptation block ``t_pulse_ahp`` in Amperes with shape (Nrec,)
        :type Ipulse_ahp: FloatVector, optinoal
        :param Ipulse: bias current setting the pulse width for neuron membrane ``t_pulse`` in Amperes with shape (Nrec,)
        :type Ipulse: FloatVector, optinoal
        :param Iref: bias current setting the refractory period ``t_ref`` in Amperes with shape (Nrec,)
        :type Iref: FloatVector, optinoal
        :param Ispkthr: spiking threshold current, neuron spikes if :math:`I_{mem} > I_{spkthr}` in Amperes with shape (Nrec,)
        :type Ispkthr: FloatVector, optinoal
        :param Itau_ahp: Spike frequency adaptation leakage current setting the time constant ``tau_ahp`` in Amperes with shape (Nrec,)
        :type Itau_ahp: FloatVector, optinoal
        :param Itau_mem: Neuron membrane leakage current setting the time constant ``tau_mem`` in Amperes with shape (Nrec,)
        :type Itau_mem: FloatVector, optinoal
        :param Itau_syn: (AMPA, GABA, NMDA, SHUNT) synapses combined leakage current setting the time constant ``tau_syn`` in Amperes with shape (Nrec,)
        :type Itau_syn: FloatVector, optinoal
        :param Iw_ahp: spike frequency adaptation weight current of the neurons of the core in Amperes with shape (Nrec,)
        :type Iw_ahp: FloatVector, optinoal
        :param C_ahp: AHP synapse capacitance in Farads with shape (Nrec,)
        :type C_ahp: FloatVector, optional
        :param C_syn: synaptic capacitance in Farads with shape (Nrec,)
        :type C_syn: FloatVector, optional
        :param C_pulse_ahp: spike frequency adaptation circuit pulse-width creation sub-circuit capacitance in Farads with shape (Nrec,)
        :type C_pulse_ahp: FloatVector, optional
        :param C_pulse: pulse-width creation sub-circuit capacitance in Farads with shape (Nrec,)
        :type C_pulse: FloatVector, optional
        :param C_ref: refractory period sub-circuit capacitance in Farads with shape (Nrec,)
        :type C_ref: FloatVector, optional
        :param C_mem: neuron membrane capacitance in Farads with shape (Nrec,)
        :type C_mem: FloatVector, optional
        :param Io: Dark current in Amperes that flows through the transistors even at the idle state with shape (Nrec,)
        :type Io: FloatVector, optional
        :param kappa_n: Subthreshold slope factor (n-type transistor) with shape (Nrec,)
        :type kappa_n: FloatVector, optional
        :param kappa_p: Subthreshold slope factor (p-type transistor) with shape (Nrec,)
        :type kappa_p: FloatVector, optional
        :param Ut: Thermal voltage in Volts with shape (Nrec,)
        :type Ut: FloatVector, optional
        :param Vth: The cut-off Vgs potential of the transistors in Volts (not type specific) with shape (Nrec,)
        :type Vth: FloatVector, optional
        :param Iscale: weight scaling current of the neurons of the core in Amperes
        :type Iscale: FloatVector, optinoal
        :param w_rec: If the module is initialised in recurrent mode, one can provide a concrete initialisation for the recurrent weights, which must be a square matrix with shape ``(Nrec, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types. If the model is not initialised in recurrent mode, then you may not provide ``w_rec``, defaults tp None
        :type w_rec: Optional[FloatVector], optional
        :param has_rec: When ``True`` the module provides a trainable recurrent weight matrix. ``False``, module is feed-forward, defaults to True
        :type has_rec: bool, optional
        :param weight_init_func: The initialisation function to use when generating weights, gets the shape and returns the initial weights, defatuls to kaiming
        :type weight_init_func: Optional[Callable[[Tuple], FloatVector]], optional
        :param dt: The time step for the forward-Euler ODE solver, defaults to 1e-3
        :type dt: float, optional
        :param percent_mismatch: Gaussian parameter mismatch percentage (check `transform.mismatch_generator` implementation), defaults to None
        :type percent_mismatch: Optional[float], optional
        :param rng_key: The Jax RNG seed to use on initialisation. By default, a new seed is generated, defaults to None
        :type rng_key: Optional[FloatVector], optional
        :param spiking_input: Whether this module receives spiking input, defaults to True
        :type spiking_input: bool, optional
        :param spiking_output: Whether this module produces spiking output, defaults to True
        :type spiking_output: bool, optional
        :raises ValueError: `shape` must be a one- or two-element tuple `(Nin, Nout)`
        :raises ValueError: Multapses are not currently supported in DynapSim pipeline!
        """

        # - Check shape argument
        if np.size(shape) == 1:
            shape = (np.array(shape).item(), np.array(shape).item())

        if np.size(shape) > 2:
            raise ValueError(
                "`shape` must be a one- or two-element tuple `(Nin, Nout)`."
            )

        super(DynapSim, self).__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        if self.size_in != self.size_out:
            raise ValueError(
                "Multapses are not currently supported in DynapSim pipeline!"
            )

        # - Seed RNG
        if rng_key is None:
            rng_key = rand.PRNGKey(np.random.randint(0, 2**63))

        ### --- States --- ####
        __state = lambda init_func: State(
            init_func=init_func,
            shape=(self.size_out,),
            permit_reshape=False,
            cast_fn=lambda _o: jnp.array(_o, dtype=jnp.float32),
        )

        __Io_state = lambda _: __state(
            lambda s: jnp.full(tuple(reversed(s)), Io, jnp.float32).T
        )
        __zero_state = lambda _: __state(lambda s: jnp.zeros(s, dtype=jnp.float32))

        ## Data
        self.iahp = __Io_state(None)
        """Spike frequency adaptation current states of the neurons in Amperes with shape (Nrec,)"""

        self.iampa = __Io_state(None)
        """Fast excitatory AMPA synapse current states of the neurons in Amperes with shape (Nrec,)"""

        self.igaba = __Io_state(None)
        """Slow inhibitory adaptation current states of the neurons in Amperes with shape (Nrec,)"""

        self.imem = __Io_state(None)
        """Membrane current states of the neurons in Amperes with shape (Nrec,)"""

        self.inmda = __Io_state(None)
        """Slow excitatory synapse current states of the neurons in Amperes with shape (Nrec,)"""

        self.ishunt = __Io_state(None)
        """Fast inhibitory shunting synapse current states of the neurons in Amperes with shape (Nrec,)"""

        self.spikes = __zero_state(None)
        """Logical spiking raster for each neuron at the last simulation time-step with shape (Nrec,)"""

        self.timer_ref = __zero_state(None)
        """timer to keep the time from the spike generation until the refractory period ends"""

        self.vmem = __zero_state(None)
        """Membrane potential states of the neurons in Volts with shape (Nrec,)"""

        ### --- Parameters --- ###
        __parameter = lambda _param: Parameter(
            data=(
                _param
                if isinstance(
                    _param, (np.ndarray, jnp.ndarray, jax.Array, jax.core.Tracer)
                )
                else jnp.full((self.size_out,), _param, dtype=jnp.float32)
            ),
            family="bias",
            shape=(self.size_out,),
            permit_reshape=False,
            cast_fn=lambda _o: jnp.array(_o, dtype=jnp.float32),
        )

        # Special handler for wrec
        if isinstance(has_rec, jax.core.Tracer) or has_rec:
            self.w_rec = Parameter(
                data=w_rec,
                family="weights",
                init_func=weight_init_func,
                shape=(self.size_out, self.size_in),
                permit_reshape=False,
                cast_fn=lambda _o: jnp.array(_o, dtype=jnp.float32),
            )
        else:
            # Do not let it break the pipeline
            self.w_rec = SimulationParameter(
                data=jnp.zeros((self.size_out, self.size_in), dtype=jnp.float32),
                family="weights",
            )

        # --- Simulation Parameters --- #
        __simparam = lambda _param: SimulationParameter(
            data=(
                _param
                if isinstance(
                    _param, (np.ndarray, jnp.ndarray, jax.Array, jax.core.Tracer)
                )
                else jnp.full((self.size_out,), _param)
            ),
            shape=(self.size_out,),
            permit_reshape=False,
            cast_fn=lambda _o: jnp.array(_o, dtype=jnp.float32),
        )

        # -- #
        self.Idc = __simparam(Idc)
        """Constant DC current injected to membrane in Amperes with shape"""

        self.If_nmda = __simparam(If_nmda)
        """NMDA gate soft cut-off current setting the NMDA gating voltage in Amperes with shape (Nrec,)"""

        self.Igain_ahp = __simparam(Igain_ahp)
        """gain bias current of the spike frequency adaptation block in Amperes with shape (Nrec,)"""

        self.Igain_mem = __simparam(Igain_mem)
        """gain bias current for neuron membrane in Amperes with shape (Nrec,)"""

        self.Igain_syn = __simparam(Igain_syn)
        """gain bias current of synaptic gates (AMPA, GABA, NMDA, SHUNT) combined in Amperes with shape (Nrec,)"""

        self.Ipulse_ahp = __simparam(Ipulse_ahp)
        """bias current setting the pulse width for spike frequency adaptation block ``t_pulse_ahp`` in Amperes with shape (Nrec,)"""

        self.Ipulse = __simparam(Ipulse)
        """bias current setting the pulse width for neuron membrane ``t_pulse`` in Amperes with shape (Nrec,)"""

        self.Iref = __simparam(Iref)
        """bias current setting the refractory period ``t_ref`` in Amperes with shape (Nrec,)"""

        self.Ispkthr = __simparam(Ispkthr)
        """spiking threshold current, neuron spikes if :math:`I_{mem} > I_{spkthr}` in Amperes with shape (Nrec,)"""

        self.Itau_ahp = __simparam(Itau_ahp)
        """Spike frequency adaptation leakage current setting the time constant ``tau_ahp`` in Amperes with shape (Nrec,)"""

        self.Itau_mem = __simparam(Itau_mem)
        """Neuron membrane leakage current setting the time constant ``tau_mem`` in Amperes with shape (Nrec,)"""

        self.Itau_syn = __simparam(Itau_syn)
        """(AMPA, GABA, NMDA, SHUNT) synapses combined leakage current setting the time constant ``tau_syn`` in Amperes with shape (Nrec,)"""

        self.Iw_ahp = __simparam(Iw_ahp)
        """spike frequency adaptation weight current of the neurons of the core in Amperes with shape (Nrec,)"""

        # -- #

        self.C_ahp = __simparam(C_ahp)
        """AHP synapse capacitance in Farads with shape (Nrec,)"""

        self.C_syn = __simparam(C_syn)
        """synaptic capacitance in Farads with shape (Nrec,)"""

        self.C_pulse_ahp = __simparam(C_pulse_ahp)
        """spike frequency adaptation circuit pulse-width creation sub-circuit capacitance in Farads with shape (Nrec,)"""

        self.C_pulse = __simparam(C_pulse)
        """pulse-width creation sub-circuit capacitance in Farads with shape (Nrec,)"""

        self.C_ref = __simparam(C_ref)
        """refractory period sub-circuit capacitance in Farads with shape (Nrec,)"""

        self.C_mem = __simparam(C_mem)
        """neuron membrane capacitance in Farads with shape (Nrec,)"""

        self.Io = __simparam(Io)
        """Dark current in Amperes that flows through the transistors even at the idle state with shape (Nrec,)"""

        self.kappa_n = __simparam(kappa_n)
        """Subthreshold slope factor (n-type transistor) with shape (Nrec,)"""

        self.kappa_p = __simparam(kappa_p)
        """Subthreshold slope factor (p-type transistor) with shape (Nrec,)"""

        self.Ut = __simparam(Ut)
        """Thermal voltage in Volts with shape (Nrec,)"""

        self.Vth = __simparam(Vth)
        """The cut-off Vgs potential of the transistors in Volts (not type specific) with shape (Nrec,)"""

        # -- #

        self.Iscale = SimulationParameter(
            np.array(Iscale, dtype=np.float32), shape=(1,)
        )
        """weight scaling current of the neurons of the core in Amperes"""

        self.dt = SimulationParameter(np.array(dt, dtype=np.float32), shape=(1,))
        """The time step for the forward-Euler ODE solver"""

        self.rng_key = State(rng_key, init_func=lambda _: rng_key)
        """The Jax RNG seed to use on initialisation. By default, a new seed is generated"""

        # One time mismatch
        if rng_key is None:
            rng_key = jnp.array(
                [np.random.randint(sys.maxsize, size=2)], dtype=jnp.uint32
            )

        if percent_mismatch is not None:
            rng_key, _ = rand.split(rng_key)
            prototype = frozen_mismatch_prototype(self)
            regenerate_mismatch = mismatch_generator(
                prototype=prototype, percent_deviation=percent_mismatch
            )
            new_params = regenerate_mismatch(self, rng_key=rng_key)
            for key in new_params:
                self.__setattr__(key, new_params[key])

        # - Define additional arguments required during initialisation
        self._init_args = {
            "has_rec": has_rec,
            "weight_init_func": Partial(weight_init_func),
        }

    @classmethod
    def from_graph(
        cls, se: DynapseNeurons, weights: Optional[LinearWeights] = None
    ) -> DynapSim:
        """
        from_graph constructs a ``DynapSim`` object from a computational graph

        :param se: the reference computational graph to restore the computational module
        :type se: DynapseNeurons
        :param weights: additional weights graph if one wants to impose recurrent weights, defaults to None
        :type weights: Optional[LinearWeights], optional
        :return: a ``DynapSim`` object
        :rtype: DynapSim
        """
        if not isinstance(se, DynapseNeurons):
            se = DynapseNeurons._convert_from(se)

        if weights is not None:
            if weights.biases is not None:
                raise ValueError("Recurrent weight layer biases cannot be defined!")

        kwargs = {k: np.array(v) for k, v in se.get_full().items()}

        return cls(
            shape=(len(se.input_nodes), len(se.output_nodes)),
            Iscale=se.Iscale,
            w_rec=np.array(weights.weights) if weights is not None else None,
            has_rec=True if weights is not None else False,
            dt=se.dt,
            **kwargs,
        )

    def evolve(
        self, input_data: FloatVector, record: bool = True
    ) -> Tuple[jax.Array, Dict[str, jax.Array], Dict[str, jax.Array]]:
        """
        evolve implements raw rockpool JAX evolution function for a DynapSim module.
        The function solves the dynamical equations introduced at the ``DynapSim`` module definition

        :param input_data: Input array of shape ``(T, Nrec, 4)`` to evolve over. Represents number of spikes at that timebin for different synaptic gates
        :type input_data: FloatVector
        :param record: record the each timestep of evolution or not, defaults to True
        :type record: bool, optional
        :return: spikes_ts, states, record_dict
            :spikes_ts: is an array with shape ``(T, Nrec)`` containing the output data(spike raster) produced by the module.
            :states: is a dictionary containing the updated module state following evolution.
            :record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True`` else empty dictionary {}
        :rtype: Tuple[jax.Array, Dict[str, jax.Array], Dict[str, jax.Array]]
        """

        kappa = (self.kappa_n + self.kappa_p) / 2

        # --- Time constant computation utils --- #
        __pw = lambda ipw, C: (self.Vth * C) / ipw
        __tau = lambda itau, C: ((self.Ut / kappa) * C.T).T / itau

        tau_mem = lambda itau: __tau(itau, self.C_mem)

        # --- Stateless Parameters --- #
        t_ref = __pw(self.Iref, self.C_ref)
        t_pulse = __pw(self.Ipulse, self.C_pulse)
        t_pulse_ahp = __pw(self.Ipulse_ahp, self.C_pulse_ahp)

        ## --- Synapse --- ## Nrec
        Itau_syn_clip = jnp.clip(self.Itau_syn, self.Io)
        Igain_syn_clip = jnp.clip(self.Igain_syn, self.Io)
        tau_syn = __tau(Itau_syn_clip, self.C_syn)

        ## --- Spike frequency adaptation --- ## Nrec
        Itau_ahp_clip = jnp.clip(self.Itau_ahp, self.Io)
        Igain_ahp_clip = jnp.clip(self.Igain_ahp, self.Io)
        tau_ahp = __tau(Itau_ahp_clip, self.C_ahp)

        ## -- Membrane -- ## Nrec
        Itau_mem_clip = jnp.clip(self.Itau_mem, self.Io)
        Igain_mem_clip = jnp.clip(self.Igain_mem, self.Io)

        # Handle Batches
        initial_state = (
            self.iahp,
            self.imem,
            self.iampa,
            self.rng_key,
            self.spikes,
            self.timer_ref,
            self.vmem,
        )

        input_data, initial_state = self._auto_batch(input_data, initial_state)

        def forward(
            state: DynapSimState, ws_input: jax.Array
        ) -> Tuple[DynapSimState, DynapSimRecord]:
            """
            forward implements single time-step neuron and synapse dynamics

            :param state: (iahp, iampa, igaba, imem, inmda, ishunt, rng_key, spikes, timer_ref, vmem)
                iahp: Spike frequency adaptation currents of each neuron [Nrec]
                imem: Membrane currents of each neuron [Nrec]
                inmda: sum of synapse currents of each neuron [Nrec]
                rng_key: The Jax RNG seed to be used for mismatch simulation
                spikes: Logical spike raster for each neuron [Nrec]
                timer_ref: Refractory timer of each neruon [Nrec]
                vmem: Membrane voltages of each neuron [Nrec]
            :type state: DynapSimState
            :param ws_input: weighted input spikes [Nrec, 4]
            :type ws_input: jax.Array
            :return: state, record
                state: Updated state at end of the forward steps
                record: Updated record instance to including spikes, igaba, ishunt, inmda, iampa, iahp, imem, and vmem states
            :rtype: Tuple[DynapSimState, DynapSimRecord]
            """

            (
                iahp,
                imem,
                isyn,
                rng_key,
                spikes,
                timer_ref,
                vmem,
            ) = state

            # ---------------------------------- #
            # --- Forward step: DPI SYNAPSES --- #
            # ---------------------------------- #

            ## Real time weight is 0 if no spike, w_rec if spike event occurs
            ws_rec = jnp.dot(self.w_rec.T, spikes).T  # Nrec
            Iws = (ws_rec + ws_input) * self.Iscale

            # isyn_inf is the current that a synapse current would reach with a sufficiently long pulse
            isyn_inf = (Igain_syn_clip / Itau_syn_clip) * Iws
            isyn_inf = jnp.clip(isyn_inf, self.Io)

            ## Exponential charge, discharge positive feedback factor arrays
            f_charge = 1.0 - jnp.exp(-t_pulse / tau_syn.T).T  # Nrecx4
            f_discharge = jnp.exp(-self.dt / tau_syn)  # Nrecx4

            ## DISCHARGE in any case
            isyn = f_discharge * isyn

            ## CHARGE if spike occurs -- UNDERSAMPLED -- dt >> t_pulse
            isyn += f_charge * isyn_inf

            # ------------------------------------------------------ #
            # --- Forward step: AHP : Spike Frequency Adaptation --- #
            # ------------------------------------------------------ #

            Iws_ahp = self.Iw_ahp * spikes  # 0 if no spike, Iw_ahp if spike
            iahp_inf = (Igain_ahp_clip / Itau_ahp_clip) * Iws_ahp

            # Calculate charge and discharge factors
            f_charge_ahp = 1.0 - jnp.exp(-t_pulse_ahp / tau_ahp)  # Nrec
            f_discharge_ahp = jnp.exp(-self.dt / tau_ahp)  # Nrec

            ## DISCHARGE in any case
            iahp = f_discharge_ahp * iahp

            ## CHARGE if spike occurs -- UNDERSAMPLED -- dt >> t_pulse
            iahp += f_charge_ahp * iahp_inf
            iahp = jnp.clip(iahp, self.Io)  # Nrec

            # ------------------------------ #
            # --- Forward step: MEMBRANE --- #
            # ------------------------------ #

            ## Feedback
            _kappa_2 = jnp.power(kappa, 2.0)
            _kappa_prime = _kappa_2 / (kappa + 1.0)
            f_feedback = jnp.exp(_kappa_prime * (vmem / self.Ut))  # 4xNrec

            ## Leakage
            Ileak = Itau_mem_clip + iahp

            ## Injection
            Iin = isyn - Ileak + self.Idc
            Iin *= jnp.logical_not(timer_ref.astype(bool)).astype(jnp.float32)
            Iin = jnp.clip(Iin, self.Io)

            ## Steady state current
            imem_inf = (Igain_mem_clip / Itau_mem_clip) * (Iin - Ileak)

            ## Positive feedback
            Ifb = self.Io * f_feedback
            f_imem = ((Ifb) / (Ileak)) * (imem + Igain_mem_clip)

            ## Forward Euler Update
            del_imem = (imem / (tau_mem(Ileak) * (imem + Igain_mem_clip))) * (
                imem_inf + f_imem - (imem * (1.0 + (iahp / Itau_mem_clip)))
            )
            imem = imem + del_imem * self.dt
            imem = jnp.clip(imem, self.Io)

            ## Membrane Potential
            vmem = (self.Ut / kappa) * jnp.log(imem / self.Io)

            # ------------------------------ #
            # --- Spike Generation Logic --- #
            # ------------------------------ #

            ## Detect next spikes (with custom gradient)
            spikes = step_pwl(imem, self.Ispkthr, self.Io)

            ## Reset imem depending on spiking activity
            bool_spikes = jnp.clip(spikes, 0, 1)
            imem = (1.0 - bool_spikes) * imem + bool_spikes * self.Io

            ## Set the refractrory timer
            timer_ref -= self.dt
            timer_ref = jnp.clip(timer_ref, 0.0)
            timer_ref = (1.0 - bool_spikes) * timer_ref + bool_spikes * t_ref

            # ------------------------------ #
            # ----------- Output ----------- #
            # ------------------------------ #

            # ! IMPORTANT ! : SHOULD BE IN THE SAME ORDER WITH THE self.state()
            state = (
                iahp,
                imem,
                isyn,
                rng_key,
                spikes,
                timer_ref,
                vmem,
            )
            record_ts = (iahp, imem, isyn, spikes, vmem)
            return state, record_ts

        # --- Evolve over spiking inputs --- #

        ## Map over batches
        @jax.vmap
        def scan_time(state, data):
            return scan(forward, state, data)

        ## Scan
        state, record_ts = scan_time(initial_state, input_data)

        # --- Output --- #

        states = {
            "iahp": state[0],
            "imem": state[1],
            "isyn": state[2],
            "rng_key": state[3],
            "spikes": state[4],
            "timer_ref": state[5],
            "vmem": state[6],
        }

        record_dict = {
            "iahp": record_ts[0],
            "imem": record_ts[1],
            "isyn": record_ts[2],
            "spikes": record_ts[3],
            "vmem": record_ts[4],
        }

        return record_ts[3], states, record_dict

    def as_graph(self) -> GraphHolder:
        """
        as_graph returns a computational graph for the for the simulated Dynap-SE neurons

        :return: a ``GraphHolder`` object wrapping the DynapseNeurons graph.
        :rtype: GraphHolder
        """

        # Get simulated current parameters
        kwargs = {
            __attr: np.array(self.__getattribute__(__attr)).flatten().tolist()
            for __attr in DynapseNeurons.current_attrs()
        }

        # Generate the main computational graph
        neurons = DynapseNeurons._factory(
            size_in=self.size_in,
            size_out=self.size_out,
            name=f"{type(self).__name__}_{self.name}_{id(self)}",
            computational_module=self,
            Iscale=float(np.array(self.Iscale).mean()),
            dt=self.dt,
            **kwargs,
        )

        # - Include recurrent weights if present
        if np.array(self.w_rec).any():
            # - Weights are connected over the existing input and output nodes
            w_rec_graph_auto_connected = LinearWeights(
                neurons.output_nodes,
                neurons.input_nodes,
                f"{type(self).__name__}_recurrent_{self.name}_{id(self)}",
                self,
                self.w_rec,
            )

        return as_GraphHolder(neurons)
