"""
XyloSim-backed module compatible with Xylo. Requires XyloSim
"""

# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.parameters import Parameter, State, SimulationParameter
from rockpool import TSContinuous, TSEvent
from rockpool.utilities.backend_management import backend_available


# - Import XyloSim
from xylosim.v1 import XyloSynapse, XyloLayer

# - Numpy
import numpy as np

# - Typing
from typing import Optional, Union, Any, Dict

XyloConfiguration = Union[Dict, Any]

# - Define exports
__all__ = ["XyloSim"]


class XyloSim(Module):
    """
    A :py:class:`.Module` simulating a digital SNN on Xylo, using XyloSim as a back-end.

    You should use the factory methods `.from_config` and `.from_specification` to build a concrete `.XyloSim` module.

    See Also:

        See the tutorials :ref:`/devices/xylo-overview.ipynb` and :ref:`/devices/torch-training-spiking-for-xylo.ipynb` for a high-level overview of building and deploying networks for Xylo.
    """

    __create_key = object()
    """ Private key to ensure factory creation """

    def __init__(
        self,
        create_key,
        config: XyloConfiguration,
        shape: tuple = (16, 1000, 8),
        dt: float = 1e-3,
        *args,
        **kwargs,
    ):
        """
        Private constructor for :py:class:`.XyloSim`

        Warnings:
            Use the factory methods :py:meth:`.XyloSim.from_config` and :py:meth:`XyloSim.from_specfication` to construct a :py:class:`.XyloSim` module.
        """
        # - Check that we are creating the object using a factory function
        if create_key is not XyloSim.__create_key:
            raise NotImplementedError(
                "XyloSim may only be instantiated using factory methods `from_config` or `from_weights`."
            )

        # - Initialise the superclass
        super().__init__(
            shape=shape,
            spiking_input=True,
            spiking_output=True,
            *args,
            **kwargs,
        )

        # - Store the configuration
        self.config: Union[XyloConfiguration, Parameter] = Parameter(
            shape=(), init_func=lambda _: config
        )
        """ (XyloConfiguration) Configuration of the Xylo module """

        # - Store the dt
        self.dt: Union[float, SimulationParameter] = SimulationParameter(dt)
        """ (float) Simulation time-step for this module """

        # - Empty attribute for the Xylo layer
        self._xylo_layer: Optional[XyloLayer] = None
        """ (XyloLayer) Handle to a XyloSim object """

    @classmethod
    def from_config(cls, config: XyloConfiguration, dt: float = 1e-3):
        """
        Creata a XyloSim based layer to simulate the Xylo hardware, from a configuration

        Parameters:
        dt: float
            Timestep for simulation, in seconds. Default: 1ms
        config: XyloConfiguration
            ``samna.xylo.XyloConfiguration`` object to specify all parameters. See samna documentation for details.

        """
        # - Instantiate the class
        mod = cls(create_key=cls.__create_key, config=config, dt=dt)

        # - Make a storage object for the extracted configuration
        class _(object):
            pass

        _xylo_sim_params = _()

        # - Convert input weights to XyloSynapse objects
        _xylo_sim_params.synapses_in = []
        for pre, w_pre in enumerate(config.input.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(XyloSynapse(post, 0, w_pre[post]))

            if config.synapse2_enable:
                w2_pre = config.input.syn2_weights[pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(XyloSynapse(post, 1, w2_pre[post]))

            _xylo_sim_params.synapses_in.append(tmp)

        # - Convert recurrent weights to XyloSynapse objects
        _xylo_sim_params.synapses_rec = []
        for pre, w_pre in enumerate(config.reservoir.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(XyloSynapse(post, 0, w_pre[post]))

            if config.synapse2_enable:
                w2_pre = config.reservoir.syn2_weights[pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(XyloSynapse(post, 1, w2_pre[post]))

            _xylo_sim_params.synapses_rec.append(tmp)

        # - Convert output weights to XyloSynapse objects
        _xylo_sim_params.synapses_out = []
        for pre, w_pre in enumerate(config.readout.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(XyloSynapse(post, 0, w_pre[post]))
            _xylo_sim_params.synapses_out.append(tmp)

        # - Configure reservoir neurons
        _xylo_sim_params.threshold = []
        _xylo_sim_params.dash_syn = []
        _xylo_sim_params.dash_mem = []
        _xylo_sim_params.aliases = []

        for neuron in config.reservoir.neurons:
            if neuron.alias_target:
                _xylo_sim_params.aliases.append([neuron.alias_target])
            else:
                _xylo_sim_params.aliases.append([])
            _xylo_sim_params.threshold.append(neuron.threshold)
            _xylo_sim_params.dash_mem.append(neuron.v_mem_decay)
            _xylo_sim_params.dash_syn.append([neuron.i_syn_decay, neuron.i_syn2_decay])

        # - Configure readout neurons
        _xylo_sim_params.threshold_out = []
        _xylo_sim_params.dash_syn_out = []
        _xylo_sim_params.dash_mem_out = []

        for neuron in config.readout.neurons:
            _xylo_sim_params.threshold_out.append(neuron.threshold)
            _xylo_sim_params.dash_mem_out.append(neuron.v_mem_decay)
            _xylo_sim_params.dash_syn_out.append([neuron.i_syn_decay])

        _xylo_sim_params.weight_shift_inp = config.input.weight_bit_shift
        _xylo_sim_params.weight_shift_rec = config.reservoir.weight_bit_shift
        _xylo_sim_params.weight_shift_out = config.readout.weight_bit_shift

        # - Instantiate a Xylo Simulation layer
        mod._xylo_layer = XyloLayer(
            synapses_in=_xylo_sim_params.synapses_in,
            synapses_rec=_xylo_sim_params.synapses_rec,
            synapses_out=_xylo_sim_params.synapses_out,
            aliases=_xylo_sim_params.aliases,
            threshold=_xylo_sim_params.threshold,
            threshold_out=_xylo_sim_params.threshold_out,
            weight_shift_inp=_xylo_sim_params.weight_shift_inp,
            weight_shift_rec=_xylo_sim_params.weight_shift_rec,
            weight_shift_out=_xylo_sim_params.weight_shift_out,
            dash_mem=_xylo_sim_params.dash_mem,
            dash_mem_out=_xylo_sim_params.dash_mem_out,
            dash_syns=_xylo_sim_params.dash_syn,
            dash_syns_out=_xylo_sim_params.dash_syn_out,
            name="XyloSim_XyloLayer",
        )

        # - Store parameters and return
        mod._xylo_sim_params = _xylo_sim_params
        return mod

    @classmethod
    def from_specification(
        cls,
        weights_in: np.ndarray,
        weights_out: np.ndarray,
        weights_rec: Optional[np.ndarray] = None,
        dash_mem: Optional[np.ndarray] = None,
        dash_mem_out: Optional[np.ndarray] = None,
        dash_syn: Optional[np.ndarray] = None,
        dash_syn_2: Optional[np.ndarray] = None,
        dash_syn_out: Optional[np.ndarray] = None,
        threshold: Optional[np.ndarray] = None,
        threshold_out: Optional[np.ndarray] = None,
        weight_shift_in: int = 0,
        weight_shift_rec: int = 0,
        weight_shift_out: int = 0,
        aliases: Optional[list] = None,
        dt: float = 1e-3,
        verify_config: bool = True,
    ) -> "XyloSim":
        """
        Instantiate a :py:class:`.XyloSim` module from a full set of parameters

        Args:
            weights_in (np.ndarray): An int8 matrix ``(Nin, Nhidden, 2)``, specifying input to hidden neuron connections. The final dimension specifies the inputs to the two available synapses of the hidden neurons.
            weights_out (np.ndarray): An int8 matrix ``(Nhidden, Nout)``, specifying hidden to output connections.
            weights_rec (Optional[np.ndarray]): An int8 matrix ``(Nhidden, Nhidden, 2)``, specifying recurrent connections within the hidden population. The final dimension specifies the input to the two available synapses on each hidden neuron. Default: ``0``, no recurrent connections.
            dash_mem (Optional[np.ndarray]): An int8 matrix ``(Nhidden)``, specifying the bitshift decay value for each hidden neuron membrane potential. Default: ``1``.
            dash_mem_out (Optional[np.ndarray]): An int8 matrix ``(Nout)``, specifying the bitshift decay value for each output neuron membrane potential. Default: ``1``.
            dash_syn (Optional[np.ndarray]): An int8 matrix ``(Nhidden)``, specifying the bitshift decay value for each hidden neuron synaptic current number 1. Default: ``1``.
            dash_syn_2 (Optional[np.ndarray]): An int8 matrix ``(Nhidden)``, specifying the bitshift decay value for each hidden neuron synaptic current number 2. Default: ``1``.
            dash_syn_out (Optional[np.ndarray]): An int8 matrix ``(Nout)``, specifying the bitshift decay value for each output neuron synaptic current. Default: ``1``.
            threshold (Optional[np.ndarray]): An int8 matrix ``(Nhidden)``, specifying the firing threshold for each hidden neuron. Default: ``0``.
            threshold_out (Optional[np.ndarray]): An int8 matrix ``(Nhidden)``, specifying the firing threshold for each output neuron. Default: ``0``.
            weight_shift_in (int): An integer number of bits to left-shift the input weight matrix
            weight_shift_rec (int): An integer number of bits to left-shift the hidden weight matrix
            weight_shift_out (int): An integer number of bits to left-shift the output weight matrix
            aliases (Optional[list]):
            dt (float): Simulation time step in seconds. Default: 1 ms
            verify_config (bool): Check for a valid configuraiton before applying it. Default ``True``.

        Returns:
            :py:class:`.XyloSim`: A :py:class:`.Module` that emulates the Xylo hardware.

        Raises:
            ValueError: If ``verify_config`` is ``True`` and the configuration is not valid.
        """
        if not backend_available("samna"):
            raise ModuleNotFoundError(
                "`samna` not installed. `samna` is required to generate and validate a HW configuration for Xylo."
            )

        from rockpool.devices.xylo import config_from_specification

        # - Convert specification to xylo configuration
        config, is_valid, status = config_from_specification(
            weights_in=weights_in,
            weights_rec=weights_rec,
            weights_out=weights_out,
            dash_mem=dash_mem,
            dash_mem_out=dash_mem_out,
            dash_syn=dash_syn,
            dash_syn_2=dash_syn_2,
            dash_syn_out=dash_syn_out,
            threshold=threshold,
            threshold_out=threshold_out,
            weight_shift_in=weight_shift_in,
            weight_shift_rec=weight_shift_rec,
            weight_shift_out=weight_shift_out,
            aliases=aliases,
        )

        if verify_config and not is_valid:
            raise ValueError("Xylo configuration is not valid: " + status)

        # - Instantiate module from config
        return cls.from_config(config, dt=dt)

    def evolve(
        self,
        input_raster: np.ndarray = None,
        record: bool = False,
        *args,
        **kwargs,
    ):
        # - Evolve using the xylo layer
        output = np.array(self._xylo_layer.evolve(input_raster))

        # - Build the recording dictionary
        if not record:
            recording = {}
        else:
            recording = {
                "Vmem": np.array(self._xylo_layer.rec_v_mem).T,
                "Isyn": np.array(self._xylo_layer.rec_i_syn).T,
                "Isyn2": np.array(self._xylo_layer.rec_i_syn2).T,
                "Spikes": np.array(self._xylo_layer.rec_recurrent_spikes),
                "Vmem_out": np.array(self._xylo_layer.rec_v_mem_out).T,
                "Isyn_out": np.array(self._xylo_layer.rec_i_syn_out).T,
            }

        # - Return output, state and recording dictionary
        return output, {}, recording

    def reset_state(self) -> "XyloSim":
        """Reset the state of this module."""
        self._xylo_layer.reset_all()
        return self

    def _wrap_recorded_state(self, state_dict: dict, t_start: float = 0.0) -> dict:
        args = {"dt": self.dt, "t_start": t_start}

        return {
            "Vmem": TSContinuous.from_clocked(
                state_dict["Vmem"], name="$V_{mem}$", **args
            ),
            "Isyn": TSContinuous.from_clocked(
                state_dict["Isyn"], name="$I_{syn}$", **args
            ),
            "Isyn2": TSContinuous.from_clocked(
                state_dict["Isyn2"], name="$I_{syn,2}$", **args
            ),
            "Spikes": TSEvent.from_raster(state_dict["Spikes"], name="Spikes", **args),
            "Vmem_out": TSContinuous.from_clocked(
                state_dict["Vmem_out"], name="$V_{mem,out}$", **args
            ),
            "Isyn_out": TSContinuous.from_clocked(
                state_dict["Isyn_out"], name="$I_{syn,out}$", **args
            ),
        }
