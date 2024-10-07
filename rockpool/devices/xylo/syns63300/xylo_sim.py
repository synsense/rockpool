"""
XyloSim-backed module compatible with Xylo™IMU and Xylo™Audio 3. Requires XyloSim package.
"""

# - Rockpool imports
from rockpool import TSContinuous, TSEvent

from rockpool.devices.xylo.syns61300.xylo_sim import XyloSim as XyloSimV1

from xylosim.v2 import XyloSynapse, XyloLayer


# - Numpy
import numpy as np

# - Typing
from typing import Optional, Union, Any, Dict

XyloConfiguration = Union[Dict, Any]

# - Define exports
__all__ = ["XyloSim"]


class XyloSim(XyloSimV1):
    """
    A :py:class:`.Module` simulating a digital SNN on Xylo, using XyloSim as a back-end.

    You should use the factory methods `.from_config` and `.from_specification` to build a concrete `.XyloSim` module.

    See Also:

        See the tutorials :ref:`/devices/xylo-overview.ipynb` and :ref:`/devices/torch-training-spiking-for-xylo.ipynb` for a high-level overview of building and deploying networks for Xylo.
    """

    @classmethod
    def from_config(
        cls,
        config: XyloConfiguration,
        dt: float = 1e-3,
        output_mode: str = "Spike",
        *args,
        **kwargs,
    ) -> "XyloSim":
        """
        Create a XyloSim based layer to simulate the Xylo hardware, from a configuration

        Args:
            config (XyloConfiguration): ``samna.xyloImu.XyloConfiguration`` object to specify all parameters. See samna documentation for details.
            dt (float, optional): Timestep for simulation. Defaults to 1e-3.
            output_mode (str, optional): readout mode. one of ["Isyn", "Vmem", "Spike"]. Defaults to "Spike".

        Returns:
            XyloSim: XyloSim object instance
        """

        # - Extract network dimensions
        IN, IEN = np.shape(config.input.weights)[0:2]
        RSN = np.shape(config.hidden.weights)[0]
        OEN, ON = np.shape(config.readout.weights)

        assert (
            OEN <= RSN
        ), f"Config must satisfy OEN <= RSN. Found OEN {OEN} and RSN {RSN}."
        assert (
            IEN <= RSN
        ), f"Config must satisfy IEN <= RSN. Found IEN {IEN} and RSN {RSN}."

        cls.output_mode = output_mode

        # - Determine the size of the simulated network
        if np.array(config.input.weights).ndim == 2:
            Nin, NIEN = np.array(config.input.weights).shape
            Nsyn = 1
        else:
            Nin, NIEN, Nsyn = np.array(config.input.weights).shape

        Nhid, *_ = np.array(config.hidden.weights).shape
        NOEN, Nout = np.array(config.readout.weights).shape

        # - Instantiate the class
        mod = cls(
            create_key=cls.__create_key,
            config=config,
            shape=(Nin, Nhid, Nout),
            dt=dt,
            output_mode=cls.output_mode,
        )

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

            _xylo_sim_params.synapses_in.append(tmp)

        # - Convert recurrent weights to XyloSynapse objects
        _xylo_sim_params.synapses_rec = []
        for pre, w_pre in enumerate(config.hidden.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(XyloSynapse(post, 0, w_pre[post]))

            _xylo_sim_params.synapses_rec.append(tmp)

        # - Convert output weights to XyloSynapse objects
        _xylo_sim_params.synapses_out = [
            [] for _ in range(RSN - OEN)
        ]  # - Skip unconnected hidden neurons
        for pre, w_pre in enumerate(config.readout.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(XyloSynapse(post, 0, w_pre[post]))
            _xylo_sim_params.synapses_out.append(tmp)

        # - Configure hidden neurons
        _xylo_sim_params.threshold = []
        _xylo_sim_params.dash_syn = []
        _xylo_sim_params.dash_mem = []
        _xylo_sim_params.aliases = []
        _xylo_sim_params.bias = []

        for neuron in config.hidden.neurons:
            if neuron.alias_target:
                _xylo_sim_params.aliases.append([neuron.alias_target])
            else:
                _xylo_sim_params.aliases.append([])
            _xylo_sim_params.threshold.append(neuron.threshold)
            _xylo_sim_params.dash_mem.append(neuron.v_mem_decay)
            _xylo_sim_params.dash_syn.append([neuron.i_syn_decay, 0])
            _xylo_sim_params.bias.append(neuron.v_mem_bias)

        # - Configure readout neurons
        _xylo_sim_params.threshold_out = []
        _xylo_sim_params.dash_syn_out = []
        _xylo_sim_params.dash_mem_out = []
        _xylo_sim_params.bias_out = []

        for neuron in config.readout.neurons:
            _xylo_sim_params.threshold_out.append(neuron.threshold)
            _xylo_sim_params.dash_mem_out.append(neuron.v_mem_decay)
            _xylo_sim_params.dash_syn_out.append([neuron.i_syn_decay])
            _xylo_sim_params.bias_out.append(neuron.v_mem_bias)

        _xylo_sim_params.weight_shift_inp = config.input.weight_bit_shift
        _xylo_sim_params.weight_shift_rec = config.hidden.weight_bit_shift
        _xylo_sim_params.weight_shift_out = config.readout.weight_bit_shift

        _xylo_sim_params.has_bias = False
        if config.bias_enable:
            _xylo_sim_params.has_bias = True

        # - Instantiate a Xylo Simulation layer
        mod._xylo_layer = XyloLayer(
            synapses_in=_xylo_sim_params.synapses_in,
            synapses_rec=_xylo_sim_params.synapses_rec,
            synapses_out=_xylo_sim_params.synapses_out,
            aliases=_xylo_sim_params.aliases,
            threshold=_xylo_sim_params.threshold,
            threshold_out=_xylo_sim_params.threshold_out,
            has_bias=_xylo_sim_params.has_bias,
            bias=_xylo_sim_params.bias,
            bias_out=_xylo_sim_params.bias_out,
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
        bias: Optional[np.ndarray] = None,
        bias_out: Optional[np.ndarray] = None,
        weight_shift_in: int = 0,
        weight_shift_rec: int = 0,
        weight_shift_out: int = 0,
        aliases: Optional[list] = None,
        dt: float = 1e-3,
        verify_config: bool = True,
        output_mode: str = "Spike",
        *args,
        **kwargs,
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
            bias (Optional[np.ndarray]): An int8 matrix ``(Nhidden)``, specifying the bias for each hidden neuron. Default: ``0``.
            bias_out (Optional[np.ndarray]): An int8 matrix ``(Nhidden)``, specifying the bias for each output neuron. Default: ``0``.
            weight_shift_in (int): An integer number of bits to left-shift the input weight matrix
            weight_shift_rec (int): An integer number of bits to left-shift the hidden weight matrix
            weight_shift_out (int): An integer number of bits to left-shift the output weight matrix
            aliases (Optional[list]):
            dt (float): Simulation time step in seconds. Default: 1 ms
            verify_config (bool): Check for a valid configuraiton before applying it. Default ``True``.
            output_mode (str, optional): readout mode. one of ["Isyn", "Vmem", "Spike"]. Defaults to "Spike".

        Returns:
            :py:class:`.XyloSim`: A :py:class:`.Module` that emulates the Xylo hardware.

        Raises:
            ValueError: If ``verify_config`` is ``True`` and the configuration is not valid.
        """
        cls.output_mode = output_mode

        # - Extract network dimensions
        IN, IEN = weights_in.shape[0:2]
        RSN = weights_rec.shape[0] if weights_rec is not None else IEN
        OEN, ON = weights_out.shape

        assert (
            OEN <= RSN
        ), f"Config must satisfy OEN <= RSN. Found OEN {OEN} and RSN {RSN}."
        assert (
            IEN <= RSN
        ), f"Config must satisfy IEN <= RSN. Found IEN {IEN} and RSN {RSN}."

        if weights_rec is None:
            weights_rec = np.zeros((RSN, RSN, 2), int)

        if dash_syn is None:
            dash_syn = np.zeros(RSN, int)

        if dash_syn_2 is None:
            dash_syn_2 = np.zeros(RSN, int)

        if dash_mem is None:
            dash_mem = np.zeros(RSN, int)

        if dash_syn_out is None:
            dash_syn_out = np.zeros(ON, int)

        if dash_mem_out is None:
            dash_mem_out = np.zeros(ON, int)

        if bias is None:
            bias = np.zeros(RSN, int)

        if bias_out is None:
            bias_out = np.zeros(ON, int)

        if aliases is None:
            aliases = [[] for _ in range(RSN)]

        if threshold is None:
            threshold = np.zeros(RSN, int)

        if threshold_out is None:
            threshold_out = np.zeros(ON, int)

        # - Instantiate the class
        mod = cls(
            create_key=cls.__create_key, config=None, dt=dt, output_mode=cls.output_mode
        )

        # - Make a storage object for the extracted configuration
        class _(object):
            pass

        _xylo_sim_params = _()

        # - Convert input weights to XyloSynapse objects
        if len(weights_in.shape) == 2:
            weights_in = np.expand_dims(weights_in, 2)

        _xylo_sim_params.synapses_in = []
        for pre, w_pre in enumerate(weights_in[:, :, 0]):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(XyloSynapse(post, 0, w_pre[post]))

            if weights_in.shape[2] == 2:
                w2_pre = weights_in[:, :, 1][pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(XyloSynapse(post, 1, w2_pre[post]))

            _xylo_sim_params.synapses_in.append(tmp)

        # - Convert recurrent weights to XyloSynapse objects
        if len(weights_rec.shape) == 2:
            weights_rec = np.expand_dims(weights_rec, 2)

        _xylo_sim_params.synapses_rec = []
        for pre, w_pre in enumerate(weights_rec[:, :, 0]):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(XyloSynapse(post, 0, w_pre[post]))

            if weights_rec.shape[2] == 2:
                w2_pre = weights_rec[:, :, 1][pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(XyloSynapse(post, 1, w2_pre[post]))

            _xylo_sim_params.synapses_rec.append(tmp)

        # - Convert output weights to XyloSynapse objects
        _xylo_sim_params.synapses_out = [
            [] for _ in range(RSN - OEN)
        ]  # - Skip unconnected hidden neurons
        for pre, w_pre in enumerate(weights_out):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(XyloSynapse(post, 0, w_pre[post]))
            _xylo_sim_params.synapses_out.append(tmp)

        # - Configure hidden neurons
        _xylo_sim_params.threshold = threshold
        _xylo_sim_params.dash_syn = [list(l) for l in list(zip(dash_syn, dash_syn_2))]
        _xylo_sim_params.dash_mem = dash_mem
        _xylo_sim_params.aliases = aliases
        _xylo_sim_params.bias = bias

        # - Configure readout neurons
        _xylo_sim_params.threshold_out = threshold_out
        _xylo_sim_params.dash_syn_out = dash_syn_out
        _xylo_sim_params.dash_syn_out = [[d] for d in dash_syn_out]
        _xylo_sim_params.dash_mem_out = dash_mem_out
        _xylo_sim_params.bias_out = bias_out

        _xylo_sim_params.weight_shift_inp = weight_shift_in
        _xylo_sim_params.weight_shift_rec = weight_shift_rec
        _xylo_sim_params.weight_shift_out = weight_shift_out

        _xylo_sim_params.has_bias = any([b != 0 for b in bias]) or any(
            [b != 0 for b in bias_out]
        )

        # - Instantiate a Xylo Simulation layer
        mod._xylo_layer = XyloLayer(
            synapses_in=_xylo_sim_params.synapses_in,
            synapses_rec=_xylo_sim_params.synapses_rec,
            synapses_out=_xylo_sim_params.synapses_out,
            aliases=_xylo_sim_params.aliases,
            threshold=_xylo_sim_params.threshold,
            threshold_out=_xylo_sim_params.threshold_out,
            has_bias=_xylo_sim_params.has_bias,
            bias=_xylo_sim_params.bias,
            bias_out=_xylo_sim_params.bias_out,
            weight_shift_inp=_xylo_sim_params.weight_shift_inp,
            weight_shift_rec=_xylo_sim_params.weight_shift_rec,
            weight_shift_out=_xylo_sim_params.weight_shift_out,
            dash_mem=_xylo_sim_params.dash_mem,
            dash_mem_out=_xylo_sim_params.dash_mem_out,
            dash_syns=_xylo_sim_params.dash_syn,
            dash_syns_out=_xylo_sim_params.dash_syn_out,
            name="XyloSimV2_XyloLayer",
        )

        # - Store parameters and return
        mod._xylo_sim_params = _xylo_sim_params
        return mod

    def evolve(
        self,
        input_raster: np.ndarray = None,
        record: bool = False,
        *args,
        **kwargs,
    ):
        # - Evolve using the xylo layer
        spike_out = np.array(self._xylo_layer.evolve(input_raster.astype(int).tolist()))
        if self.output_mode == "Spike":
            output = spike_out
        elif self.output_mode == "Vmem":
            output = np.array(self._xylo_layer.rec_v_mem_out).T
        elif self.output_mode == "Isyn":
            output = np.array(self._xylo_layer.rec_i_syn_out).T

        # - Build the recording dictionary
        if not record:
            recording = {}
        else:
            recording = {
                "Vmem": np.array(self._xylo_layer.rec_v_mem).T,
                "Isyn": np.array(self._xylo_layer.rec_i_syn).T,
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
            "Spikes": TSEvent.from_raster(state_dict["Spikes"], name="Spikes", **args),
            "Vmem_out": TSContinuous.from_clocked(
                state_dict["Vmem_out"], name="$V_{mem,out}$", **args
            ),
            "Isyn_out": TSContinuous.from_clocked(
                state_dict["Isyn_out"], name="$I_{syn,out}$", **args
            ),
        }
