import multiprocessing
import importlib
from typing import Optional, Union, List, Dict

import numpy as np

from .iaf_nest import (
    RecIAFSpkInNest,
    _BaseNestProcess,
    s2ms,
    V2mV,
    COMMAND_GET,
    COMMAND_SET,
)


if importlib.util.find_spec("nest") is None:
    raise ModuleNotFoundError("No module named 'nest'.")


# - RecAEIFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
class RecAEIFSpkInNest(RecIAFSpkInNest):
    """ RecAEIFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
    """

    class NestProcess(_BaseNestProcess):
        """ Class for running NEST in its own process """

        def __init__(
            self,
            request_q,
            result_q,
            weights_in: np.ndarray,
            weights_rec: np.ndarray,
            delay_in: Union[float, np.ndarray],
            delay_rec: Union[float, np.ndarray],
            bias: Union[float, np.ndarray],
            dt: float,
            tau_mem: Union[float, np.ndarray],
            tau_syn_exc: Union[float, np.ndarray],
            tau_syn_inh: Union[float, np.ndarray],
            capacity: Union[float, np.ndarray],
            v_thresh: Union[float, np.ndarray],
            v_peak: Union[float, np.ndarray],
            v_reset: Union[float, np.ndarray],
            v_rest: Union[float, np.ndarray],
            refractory,
            a: Union[float, np.ndarray],
            b: Union[float, np.ndarray],
            delta_t: Union[float, np.ndarray],
            tau_w: Union[float, np.ndarray],
            record: bool = False,
            num_cores: int = 1,
        ):
            """ initializes the process """
            super().__init__(
                request_q=request_q,
                result_q=result_q,
                bias=bias,
                dt=dt,
                tau_mem=tau_mem,
                capacity=capacity,
                v_thresh=v_thresh,
                v_reset=v_reset,
                v_rest=v_rest,
                refractory=refractory,
                record=record,
                num_cores=num_cores,
                model="aeif_psc_exp",
            )

            # - Record weights and layer-specific parameters
            self.weights_in = V2mV(weights_in)
            self.weights_rec = V2mV(weights_rec)
            self.size = np.shape(weights_in)[1]
            self.delay_in = s2ms(delay_in)
            self.delay_rec = s2ms(delay_rec)
            self.tau_syn_exc = s2ms(tau_syn_exc)
            self.tau_syn_inh = s2ms(tau_syn_inh)
            self.v_peak = V2mV(v_peak)
            self.a = a
            self.b = V2mV(b)
            self.delta_t = delta_t
            self.tau_w = s2ms(tau_w)

        ######### DEFINE IPC COMMANDS ######

        def set_param(self, name, value):
            """ IPC command for setting a parameter """

            if name == "weights_in":
                weights_old = self.weights_in.copy()
                self.weights_in = V2mV(value)
                self.update_weights(
                    self._sg, self._pop, self.weights_in, weights_old, self.delay_in
                )
            elif name == "weights_rec":
                weights_old = self.weights_rec.copy()
                self.weights_rec = V2mV(value)
                self.update_weights(
                    self._pop, self._pop, self.weights_rec, weights_old, self.delay_rec
                )
            else:
                super().set_param(name, value)

        def reset(self):
            """
            reset_all - IPC command which resets time and state
            """
            self.nest_module.ResetNetwork()
            self.nest_module.SetKernelStatus({"time": 0.0})
            # - Manually reset state parameters
            for name in ("I_syn_ex", "I_syn_in", "w"):
                self.set_param(name, 0.0)
            self.set_param("V_m", self.v_rest)

        def evolve(
            self, event_times, event_channels, num_timesteps: Optional[int] = None
        ) -> (np.ndarray, np.ndarray, Union[np.ndarray, None]):
            """ IPC command running the network for num_timesteps with input_steps as input """

            if len(event_channels > 0):
                # convert input index to NEST id
                event_channels += np.min(self._sg)

                # NEST time starts with 1 (not with 0)
                self.nest_module.SetStatus(
                    self._sg,
                    [
                        {"spike_times": s2ms(event_times[event_channels == i])}
                        for i in self._sg
                    ],
                )

            return self.evolve_nest(num_timesteps)

        def setup_nest_network(self):
            # - Add stimulation device
            self._sg = self.nest_module.Create(
                "spike_generator", self.weights_in.shape[0]
            )

            super().setup_nest_network()

        def generate_nest_params_list(self) -> List[Dict[str, np.ndarray]]:
            """init_nest_params - Initialize nest neuron parameters and return as list"""

            params = super().generate_nest_params_list()
            for n in range(self.size):
                params[n]["tau_syn_ex"] = self.tau_syn_exc[n]
                params[n]["tau_syn_in"] = self.tau_syn_inh[n]
                params[n]["g_L"] = self.capacity[n] / self.tau_mem[n]
                params[n]["V_peak"] = self.v_peak[n]
                params[n]["a"] = self.a[n]
                params[n]["b"] = self.b[n]
                params[n]["Delta_T"] = self.delta_t[n]
                params[n]["tau_w"] = self.tau_w[n]
                params[n].pop("tau_m")

            return params

        def set_all_connections(self):
            """Set input connections and recurrent connections"""
            # - Input connections
            self.set_connections(self._sg, self._pop, self.weights_in, self.delay_in)
            # - Recurrent connections
            self.set_connections(self._pop, self._pop, self.weights_rec, self.delay_rec)

    # - Default difference between v_peak and v_thresh when v_peak not set and
    #   delta_t != 0
    _v_peak_offset = 0.01

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        delay_in: Union[float, np.ndarray] = 0.0001,
        delay_rec: Union[float, np.ndarray] = 0.0001,
        bias: Union[float, np.ndarray] = 0.0,
        dt: float = 0.0001,
        tau_mem: Union[float, np.ndarray] = 0.02,
        tau_syn: Union[np.ndarray, float, None] = 0.05,
        tau_syn_exc: Union[float, np.ndarray, None] = None,
        tau_syn_inh: Union[float, np.ndarray, None] = None,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_peak: Union[float, np.ndarray, None] = None,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        capacity: Union[float, np.ndarray, None] = None,
        refractory: Union[float, np.ndarray] = 0.001,
        subthresh_adapt: Union[float, np.ndarray] = 4.0,
        spike_adapt: Union[float, np.ndarray] = 80.5,
        delta_t: Union[float, np.ndarray] = 2.0,
        tau_adapt: Union[float, np.ndarray] = 0.144,
        name: str = "unnamed",
        record: bool = False,
        num_cores: int = 1,
    ):
        """
        RecAEIFSpkInNest - Construct a spiking recurrent layer with AEIF neurons, with a NEST back-end
                           in- and outputs are spiking events

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10.5mA

        :param dt:             float Time-step. Default: 0.1 ms

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 50ms
        :param tau_syn:          np.array Nx1 vector of synapse time constants. Used
                                 Used instead of `tau_syn_exc` or `tau_syn_inh` if they are
                                 None. Default: 20ms
        :param tau_syn_exc:          np.array Nx1 vector of excitatory synapse time constants.
                                     If `None`, use `tau_syn`. Default: `None`
        :param tau_syn_inh:          np.array Nx1 vector of inhibitory synapse time constants.
                                     If `None`, use `tau_syn`. Default: `None`

        :param v_thresh:       np.array Nx1 vector of neuron thresholds ("point of no return"). Default: -55mV
        :param v_peak:         np.array Nx1 vector of neuron spike thresholds. Is set to
                               `v_thresh`If `None` if `delta_T`==0 or to `v_thresh` + 10mV
                               if `deltaT`!=0.
        :param v_reset:        np.array Nx1 vector of neuron reset potential. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron resting potential. Default: -65mV

        :param capacity:       np.array Nx1 vector of neuron membrance capacity. Default: 100 pF
        :param refractory: float Refractory period after each spike. Default: 0ms

        :param subthresh_adapt:              float or np.ndarray scaling for subthreshold adaptation. Default: 4.
        :param spike_adapt:              float or np.ndarray additive value for spike triggered adaptation. Default: 80.5
        :param delta_t:        float or np.ndarray scaling for exponential part of the activation function. Default: 2.
        :param tau_adapt:          float or np.ndarray time constant for adaptation relaxation. Default: 144.0 ms


        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions
        """

        # - Determine layer size and name to run `_expand_to_net_size` method
        self._size_in, self._size = np.atleast_2d(weights_in).shape
        self.name = name

        # - Prepare parameters that are specific to this class
        self._subthresh_adapt = self._expand_to_net_size(
            subthresh_adapt, "subthresh_adapt", allow_none=False
        ).astype(float)
        self._spike_adapt = self._expand_to_net_size(
            spike_adapt, "spike_adapt", allow_none=False
        ).astype(float)
        self._tau_adapt = self._expand_to_net_size(
            tau_adapt, "tau_adapt", allow_none=False
        ).astype(float)
        delta_t = self._expand_to_net_size(delta_t, "delta_t", allow_none=False)
        self._delta_t = delta_t.astype(float)
        if v_peak is None:
            # - Determine v_thresh to determine v_peak (otherwise done by super().__init__)
            v_thresh = self._expand_to_net_size(v_thresh, "v_thresh", allow_none=False)
            self._v_peak = v_thresh.astype(float)
            self._v_peak[self._delta_t != 0] += self._v_peak_offset
        else:
            v_peak = self._expand_to_net_size(v_peak, "v_peak", allow_none=False)
            self._v_peak = v_peak.astype(float)

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            weights_in=weights_in,
            weights_rec=weights_rec,
            delay_in=delay_in,
            delay_rec=delay_rec,
            bias=bias,
            dt=dt,
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            tau_syn_exc=tau_syn_exc,
            tau_syn_inh=tau_syn_inh,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            capacity=capacity,
            refractory=refractory,
            name=name,
            record=record,
            num_cores=num_cores,
        )

    def _setup_nest(self):
        """_setup_nest - Set up and start a nest process"""
        self.request_q = multiprocessing.Queue()
        self.result_q = multiprocessing.Queue()

        self.nest_process = self.NestProcess(
            self.request_q,
            self.result_q,
            weights_in=self._weights_in,
            weights_rec=self._weights_rec,
            delay_in=self._delay_in,
            delay_rec=self._delay_rec,
            bias=self._bias,
            dt=self._dt,
            tau_mem=self._tau_mem,
            tau_syn_exc=self._tau_syn_exc,
            tau_syn_inh=self._tau_syn_inh,
            capacity=self._capacity,
            v_thresh=self._v_thresh,
            v_peak=self._v_peak,
            v_reset=self._v_reset,
            v_rest=self._v_rest,
            refractory=self._refractory,
            record=self._record,
            num_cores=self._num_cores,
            a=self._subthresh_adapt,
            b=self._spike_adapt,
            delta_t=self._delta_t,
            tau_w=self._tau_adapt,
        )
        self.nest_process.start()

    def to_dict(self):

        config = super().to_dict()
        config["subthresh_adapt"] = self._subthresh_adapt.tolist()
        config["spike_adapt"] = self._spike_adapt.tolist()
        config["delta_t"] = self._delta_t.tolist()
        config["tau_adapt"] = self._tau_adapt.tolist()
        config["class_name"] = "RecAEIFSpkInNest"

        return config

    ### --- Properties

    @property
    def adapt(self):
        self.request_q.put([COMMAND_GET, "w"])
        return np.array(self.result_q.get())

    @RecIAFSpkInNest.tau_mem.setter
    def tau_mem(self, new_tau_mem):
        new_tau_mem = self._expand_to_net_size(new_tau_mem, "tau_mem", allow_none=False)
        new_tau_mem = new_tau_mem.astype(float)
        self._tau_mem = new_tau_mem
        self.request_q.put([COMMAND_SET, "g_L", self.capacity / s2ms(new_tau_mem)])

    @property
    def subthresh_adapt(self):
        return self._subthresh_adapt

    @subthresh_adapt.setter
    def subthresh_adapt(self, new_a):
        new_a = self._expand_to_net_size(new_a, "subthresh_adapt", allow_none=False)
        new_a = new_a.astype(float)
        self._subthresh_adapt = new_a
        self.request_q.put([COMMAND_SET, "a", new_a])

    @property
    def spike_adapt(self):
        return self._spike_adapt

    @spike_adapt.setter
    def spike_adapt(self, new_b):
        new_b = self._expand_to_net_size(new_b, "spike_adapt", allow_none=False)
        new_b = new_b.astype(float)
        self._spike_adapt = new_b
        self.request_q.put([COMMAND_SET, "b", V2mV(new_b)])

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, new_delta_t):
        new_delta_t = self._expand_to_net_size(new_delta_t, "delta_t", allow_none=False)
        new_delta_t = new_delta_t.astype(float)
        self.delta_t = new_delta_t
        self.request_q.put([COMMAND_SET, "Delta_T", new_delta_t])

    @property
    def v_peak(self):
        return self._v_peak

    @v_peak.setter
    def v_peak(self, new_v_peak):
        new_v_peak = self._expand_to_net_size(new_v_peak, "v_peak", allow_none=False)
        new_v_peak = new_v_peak.astype(float)
        self.v_peak = new_v_peak
        self.request_q.put([COMMAND_SET, "V_peak", new_v_peak])

    @property
    def tau_adapt(self):
        return self._tau_adapt

    @tau_adapt.setter
    def tau_adapt(self, new_tau):
        new_tau = self._expand_to_net_size(new_tau, "tau_adapt", allow_none=False)
        new_tau = new_tau.astype(float)
        self._tau_adapt = new_tau
        self.request_q.put([COMMAND_SET, "tau_w", s2ms(new_tau)])
