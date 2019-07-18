import multiprocessing
import importlib
from typing import Optional, Union

import numpy as np

from .iaf_nest import RecIAFSpkInNest, s2ms, ms2s, V2mV, mV2V


if importlib.util.find_spec("nest") is None:
    raise ModuleNotFoundError("No module named 'nest'.")


COMMAND_GET = 0
COMMAND_SET = 1
COMMAND_RESET = 2
COMMAND_EVOLVE = 3


# - RecAEIFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
class RecAEIFSpkInNest(RecIAFSpkInNest):
    """ RecAEIFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
    """

    class NestProcess(multiprocessing.Process):
        """ Class for running NEST in its own process """

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

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

            multiprocessing.Process.__init__(self, daemon=True)

            self.request_q = request_q
            self.result_q = result_q

            # - Record neuron parameters
            self.dt = s2ms(dt)
            self.v_thresh = V2mV(v_thresh)
            self.v_reset = V2mV(v_reset)
            self.v_rest = V2mV(v_rest)
            self.tau_mem = s2ms(tau_mem)
            self.tau_syn_exc = s2ms(tau_syn_exc)
            self.tau_syn_inh = s2ms(tau_syn_inh)
            self.bias = V2mV(bias)
            self.capacity = capacity
            self.weights_in = V2mV(weights_in)
            self.weights_rec = V2mV(weights_rec)
            self.delay_in = s2ms(delay_in)
            self.delay_rec = s2ms(delay_rec)
            self.refractory = s2ms(refractory)
            self.record = record
            self.size = np.shape(weights_rec)[0]
            self.num_cores = num_cores
            self.a = a
            self.b = b
            self.delta_t = delta_t
            self.tau_w = s2ms(tau_w)

        def run(self):
            """ start the process. Initializes the network, defines IPC commands and waits for commands. """

            #### INITIALIZE NEST ####
            import nest

            numCPUs = multiprocessing.cpu_count()
            # if self.num_cores >= numCPUs:
            #    self.num_cores = numCPUs

            nest.ResetKernel()
            nest.hl_api.set_verbosity("M_FATAL")
            nest.SetKernelStatus(
                {
                    "resolution": self.dt,
                    "local_num_threads": self.num_cores,
                    "print_time": True,
                }
            )

            self._pop = nest.Create("aeif_psc_exp", self.size)

            params = []
            for n in range(self.size):
                p = {}

                if type(self.tau_syn_exc) is np.ndarray:
                    p["tau_syn_ex"] = self.tau_syn_exc[n]
                else:
                    p["tau_syn_ex"] = self.tau_syn_exc

                if type(self.tau_syn_inh) is np.ndarray:
                    p["tau_syn_in"] = self.tau_syn_inh[n]
                else:
                    p["tau_syn_in"] = self.tau_syn_inh

                if type(self.tau_mem) is np.ndarray:
                    if type(self.capacity) == np.ndarray:
                        p["g_L"] = self.capacity[n] / self.tau_mem[n]
                    else:
                        p["g_L"] = self.capacity / self.tau_mem[n]
                else:
                    if type(self.capacity) == np.ndarray:
                        p["g_L"] = self.capacity[n] / self.tau_mem
                    else:
                        p["g_L"] = self.capacity / self.tau_mem

                if type(self.v_thresh) is np.ndarray:
                    p["V_th"] = self.v_thresh[n]
                else:
                    p["V_th"] = self.v_thresh

                if type(self.v_reset) is np.ndarray:
                    p["V_reset"] = self.v_reset[n]
                else:
                    p["V_reset"] = self.v_reset

                if type(self.v_reset) is np.ndarray:
                    p["E_L"] = self.v_rest[n]
                    p["V_m"] = self.v_rest[n]
                else:
                    p["E_L"] = self.v_rest
                    p["V_m"] = self.v_rest

                if type(self.refractory) is np.ndarray:
                    p["t_ref"] = self.refractory[n]
                else:
                    p["t_ref"] = self.refractory

                if type(self.bias) is np.ndarray:
                    p["I_e"] = self.bias[n]
                else:
                    p["I_e"] = self.bias

                if type(self.capacity) is np.ndarray:
                    p["C_m"] = self.capacity[n]
                else:
                    p["C_m"] = self.capacity

                if type(self.a) is np.ndarray:
                    p["a"] = self.a[n]
                else:
                    p["a"] = self.a

                if type(self.b) is np.ndarray:
                    p["b"] = self.b[n]
                else:
                    p["b"] = self.b

                if type(self.delta_t) is np.ndarray:
                    p["Delta_T"] = self.delta_t[n]
                else:
                    p["Delta_T"] = self.delta_t

                if type(self.tau_w) is np.ndarray:
                    p["tau_w"] = self.tau_w[n]
                else:
                    p["tau_w"] = self.tau_w

                params.append(p)

            nest.SetStatus(self._pop, params)

            # - Add spike detector to record layer outputs
            self._sd = nest.Create("spike_detector")
            nest.Connect(self._pop, self._sd)

            # - Add stimulation device
            self._sg = nest.Create("spike_generator", self.weights_in.shape[0])

            # - Create input connections
            pres = []
            posts = []
            weights = []
            delays = []

            for pre, row in enumerate(self.weights_in):
                for post, w in enumerate(row):
                    if w == 0:
                        continue
                    pres.append(self._sg[pre])
                    posts.append(self._pop[post])
                    weights.append(w)

                    if isinstance(self.delay_in, np.ndarray):
                        delays.append(self.delay_in[pre, post])
                    else:
                        delays.append(self.delay_in)

            if len(weights) > 0:
                delays = np.clip(delays, self.dt, np.max(delays))
                nest.Connect(
                    pres, posts, "one_to_one", {"weight": weights, "delay": delays}
                )

            # - Create recurrent connections
            pres = []
            posts = []
            weights = []
            delays = []

            for pre, row in enumerate(self.weights_rec):
                for post, w in enumerate(row):
                    if w == 0:
                        continue
                    pres.append(self._pop[pre])
                    posts.append(self._pop[post])
                    weights.append(w)
                    if isinstance(self.delay_rec, np.ndarray):
                        delays.append(self.delay_rec[pre, post])
                    else:
                        delays.append(self.delay_rec)

            if len(weights) > 0:
                delays = np.clip(delays, self.dt, np.max(delays))
                nest.Connect(
                    pres, posts, "one_to_one", {"weight": weights, "delay": delays}
                )

            if self.record:
                # - Monitor for recording network potential
                nest.SetDefaults("multimeter", {"interval": self.dt})
                self._mm = nest.Create(
                    "multimeter", 1, {"record_from": ["V_m"], "interval": 1.0}
                )
                nest.Connect(self._mm, self._pop)

            ######### DEFINE IPC COMMANDS ######

            def get_param(name):
                """ IPC command for getting a parameter """
                vms = nest.GetStatus(self._pop, name)
                return vms

            def set_param(name, value):
                """ IPC command for setting a parameter """
                params = []

                for n in range(self.size):
                    p = {}
                    if type(value) is np.ndarray:
                        p[name] = value[n]
                    else:
                        p[name] = value

                    params.append(p)

                nest.SetStatus(self._pop, params)

            def reset():
                """
                reset_all - IPC command which resets time and state
                """
                nest.ResetNetwork()
                nest.SetKernelStatus({"time": 0.0})

            def evolve(
                event_times, event_channels, num_timesteps: Optional[int] = None
            ):
                """ IPC command running the network for num_timesteps with input_steps as input """

                if len(event_channels > 0):
                    # convert input index to NEST id
                    event_channels += np.min(self._sg)

                    # NEST time starts with 1 (not with 0)
                    nest.SetStatus(
                        self._sg,
                        [
                            {"spike_times": s2ms(event_times[event_channels == i])}
                            for i in self._sg
                        ],
                    )

                startTime = nest.GetKernelStatus("time")

                if startTime == 0:
                    # weird behavior of NEST; the recording stops a timestep before the simulation stops. Therefore
                    # the recording has one entry less in the first batch
                    nest.Simulate(num_timesteps * self.dt + 1.0)
                else:
                    nest.Simulate(num_timesteps * self.dt)

                # - record states
                if self.record:
                    events = nest.GetStatus(self._mm, "events")[0]
                    use_event = events["times"] >= startTime

                    senders = events["senders"][use_event]
                    times = events["times"][use_event]
                    vms = events["V_m"][use_event]

                    record_states = []
                    u_senders = np.unique(senders)
                    for i, nid in enumerate(u_senders):
                        ind = np.where(senders == nid)[0]
                        _times = times[ind]
                        order = np.argsort(_times)
                        _vms = vms[ind][order]
                        record_states.append(_vms)

                    record_states = np.array(record_states)

                # - Build response TimeSeries
                events = nest.GetStatus(self._sd, "events")[0]
                use_event = events["times"] >= startTime
                event_time_out = ms2s(events["times"][use_event])
                event_channel_out = events["senders"][use_event]

                # sort spiking response
                order = np.argsort(event_time_out)
                event_time_out = event_time_out[order]
                event_channel_out = event_channel_out[order]

                # transform from NEST id to index
                event_channel_out -= np.min(self._pop)

                if self.record:
                    return [event_time_out, event_channel_out, mV2V(record_states)]
                else:
                    return [event_time_out, event_channel_out, None]

            IPC_switcher = {
                COMMAND_GET: get_param,
                COMMAND_SET: set_param,
                COMMAND_RESET: reset,
                COMMAND_EVOLVE: evolve,
            }

            # wait for an IPC command

            while True:
                req = self.request_q.get()

                func = IPC_switcher.get(req[0])

                result = func(*req[1:])

                if result is not None:
                    self.result_q.put(result)

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

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
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
        )
        self._spike_adapt = self._expand_to_net_size(
            spike_adapt, "spike_adapt", allow_none=False
        )
        self._delta_t = self._expand_to_net_size(delta_t, "delta_t", allow_none=False)
        self._tau_adapt = self._expand_to_net_size(
            tau_adapt, "tau_adapt", allow_none=False
        )

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

    # @tau_mem.setter
    # def tau_mem(self, new_tau_mem):
    #     self._tau_mem = new_tau_mem

    #     if type(new_tau_mem) is np.ndarray:
    #         gls = []
    #         for nid in range(self.new_tau_mem):
    #             if type(self.capacity) == np.ndarray:
    #                 gls.append(self.capacity[nid] / self.tau_mem[nid])
    #             else:
    #                 gls.append(self.capacity / self.tau_mem[nid])
    #     else:
    #         if type(self.capacity) == np.ndarray:
    #             gls = []
    #             for nid in range(self.capacity):
    #                 gls.append(self.capacity[nid] / self.tau_mem)
    #         else:
    #             gls = self.capacity / self.tau_mem

    #     self.request_q.put([COMMAND_SET, "g_L", s2ms(gls)])

    @property
    def subthresh_adapt(self):
        return self._subthresh_adapt

    @subthresh_adapt.setter
    def subthresh_adapt(self, new_a):
        new_a = self._expand_to_net_size(new_a, "subthresh_adapt", allow_none=False)
        self._subthresh_adapt = new_a
        self.request_q.put([COMMAND_SET, "a", new_a])

    @property
    def spike_adapt(self):
        return self._spike_adapt

    @spike_adapt.setter
    def spike_adapt(self, new_b):
        new_b = self._expand_to_net_size(new_b, "spike_adapt", allow_none=False)
        self._spike_adapt = new_b
        self.request_q.put([COMMAND_SET, "b", new_b])

    @property
    def delta_t(self):
        return self._delta_t

    @delta_t.setter
    def delta_t(self, new_delta_t):
        new_delta_t = self._expand_to_net_size(new_delta_t, "delta_t", allow_none=False)
        self.delta_t = new_delta_t
        self.request_q.put([COMMAND_SET, "Delta_T", new_delta_t])

    @property
    def tau_adapt(self):
        return self._tau_adapt

    @tau_adapt.setter
    def tau_adapt(self, new_tau):
        new_tau = self._expand_to_net_size(new_tau, "tau_adapt", allow_none=False)
        self._tau_adapt = new_tau
        self.request_q.put([COMMAND_SET, "tau_w", s2ms(new_tau)])
