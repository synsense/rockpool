import json

import numpy as np
from ...timeseries import TSContinuous, TSEvent
import multiprocessing
import importlib

from ..layer import Layer


from typing import Optional, Union
import time

if importlib.util.find_spec("nest") is None:
    raise ModuleNotFoundError("No module named 'nest'.")


def s2ms(t):
    return t * 1000.0


def ms2s(t):
    return t / 1000.0


def V2mV(v):
    return v * 1000.0


def mV2V(v):
    return v / 1000.0


COMMAND_GET = 0
COMMAND_SET = 1
COMMAND_RESET = 2
COMMAND_EVOLVE = 3


# - FFIAFNest- Class: define a spiking feedforward layer with spiking outputs
class FFIAFNest(Layer):
    """ FFIAFNest - Class: define a spiking feedforward layer with spiking outputs
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
            weights: np.ndarray,
            bias: Union[float, np.ndarray],
            dt: float,
            tau_mem: Union[float, np.ndarray],
            capacity: Union[float, np.ndarray],
            v_thresh: Union[float, np.ndarray],
            v_reset: Union[float, np.ndarray],
            v_rest: Union[float, np.ndarray],
            refractory: Union[float, np.ndarray],
            record: bool = False,
            num_cores: int = 1,
        ):
            """ initialize the process"""

            multiprocessing.Process.__init__(self, daemon=True)

            self.request_q = request_q
            self.result_q = result_q

            # - Record neuron parameters
            self.dt = s2ms(dt)
            self.v_thresh = V2mV(v_thresh)
            self.v_reset = V2mV(v_reset)
            self.v_rest = V2mV(v_rest)
            self.tau_mem = s2ms(tau_mem)
            self.bias = V2mV(bias)
            self.capacity = capacity
            self.weights = V2mV(weights)
            self.refractory = s2ms(refractory)
            self.record = record
            self.size = np.shape(weights)[1]
            self.num_cores = num_cores

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
                    "print_time": False,
                }
            )

            self._pop = nest.Create("iaf_psc_exp", self.size)

            params = []
            for n in range(self.size):
                p = {}

                if type(self.tau_mem) is np.ndarray:
                    p["tau_m"] = self.tau_mem[n]
                else:
                    p["tau_m"] = self.tau_mem

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

                params.append(p)

            nest.SetStatus(self._pop, params)

            # - Add spike detector to record layer outputs
            self._sd = nest.Create("spike_detector")
            nest.Connect(self._pop, self._sd)

            # - Add stimulation device
            self._scg = nest.Create("step_current_generator", self.weights.shape[0])
            nest.Connect(self._scg, self._pop, "all_to_all", {"weight": self.weights.T})

            if self.record:
                # - Monitor for recording network potential
                nest.SetDefaults("multimeter", {"interval": self.dt})
                self._mm = nest.Create(
                    "multimeter", 1, {"record_from": ["V_m"], "interval": self.dt}
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

            def evolve(time_base, input_steps, num_timesteps: Optional[int] = None):
                """ IPC command running the network for num_timesteps with input_steps as input """

                # NEST time starts with 1 (not with 0)

                time_base = s2ms(time_base) + 1

                nest.SetStatus(
                    self._scg,
                    [
                        {
                            "amplitude_times": time_base,
                            "amplitude_values": V2mV(input_steps[:, i]),
                        }
                        for i in range(len(self._scg))
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

                if result is None:
                    self.result_q.put(result)

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: Union[float, np.ndarray] = 0.0,
        dt: float = 0.0001,
        tau_mem: Union[float, np.ndarray] = 0.02,
        capacity: Union[float, np.ndarray] = None,
        v_thresh: Union[float, np.ndarray] = -0.055,
        v_reset: Union[float, np.ndarray] = -0.065,
        v_rest: Union[float, np.ndarray] = -0.065,
        refractory: Union[float, np.ndarray] = 0.001,
        name: str = "unnamed",
        record: bool = False,
        num_cores=1,
    ):
        """
        FFIAFNest - Construct a spiking feedforward layer with IAF neurons, with a NEST back-end
                     Inputs are continuous currents; outputs are spiking events

        :param weights:             np.array MxN weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10mA

        :param dt:             float Time-step. Default: 0.1 ms

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param capacity:       np.array Nx1 vector of neuron membrance capacity. Default: 100 pF

        :param v_thresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param v_reset:        np.array Nx1 vector of neuron reset potential. Default: -65mV
        :param v_rest:         np.array Nx1 vector of neuron resting potential. Default: -65mV

        :param refractory: float Refractory period after each spike. Default: 0ms

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions
        """

        # - Call super constructor
        super().__init__(weights=weights, dt=dt, name=name)

        # - Convert parameters to arrays before passing to nest process
        v_thresh = self._expand_to_net_size(v_thresh, "v_thresh", allow_none=False)
        v_reset = self._expand_to_net_size(v_reset, "v_reset", allow_none=False)
        v_rest = self._expand_to_net_size(v_rest, "v_rest", allow_none=False)
        tau_mem = self._expand_to_net_size(tau_mem, "tau_mem", allow_none=False)
        bias = self._expand_to_net_size(bias, "bias", allow_none=False)
        refractory = self._expand_to_net_size(
            refractory, "refractory", allow_none=False
        )
        # Set capacity to the membrane time constant to be consistent with other layers
        if capacity is None:
            capacity = tau_mem * 1000.0
        else:
            capacity = self._expand_to_net_size(capacity, "capacity", allow_none=False)

        # set capacity to the membrane time constant to be consistent with other layers
        if capacity is None:
            capacity = tau_mem * 1000.0

        self.num_cores = num_cores

        self.request_q = multiprocessing.Queue()
        self.result_q = multiprocessing.Queue()

        self.nest_process = self.NestProcess(
            self.request_q,
            self.result_q,
            self._weights,
            bias,
            dt,
            tau_mem,
            capacity,
            v_thresh,
            v_reset,
            v_rest,
            refractory,
            record,
            num_cores,
        )
        self.nest_process.start()

        # - Record neuron parameters
        self._v_thresh = v_thresh
        self._v_reset = v_reset
        self._v_rest = v_rest
        self._tau_mem = tau_mem
        self._bias = bias
        self._capacity = capacity
        self._refractory = refractory
        self.record = record

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """

        self.request_q.put([COMMAND_SET, "V_m", V2mV(self._v_rest)])

    def randomize_state(self):
        """ .randomize_state() - arguments:: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        v_range = abs(self._v_thresh - self._v_reset)
        randV = np.random.rand(self._size) * v_range + self._v_reset

        self.request_q.put([COMMAND_SET, "V_m", V2mV(randV)])

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        print("WARNING: This function resets the whole network")

        self.request_q.put([COMMAND_RESET])
        self._timestep = 0

    def reset_all(self):
        """
        reset_all - resets time and state
        """

        self.request_q.put([COMMAND_RESET])
        self._timestep = 0

    # --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:                TSEvent  output spike series

        """
        # - Prepare time base
        time_base, input_steps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        self.request_q.put([COMMAND_EVOLVE, time_base, input_steps, num_timesteps])

        if self.record:
            event_time_out, event_channel_out, self.record_states = self.result_q.get()
        else:
            event_time_out, event_channel_out, _ = self.result_q.get()

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # - Update layer time step
        self._timestep += num_timesteps

        return TSEvent(
            np.clip(event_time_out, t_start, t_stop),
            event_channel_out,
            name="Layer spikes",
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

    def terminate(self):
        self.request_q.close()
        self.result_q.close()
        self.request_q.cancel_join_thread()
        self.result_q.cancel_join_thread()
        self.nest_process.terminate()
        self.nest_process.join()

    def to_dict(self):

        config = {}
        config["name"] = self.name
        config["weights_in"] = self.weights.tolist()
        config["dt"] = self.dt
        config["v_thresh"] = self.v_thresh.tolist()
        config["v_reset"] = self.v_reset.tolist()
        config["v_rest"] = self.v_rest.tolist()
        config["capacity"] = self.capacity.tolist()
        config["refractory"] = self.refractory.tolist()
        config["tau_mem"] = self.tau_mem.tolist()
        config["num_cores"] = self.num_cores
        config["record"] = self.record
        config["bias"] = self.bias.tolist()
        config["class_name"] = "FFIAFNest"

        return config

    def save(self, config, filename):
        with open(filename, "w") as f:
            json.dump(config, f)

    @staticmethod
    def load_from_dict(config):

        return FFIAFNest(
            weights=config["weights_in"],
            bias=config["bias"],
            dt=config["dt"],
            tau_mem=config["tau_mem"],
            capacity=config["capacity"],
            v_thresh=config["v_thresh"],
            v_reset=config["v_reset"],
            v_rest=config["v_rest"],
            refractory=config["refractory"],
            name=config["name"],
            record=config["record"],
            num_cores=config["num_cores"],
        )

    @staticmethod
    def load_from_file(filename):
        with open(filename, "r") as f:
            config = json.load(f)

        return FFIAFNest(
            weights=config["weights_in"],
            bias=config["bias"],
            dt=config["dt"],
            tau_mem=config["tau_mem"],
            capacity=config["capacity"],
            v_thresh=config["v_thresh"],
            v_reset=config["v_reset"],
            v_rest=config["v_rest"],
            refractory=config["refractory"],
            name=config["name"],
            record=config["record"],
            num_cores=config["num_cores"],
        )

    ### --- Properties

    @property
    def output_type(self):
        return TSEvent

    @property
    def refractory(self):
        return self._refractory

    @refractory.setter
    def refractory(self, new_refractory):
        new_refractory = self._expand_to_net_size(
            new_refractory, "refractory", allow_none=False
        )
        self._refractory = new_refractory
        self.request_q.put([COMMAND_SET, "t_ref", s2ms(new_refractory)])

    @property
    def capacity(self):
        return self._capacity

    @property
    def state(self):
        self.request_q.put([COMMAND_GET, "V_m"])
        vms = np.array(self.result_q.get())
        return mV2V(vms)

    @state.setter
    def state(self, new_state):
        new_state = self._expand_to_net_size(new_state, "state", allow_none=False)
        self.request_q.put([COMMAND_SET, "V_m", V2mV(new_state)])

    @property
    def tau_mem(self):
        return self._tau_mem

    @tau_mem.setter
    def tau_mem(self, new_tau_mem):
        new_tau_mem = self._expand_to_net_size(new_tau_mem, "tau_mem", allow_none=False)
        self._tau_mem = new_tau_mem
        self.request_q.put([COMMAND_SET, "tau_m", s2ms(new_tau_mem)])

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        new_bias = self._expand_to_net_size(new_bias, "bias", allow_none=False)
        self._bias = new_bias
        self.request_q.put([COMMAND_SET, "I_e", V2mV(new_bias)])

    @property
    def v_thresh(self):
        return self._v_thresh

    @v_thresh.setter
    def v_thresh(self, new_v_thresh):
        new_v_thresh = self._expand_to_net_size(
            new_v_thresh, "v_thresh", allow_none=False
        )
        self._v_thresh = new_v_thresh
        self.request_q.put([COMMAND_SET, "V_th", V2mV(new_v_thresh)])

    @property
    def v_reset(self):
        return self._v_reset

    @v_reset.setter
    def v_reset(self, new_v_reset):
        new_v_reset = self._expand_to_net_size(new_v_reset, "v_reset", allow_none=False)
        self._v_reset = new_v_reset
        self.request_q.put([COMMAND_SET, "V_reset", V2mV(new_v_reset)])

    @property
    def v_rest(self):
        return self._v_rest

    @v_rest.setter
    def v_rest(self, new_v_rest):
        new_v_rest = self._expand_to_net_size(new_v_rest, "v_rest", allow_none=False)
        self._v_rest = new_v_rest
        self.request_q.put([COMMAND_SET, "E_L", V2mV(new_v_rest)])

    @property
    def t(self):
        return self._timestep * self.dt

    @Layer.dt.setter
    def dt(self, _):
        raise ValueError("The `dt` property cannot be set for this layer")


# - RecIAFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
class RecIAFSpkInNest(Layer):
    """ RecIAFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
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
            refractory: Union[float, np.ndarray],
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
                    "print_time": False,
                }
            )

            self._pop = nest.Create("iaf_psc_exp", self.size)

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
                    p["tau_m"] = self.tau_mem[n]
                else:
                    p["tau_m"] = self.tau_mem

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
                    "multimeter", 1, {"record_from": ["V_m"], "interval": self.dt}
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
                    nest.Simulate((num_timesteps + 1) * self.dt)
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
        name: str = "unnamed",
        record: bool = False,
        num_cores: int = 1,
    ):
        """
        RecIAFSpkInNest - Construct a spiking recurrent layer with IAF neurons, with a NEST back-end
                           in- and outputs are spiking events

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param bias:          np.array Nx1 bias vector. Default: 10.5mA

        :param dt:             float Time-step. Default: 0.1 ms

        :param tau_mem:          np.array Nx1 vector of neuron time constants. Default: 20ms
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

        :param name:         str Name for the layer. Default: 'unnamed'

        :param record:         bool Record membrane potential during evolutions
        """

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(weights=weights_in, dt=dt, name=name)
        self._weights_in = self.weights

        if tau_syn_exc is None:
            tau_syn_exc = tau_syn
        if tau_syn_inh is None:
            tau_syn_inh = tau_syn

        # - Convert parameters to arrays before passing to nest process
        self._weights_rec = self._expand_to_shape(
            weights_rec, (self.size, self.size), "weights_rec", allow_none=False
        )
        self._v_thresh = self._expand_to_net_size(
            v_thresh, "v_thresh", allow_none=False
        )
        self._v_reset = self._expand_to_net_size(v_reset, "v_reset", allow_none=False)
        self._v_rest = self._expand_to_net_size(v_rest, "v_rest", allow_none=False)
        self._tau_mem = self._expand_to_net_size(tau_mem, "tau_mem", allow_none=False)
        self._bias = self._expand_to_net_size(bias, "bias", allow_none=False)
        self._refractory = self._expand_to_net_size(
            refractory, "refractory", allow_none=False
        )
        self._tau_syn_exc = self._expand_to_net_size(
            tau_syn_exc, "tau_syn_exc", allow_none=False
        )
        self._tau_syn_inh = self._expand_to_net_size(
            tau_syn_inh, "tau_syn_inh", allow_none=False
        )
        self._delay_in = self._expand_to_shape(
            delay_in, (self.size_in, self.size), "delay_in", allow_none=False
        )
        self._delay_rec = self._expand_to_shape(
            delay_rec, (self.size, self.size), "delay_rec", allow_none=False
        )
        # Set capacity to the membrane time constant to be consistent with other layers
        if capacity is None:
            self._capacity = self._tau_mem * 1000.0
        else:
            self._capacity = self._expand_to_net_size(
                capacity, "capacity", allow_none=False
            )

        # - Record settings
        self._record = record
        self._num_cores = num_cores

        self._setup_nest()

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
        )
        self.nest_process.start()

    def reset_state(self):
        """ .reset_state() - arguments:: reset the internal state of the layer
            Usage: .reset_state()
        """

        self.request_q.put([COMMAND_SET, "V_m", V2mV(self.v_rest)])

    def randomize_state(self):
        """ .randomize_state() - arguments:: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        v_range = abs(self._v_thresh - self._v_reset)
        randV = np.random.rand(self.size) * v_range + self._v_reset

        self.request_q.put([COMMAND_SET, "V_m", V2mV(randV)])

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        print("WARNING: This function resets the whole network")

        self.request_q.put([COMMAND_RESET])
        self._timestep = 0

    def reset_all(self):
        """
        reset_all - resets time and state
        """

        self.request_q.put([COMMAND_RESET])
        self._timestep = 0

    # --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param ts_input:       TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:                TSEvent  output spike series

        """

        # - Prepare time base
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        time_base = self._gen_time_trace(self.t, num_timesteps)

        # - Set spikes for spike generator
        if ts_input is not None:
            event_times, event_channels = ts_input(
                time_base[0], time_base[-1] + self.dt
            )
            # - Round event times to time base
            event_timesteps = np.floor(event_times / self.dt)
            event_times = event_timesteps * self.dt

        else:
            event_times = np.array([])
            event_channels = np.array([])

        self.request_q.put([COMMAND_EVOLVE, event_times, event_channels, num_timesteps])

        if self.record:
            event_time_out, event_channel_out, self.record_states = self.result_q.get()
        else:
            event_time_out, event_channel_out, _ = self.result_q.get()

        # - Start and stop times for output time series
        t_start = self._timestep * self.dt
        t_stop = (self._timestep + num_timesteps) * self.dt

        # - Update layer time step
        self._timestep += num_timesteps

        return TSEvent(
            np.clip(event_time_out, t_start, t_stop),
            event_channel_out,
            name="Layer spikes",
            num_channels=self.size,
            t_start=t_start,
            t_stop=t_stop,
        )

    def terminate(self):
        self.request_q.close()
        self.result_q.close()
        self.request_q.cancel_join_thread()
        self.result_q.cancel_join_thread()
        self.nest_process.terminate()
        self.nest_process.join()

    def to_dict(self):

        config = {}
        config["name"] = self.name
        config["weights_in"] = self._weights_in.tolist()
        config["weights_rec"] = self._weights_rec.tolist()

        config["delay_in"] = self._delay_in.tolist()
        config["delay_rec"] = self._delay_rec.tolist()

        config["bias"] = self.bias.tolist()
        config["dt"] = self.dt
        config["v_thresh"] = self.v_thresh.tolist()
        config["v_reset"] = self.v_reset.tolist()
        config["v_rest"] = self.v_rest.tolist()
        config["capacity"] = self.capacity.tolist()
        config["refractory"] = self.refractory.tolist()
        config["num_cores"] = self.num_cores
        config["tau_mem"] = self.tau_mem.tolist()
        config["tau_syn_exc"] = self.tau_syn_exc.tolist()
        config["tau_syn_inh"] = self.tau_syn_inh.tolist()
        config["record"] = self.record
        config["class_name"] = "RecIAFSpkInNest"

        return config

    @classmethod
    def load_from_dict(cls, config):

        lyr = super().load_from_dict(config)
        lyr.reset_all()
        return lyr

    @classmethod
    def load_from_file(cls, filename: str):
        lyr = super().load_from_file(filename)
        lyr.reset_all()
        return lyr

    ### --- Properties

    @property
    def input_type(self):
        return TSEvent

    @property
    def output_type(self):
        return TSEvent

    @property
    def weights_in(self):
        return self._weights_in

    @property
    def weights_rec(self):
        return self._weights_rec

    @property
    def capacity(self):
        return self._capacity

    @property
    def refractory(self):
        return self._refractory

    @refractory.setter
    def refractory(self, new_refractory):
        new_refractory = self._expand_to_net_size(
            new_refractory, "refractory", allow_none=False
        )
        new_refractory = new_refractory.astype(float)
        self._refractory = new_refractory
        self.request_q.put([COMMAND_SET, "t_ref", s2ms(new_refractory)])

    @property
    def state(self):
        time.sleep(0.1)
        self.request_q.put([COMMAND_GET, "V_m"])
        vms = np.array(self.result_q.get())
        return mV2V(vms)

    @state.setter
    def state(self, new_state):
        new_state = self._expand_to_net_size(new_state, "state", allow_none=False)
        self.request_q.put([COMMAND_SET, "V_m", V2mV(new_state.astype(float))])

    @property
    def delay_in(self):
        return self._delay_in

    @delay_in.setter
    def delay_in(self, new_delay_in):
        new_delay_in = self._expand_to_shape(
            new_delay_in, (self.size_in, self.size), "delay_in", allow_none=False
        )
        new_delay_in = new_delay_in.astype(float)
        self._delay_in = new_delay_in
        self.request_q.put([COMMAND_SET, "delay", s2ms(new_delay_in)])

    @property
    def delay_rec(self):
        return self._delay_rec

    @delay_rec.setter
    def delay_rec(self, new_delay_rec):
        new_delay_rec = self._expand_to_shape(
            new_delay_rec, (self.size_in, self.size), "delay_rec", allow_none=False
        )
        new_delay_rec = new_delay_rec.astype(float)
        self._delay_rec = new_delay_rec
        self.request_q.put([COMMAND_SET, "delay", s2ms(new_delay_rec)])

    @property
    def tau_mem(self):
        return self._tau_mem

    @tau_mem.setter
    def tau_mem(self, new_tau_mem):
        new_tau_mem = self._expand_to_net_size(new_tau_mem, "tau_mem", allow_none=False)
        new_tau_mem = new_tau_mem.astype(float)
        self._tau_mem = new_tau_mem
        self.request_q.put([COMMAND_SET, "tau_m", s2ms(new_tau_mem)])

    @property
    def tau_syn(self):
        return self._tau_syn

    @tau_syn.setter
    def tau_syn(self, new_tau_syn):
        new_tau_syn = self._expand_to_net_size(new_tau_syn, "tau_syn", allow_none=False)
        new_tau_syn = new_tau_syn.astype(float)
        self._tau_syn = new_tau_syn
        self._tau_syn_inh = new_tau_syn
        self._tau_syn_exc = new_tau_syn
        self.request_q.put([COMMAND_SET, "tau_syn_ex", s2ms(new_tau_syn)])
        self.request_q.put([COMMAND_SET, "tau_syn_in", s2ms(new_tau_syn)])

    @property
    def tau_syn_exc(self):
        return self._tau_syn_exc

    @tau_syn_exc.setter
    def tau_syn_exc(self, new_tau_syn_exc):
        new_tau_syn_exc = self._expand_to_net_size(
            new_tau_syn_exc, "tau_syn_exc", allow_none=False
        )
        new_tau_syn_exc = new_tau_syn_exc.astype(float)
        self._tau_syn_exc = new_tau_syn_exc
        self._tau_syn = None
        self.request_q.put([COMMAND_SET, "tau_syn_ex", s2ms(new_tau_syn_exc)])

    @property
    def tau_syn_inh(self):
        return self._tau_syn_inh

    @tau_syn_inh.setter
    def tau_syn_inh(self, new_tau_syn_inh):
        new_tau_syn_inh = self._expand_to_net_size(
            new_tau_syn_inh, "tau_syn_inh", allow_none=False
        )
        new_tau_syn_inh = new_tau_syn_inh.astype(float)
        self._tau_syn_inh = new_tau_syn_inh
        self._tau_syn = None
        self.request_q.put([COMMAND_SET, "tau_syn_in", s2ms(new_tau_syn_inh)])

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, new_bias):
        new_bias = self._expand_to_net_size(new_bias, "bias", allow_none=False)
        new_bias = new_bias.astype(float)
        self._bias = new_bias
        self.request_q.put([COMMAND_SET, "I_e", V2mV(new_bias)])

    @property
    def v_thresh(self):
        return self._v_thresh

    @v_thresh.setter
    def v_thresh(self, new_v_thresh):
        new_v_thresh = self._expand_to_net_size(
            new_v_thresh, "v_thresh", allow_none=False
        )
        new_v_thresh = new_v_thresh.astype(float)
        self._v_thresh = new_v_thresh
        self.request_q.put([COMMAND_SET, "V_th", V2mV(new_v_thresh)])

    @property
    def v_reset(self):
        return self._v_reset

    @v_reset.setter
    def v_reset(self, new_v_reset):
        new_v_reset = self._expand_to_net_size(new_v_reset, "v_reset", allow_none=False)
        new_v_reset = new_v_reset.astype(float)
        self._v_reset = new_v_reset
        self.request_q.put([COMMAND_SET, "V_reset", V2mV(new_v_reset)])

    @property
    def v_rest(self):
        return self._v_reset

    @v_rest.setter
    def v_rest(self, new_v_rest):
        new_v_rest = self._expand_to_net_size(new_v_rest, "v_rest", allow_none=False)
        new_v_rest = new_v_rest.astype(float)
        self._v_rest = new_v_rest
        self.request_q.put([COMMAND_SET, "E_L", V2mV(new_v_rest)])

    @property
    def t(self):
        return self._timestep * self.dt

    @Layer.dt.setter
    def dt(self):
        raise ValueError("The `dt` property cannot be set for this layer")

    @property
    def record(self):
        return self._record

    @property
    def num_cores(self):
        return self._num_cores
