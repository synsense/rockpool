"""
IAF neuron layers with a NEST backend
"""


import multiprocessing

import numpy as np

from typing import Optional, Union, List, Dict, Tuple
from warnings import warn

# - Define a FloatVector type
FloatVector = Union[float, np.ndarray]


from rockpool.timeseries import TSContinuous, TSEvent
from rockpool.utilities.property_arrays import SetterArray, ImmutableArray
from rockpool.nn.layers.layer import Layer
from rockpool.nn.modules.timed_module import astimedmodule

# - Unit conversion functions
U2mU = lambda x: x * 1e3
mU2U = lambda x: x / 1e3

s2ms = U2mU
ms2s = mU2U

V2mV = U2mU
mV2V = mU2U

A2mA = U2mU
mA2A = mU2U

uA2nA = U2mU
nA2uA = mU2U

F2mF = U2mU
mF2F = mU2U

COMMAND_GET = 0
COMMAND_SET = 1
COMMAND_RESET = 2
COMMAND_EVOLVE = 3
COMMAND_EXEC = 4


class _BaseNestProcess(multiprocessing.Process):
    """Base Class for running NEST in its own process"""

    def __init__(
        self,
        request_q,
        result_q,
        model: str,
        bias: np.ndarray,
        dt: float,
        capacity: np.ndarray,
        v_thresh: np.ndarray,
        v_reset: np.ndarray,
        v_rest: np.ndarray,
        refractory: np.ndarray,
        record: bool = False,
        num_cores: int = 1,
    ):
        """initializes the process"""

        multiprocessing.Process.__init__(self, daemon=True)

        self.request_q = request_q
        self.result_q = result_q

        # - Record neuron parameters
        self.dt = s2ms(dt)
        self.v_thresh = V2mV(v_thresh)
        self.v_reset = V2mV(v_reset)
        self.v_rest = V2mV(v_rest)
        self.bias = A2mA(bias)
        self.capacity = F2mF(capacity)
        self.refractory = s2ms(refractory)
        self.record = record
        self.num_cores = num_cores
        self.model = model
        self.v_init = V2mV(v_rest)

    ######### DEFINE IPC COMMANDS ######

    def nest_exec(self, command):
        return exec(command)

    def get_param(self, name):
        """IPC command for getting a parameter"""
        vms = self.nest_module.GetStatus(self._pop, name)
        return vms

    def set_param(self, name, value):
        """IPC command for setting a parameter"""
        try:
            params = [{name: val} for val in value[: self.size]]
        except TypeError:
            params = [{name: value} for _ in range(self.size)]
        self.nest_module.SetStatus(self._pop, params)

    def reset(self):
        """
        IPC command which resets time and state
        """
        self.nest_module.ResetNetwork()
        self.nest_module.SetKernelStatus({"time": 0.0})

    def record_states(self, t_start: float) -> np.ndarray:
        """
        Record neuron states over time

        :param float t_start:  Time from which on events should be recorded.

        :return: 2D-Array of recorded neuron states
        """
        events = self.nest_module.GetStatus(self._mm, "events")[0]
        use_event = events["times"] >= t_start

        senders = events["senders"][use_event]
        times = events["times"][use_event]
        vms = events["V_m"][use_event]

        recorded_states = []
        u_senders = np.unique(senders)
        for i, nid in enumerate(u_senders):
            ind = np.where(senders == nid)[0]
            _times = times[ind]
            order = np.argsort(_times)
            _vms = vms[ind][order]

            if t_start == 0:
                # weird behavior of NEST; the recording stops a timestep before the simulation stops. Therefore
                # the recording has one entry less in the first batch
                recorded_states.append(np.insert(_vms, 0, self.v_init[i]))
            else:
                recorded_states.append(_vms)

        return np.array(recorded_states)

    def evolve_nest(
        self, num_timesteps: int
    ) -> (np.ndarray, np.ndarray, Union[np.ndarray, None]):
        """
        Evolve state of nest simulation by defined number of timesteps

        :param int num_timesteps:  Number of timesteps over which to evolve

        :return (np.ndarray, np.ndarray, Union[np.ndarray, None]): event_time_out, event_channel_out, recorded_states
            1D-array of recorded event times
            1D-array of recorded event channels
            If ``self.record``: 2D-array of recorded neuron states, otherwise ``None``
        """
        t_start = self.nest_module.GetKernelStatus("time")

        self.nest_module.Simulate(num_timesteps * self.dt)

        # - Fetch events from spike detector
        events = self.nest_module.GetStatus(self._sd, "events")[0]

        # - Clear memory of spike detector to avoid accumulating past events
        self.nest_module.SetStatus(self._sd, {"n_events": 0})

        # - Process fetched events
        use_event = events["times"] >= t_start
        event_time_out = ms2s(events["times"][use_event])
        event_channel_out = events["senders"][use_event]

        # sort spiking response
        order = np.argsort(event_time_out)
        event_time_out = event_time_out[order]
        event_channel_out = event_channel_out[order]

        # transform from NEST id to index
        event_channel_out -= np.min(self._pop)

        # - record states
        if self.record:
            recorded_states = self.record_states(t_start)
            return [event_time_out, event_channel_out, mV2V(recorded_states)]
        else:
            return [event_time_out, event_channel_out, None]

    def evolve(
        self, event_times, event_channels, num_timesteps: int
    ) -> (np.ndarray, np.ndarray, Union[np.ndarray, None]):
        """
        Evolve state of nest simulation by defined number of timesteps

        :param event_times:         Only used in child classes
        :param event_channels:      Only used in child classes
        :param int num_timesteps:   Number of timesteps over which to evolve.

        :return (np.ndarray, np.ndarray, Union[np.ndarray, None]): event_time_out, event_channel_out, mV2V(recorded_states)
            1D-array of recorded event times
            1D-array of recorded event channels
            If ``self.record``: 2D-array of recorded neuron states, otherwise ``None``
        """
        return self.evolve_nest(num_timesteps)

    def read_weights(self, pop_pre: Tuple[int], pop_post: Tuple[int]):
        # - Read out connections and convert to array
        connections = self.nest_module.GetConnections(pop_pre, pop_post)
        if not connections:
            return np.zeros((len(pop_pre), len(pop_post)))
        ids_pre, ids_post = np.array(connections)[:, :2].T
        # - Map population IDs to indices starting from 0
        ids_pre -= pop_pre[0]
        ids_post -= pop_post[0]
        # - Read out weights from connections
        weights = self.nest_module.GetStatus(connections, "weight")
        # - Generate weight matrix
        weight_array = np.zeros((len(pop_pre), len(pop_post)))
        weight_array[ids_pre, ids_post] = weights
        return weight_array

    def update_weights(
        self,
        pop_pre: tuple,
        pop_post: tuple,
        weights_new: np.ndarray,
        weights_old: np.ndarray,
        delays: Optional[np.ndarray] = None,
        connection_exists: Optional[np.ndarray] = None,
    ):
        """
        update_weights - Update nest connections and their weights
        :param pop_pre:             Presynaptic population
        :param pop_post:            Postsynaptic population
        :param weights_new:         New weights
        :param weights_old:         Old weights
        :param delays:              Delays corresponding to the connections
        :param connection_exists:   2D boolean array indicating which connections already exist
        """
        if connection_exists is None:
            connection_exists = False
            log_existing_conn = False
        else:
            log_existing_conn = True
        # - Connections that need to be updated
        update_connection = weights_old != weights_new
        # - Connections that exist already and whose weights need to be updated
        update_existing_conn = np.logical_and(update_connection, connection_exists)
        idcs_pre_upd, idcs_post_upd = np.where(update_existing_conn)
        # - First global ID of each population
        id_start_pre = pop_pre[0]
        id_start_post = pop_post[0]
        if idcs_pre_upd.size > 0:
            # - Extract existing connections that need to be updated
            existing_conns: Tuple = self.nest_module.GetConnections(
                list(np.asarray(pop_pre)[np.unique(idcs_pre_upd)]),
                list(np.asarray(pop_post)[np.unique(idcs_post_upd)]),
            )
            existing_conns = np.array(existing_conns)
            # - Indices of existing connections wrt weight matrix
            existing_pre, existing_post = existing_conns[:, :2].copy().T
            existing_pre -= id_start_pre
            existing_post -= id_start_post
            # - Dict for mapping from 2D array indices to connection
            map_2d_conn = {
                (idx_pre, idx_post): conn
                for idx_pre, idx_post, conn in zip(
                    existing_pre, existing_post, existing_conns
                )
            }
            # - Connections to be updated
            conns_to_update = [
                map_2d_conn[idcs] for idcs in zip(idcs_pre_upd, idcs_post_upd)
            ]
            new_weights = [
                {"weight": w} for w in weights_new[idcs_pre_upd, idcs_post_upd]
            ]
            # - Update weights
            self.nest_module.SetStatus(conns_to_update, new_weights)

        # - Connections that need to be created
        idcs_pre_new, idcs_post_new = np.where(
            np.logical_and(connection_exists == False, update_connection)
        )
        if idcs_pre_new.size > 0:
            if delays is not None:
                # delays = delays[idcs_pre_new, idcs_post_new]
                self.nest_module.Connect(
                    idcs_pre_new + id_start_pre,
                    idcs_post_new + id_start_post,
                    "one_to_one",
                    {
                        "weight": weights_new[idcs_pre_new, idcs_post_new],
                        # "delay": delays[idcs_pre_new, idcs_post_new],
                    },
                )
            else:
                self.nest_module.Connect(
                    idcs_pre_new + id_start_pre,
                    idcs_post_new + id_start_post,
                    "one_to_one",
                    {"weight": weights_new[idcs_pre_new, idcs_post_new]},
                )
        if log_existing_conn:
            # - Mark new connections as existing
            connection_exists[(idcs_pre_new, idcs_post_new)] = True

    def init_nest(self):
        """init_nest - Initialize nest"""

        #### INITIALIZE NEST ####
        import nest

        # - Make module accessible to class methods
        self.nest_module = nest

        self.nest_module.ResetKernel()
        self.nest_module.hl_api.set_verbosity("M_FATAL")
        try:
            self.nest_module.SetKernelStatus(
                {
                    "resolution": self.dt,
                    "local_num_threads": self.num_cores,
                    "print_time": False,
                }
            )
        except self.nest_module.pynestkernel.NESTError:
            raise ValueError("The provided value for `dt` is not supported.")

    def setup_nest_network(self):
        """
        setup_nest_objects - Generate nest objects (neurons, input generators,
                             monitors,...) and connect them.
        """
        self._pop = self.nest_module.Create(self.model, self.size)

        # - Add spike detector to record layer outputs
        self._sd = self.nest_module.Create("spike_detector")
        self.nest_module.Connect(self._pop, self._sd)

        # - Set parameters
        param_list: List[Dict[str, np.ndarray]] = self.generate_nest_params_list()
        self.nest_module.SetStatus(self._pop, param_list)

        # - Set connections with weights and delays
        self.set_all_connections()

        if self.record:
            # - Monitor for recording network potential
            self.nest_module.SetDefaults("multimeter", {"interval": self.dt})
            self._mm = self.nest_module.Create(
                "multimeter", 1, {"record_from": ["V_m"], "interval": self.dt}
            )
            self.nest_module.Connect(self._mm, self._pop)

    def generate_nest_params_list(self) -> List[Dict[str, np.ndarray]]:
        """init_nest_params - Initialize nest neuron parameters and return as list"""
        params = []
        for n in range(self.size):
            p = {}

            p["V_th"] = self.v_thresh[n]
            p["V_reset"] = self.v_reset[n]
            p["E_L"] = self.v_rest[n]
            p["V_m"] = self.v_rest[n]
            p["t_ref"] = self.refractory[n]
            p["I_e"] = self.bias[n]
            p["C_m"] = self.capacity[n]

            params.append(p)

        return params

    def set_all_connections(self):
        """
        set_all_connections - To be used for setting up layer connections. Is called
                              from `self.setup_nest_network'.
        """
        pass

    def set_connections(
        self,
        pop_pre: tuple,
        pop_post: tuple,
        weights: np.ndarray,
        delays: Optional[np.ndarray] = None,
        connection_exists: Optional[np.ndarray] = None,
    ):
        """
        set_connections - Set connections between two neuron groups.
        :param tuple pop_pre:                           Presynaptic (nest) population
        :param tuple pop_post:                          Postsynaptic (nest) population
        :param np.ndarray weights:                      2D-array of weights to be set.
        :param Optional[np.ndarray] delays:             If not ``None``: 2D-array of delays to be set.
        :param Optional[np.ndarray] connection_exists:  If not ``None``, 2D boolean array to indicate which connections exist. Will be set to ``True`` for non-zero connections that are set here.
        """
        # - Indices of pre- and postsynaptic nonzero weights
        idcs_pre, idcs_post = np.nonzero(weights)
        # - Global IDs of pre- and postsynaptic neurons
        pres = np.asarray(pop_pre)[idcs_pre]
        posts = np.asarray(pop_post)[idcs_post]
        # - Weights corresponding to the neuron-pairs
        weight_list = weights[idcs_pre, idcs_post]

        if len(weight_list) > 0:
            if delays is not None:
                # - Create connections, set weights and delays
                delays = np.clip(delays[idcs_pre, idcs_post], self.dt, None)
                # delays = np.clip(delay_list, self.dt, np.max(delay_list))
                self.nest_module.Connect(
                    pres, posts, "one_to_one", {"weight": weight_list, "delay": delays}
                )
            else:
                # - Create connections, set weights
                self.nest_module.Connect(
                    pres, posts, "one_to_one", {"weight": weight_list}
                )

        if connection_exists is not None:
            # - Mark new connections as existing
            connection_exists[(idcs_pre, idcs_post)] = True

    def run(self):
        """start the process. Initializes the network, defines IPC commands and waits for commands."""

        self.init_nest()
        self.setup_nest_network()

        IPC_switcher = {
            COMMAND_GET: self.get_param,
            COMMAND_SET: self.set_param,
            COMMAND_RESET: self.reset,
            COMMAND_EVOLVE: self.evolve,
            COMMAND_EXEC: self.nest_exec,
        }

        # wait for an IPC command

        while True:
            req = self.request_q.get()
            func = IPC_switcher.get(req[0])
            result = func(*req[1:])
            if req[0] in [COMMAND_EXEC, COMMAND_GET, COMMAND_EVOLVE]:
                self.result_q.put(result)


class _BaseNestProcessSpkInRec(_BaseNestProcess):
    """Class for running NEST in its own process"""

    def __init__(
        self,
        request_q,
        result_q,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        delay_in: FloatVector,
        delay_rec: FloatVector,
        bias: FloatVector,
        dt: float,
        model: str,
        tau_syn_exc: FloatVector,
        tau_syn_inh: FloatVector,
        capacity: FloatVector,
        v_thresh: FloatVector,
        v_reset: FloatVector,
        v_rest: FloatVector,
        refractory: FloatVector,
        record: bool = False,
        num_cores: int = 1,
    ):
        """initializes the process"""
        super().__init__(
            request_q=request_q,
            result_q=result_q,
            bias=bias,
            dt=dt,
            capacity=capacity,
            v_thresh=v_thresh,
            v_reset=v_reset,
            v_rest=v_rest,
            refractory=refractory,
            record=record,
            num_cores=num_cores,
            model=model,
        )

        # - Record weights and layer-specific parameters
        self.weights_in = A2mA(weights_in)
        self.weights_rec = A2mA(weights_rec)
        self.size = np.shape(weights_in)[1]
        self.delay_in = s2ms(delay_in)
        self.delay_rec = s2ms(delay_rec)
        self.tau_syn_exc = s2ms(tau_syn_exc)
        self.tau_syn_inh = s2ms(tau_syn_inh)
        # - Keep track of existing connections for more efficient weight updates
        self.connection_rec_exists = np.zeros_like(self.weights_rec, bool)
        self.connection_in_exists = np.zeros_like(self.weights_in, bool)

    ######### DEFINE IPC COMMANDS ######

    def get_param(self, name):
        """IPC command for getting a parameter"""
        if name == "weights_in":
            vms = self.read_weights(pop_pre=self._sg, pop_post=self._pop)
        elif name == "weights_rec":
            vms = self.read_weights(pop_pre=self._pop, pop_post=self._pop)
        else:
            try:
                vms = self.nest_module.GetStatus(self._pop, name)
            except self.nest_module.kernel.NESTError:
                # - Catch 'DictErrors' from nest that sometimes occur with
                #   auto-completion in iPython when typing "wei" + <TAB>
                return
        return vms

    def set_param(self, name, value):
        """IPC command for setting a parameter"""

        if name == "weights_in":
            weights_old = self.weights_in.copy()
            self.weights_in = value
            self.update_weights(
                pop_pre=self._sg,
                pop_post=self._pop,
                weights_new=self.weights_in,
                weights_old=weights_old,
                delays=self.delay_in,
                connection_exists=self.connection_in_exists,
            )
        elif name == "weights_rec":
            weights_old = self.weights_rec.copy()
            self.weights_rec = value
            self.update_weights(
                pop_pre=self._pop,
                pop_post=self._pop,
                weights_new=self.weights_rec,
                weights_old=weights_old,
                delays=self.delay_rec,
                connection_exists=self.connection_rec_exists,
            )
        else:
            super().set_param(name, value)

    def evolve(
        self, event_times: np.ndarray, event_channels: np.ndarray, num_timesteps: int
    ) -> (np.ndarray, np.ndarray, Union[np.ndarray, None]):
        """
        Evolve state of nest simulation by defined number of timesteps

        :param np.ndarray event_times:      Input spike times
        :param np.ndarray event_channels:   Input spike channels
        :param int num_timesteps:           Number of timesteps over which to evolve.

        :return (np.ndarray, np.ndarray, Union[np.ndarray, None]: event_time_out, event_channel_out, mV2V(recorded_states)
            1D-array of recorded event times
            1D-array of recorded event channels
            If ``self.record``: 2D-array of recorded neuron states, otherwise ``None``
        """
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
        """
        Generate nest objects (neurons, input generators, monitors,...) and connect them. In addition to parent class generate spike generator object for input events.
        """
        # - Add stimulation device
        self._sg = self.nest_module.Create("spike_generator", self.weights_in.shape[0])

        super().setup_nest_network()

    def generate_nest_params_list(self) -> List[Dict[str, np.ndarray]]:
        """Initialize nest neuron parameters and return as list"""

        params = super().generate_nest_params_list()
        for n in range(self.size):
            params[n]["tau_syn_ex"] = self.tau_syn_exc[n]
            params[n]["tau_syn_in"] = self.tau_syn_inh[n]

        return params

    def set_all_connections(self):
        """Set input connections and recurrent connections"""
        # - Input connections
        self.set_connections(
            pop_pre=self._sg,
            pop_post=self._pop,
            weights=self.weights_in,
            delays=self.delay_in,
            connection_exists=self.connection_in_exists,
        )
        # - Recurrent connections
        self.set_connections(
            pop_pre=self._pop,
            pop_post=self._pop,
            weights=self.weights_rec,
            delays=self.delay_rec,
            connection_exists=self.connection_rec_exists,
        )


# - FFIAFNest- Class: define a spiking feedforward layer with spiking outputs
class FFIAFNestV1(Layer):
    """
    Define a spiking feedforward layer with spiking outputs. NEST backend
    """

    class NestProcess(_BaseNestProcess):
        """Class for running NEST in its own process"""

        def __init__(
            self,
            request_q,
            result_q,
            weights: np.ndarray,
            bias: np.ndarray,
            dt: float,
            tau_mem: np.ndarray,
            capacity: np.ndarray,
            v_thresh: np.ndarray,
            v_reset: np.ndarray,
            v_rest: np.ndarray,
            refractory: np.ndarray,
            record: bool = False,
            num_cores: int = 1,
        ):
            """initialize the process"""

            super().__init__(
                request_q=request_q,
                result_q=result_q,
                bias=bias,
                dt=dt,
                capacity=capacity,
                v_thresh=v_thresh,
                v_reset=v_reset,
                v_rest=v_rest,
                refractory=refractory,
                record=record,
                num_cores=num_cores,
                model="iaf_psc_exp",
            )

            # - Record weights and layer specific parameters
            self.weights = A2mA(weights)
            self.size = np.shape(weights)[1]
            self.tau_mem = s2ms(tau_mem)
            # - Keep track of existing connections for more efficient weight updates
            self.connection_exists = np.zeros_like(self.weights, bool)

        def setup_nest_network(self):
            """
            Generate nest objects (neurons, input generators, monitors,...) and connect them. In addition to parent class generate step current generator for inputs.
            """
            # - Add stimulation device
            self._scg = self.nest_module.Create(
                "step_current_generator", self.weights.shape[0]
            )

            super().setup_nest_network()

        def set_all_connections(self):
            """Set connections from step current generator to neuron population"""
            self.set_connections(
                pop_pre=self._scg,
                pop_post=self._pop,
                weights=self.weights,
                connection_exists=self.connection_exists,
                delays=np.ones((len(self._scg), len(self._pop))) * self.dt,
            )

        def generate_nest_params_list(self) -> List[Dict[str, np.ndarray]]:
            """Initialize nest neuron parameters and return as list"""

            params = super().generate_nest_params_list()
            for n in range(self.size):
                params[n]["tau_m"] = self.tau_mem[n]

            return params

        ######### DEFINE IPC COMMANDS ######

        def get_param(self, name):
            """IPC command for getting a parameter"""
            if name == "weights":
                vms = self.read_weights(pop_pre=self._scg, pop_post=self._pop)
            else:
                vms = self.nest_module.GetStatus(self._pop, name)
            return vms

        def set_param(self, name, value):
            """IPC command for setting a parameter"""

            if name == "weights":
                weights_old = self.weights.copy()
                self.weights = value
                self.update_weights(
                    self._scg,
                    self._pop,
                    self.weights,
                    weights_old,
                    delays=None,
                    connection_exists=self.connection_exists,
                )
            else:
                super().set_param(name, value)

        def evolve(
            self, time_base: np.ndarray, input_steps: np.ndarray, num_timesteps: int
        ) -> (np.ndarray, np.ndarray, Union[np.ndarray, None]):
            """
            Evolve state of nest simulation by defined number of timesteps

            :param np.ndarray time_base:    Input time base
            :param np.ndarray input_steps:  Input current steps
            :param int num_timesteps:       Number of timesteps over which to evolve.

            :return (np.ndarray, np.ndarray, Union[np.ndarray, None]):
                1D-array of recorded event times
                1D-array of recorded event channels
                If `self.record`: 2D-array of recorded neuron states, otherwise `None`
            """

            # NEST time starts with 1 (not with 0)
            time_base = s2ms(time_base) + self.dt

            self.nest_module.SetStatus(
                self._scg,
                [
                    {
                        "amplitude_times": time_base,
                        "amplitude_values": input_steps[:, i],
                    }
                    for i in range(len(self._scg))
                ],
            )

            return self.evolve_nest(num_timesteps)

    ## - Constructor
    def __init__(
        self,
        weights: np.ndarray,
        bias: FloatVector = 0.0,
        dt: float = 0.1e-3,
        tau_mem: FloatVector = 20e-3,
        capacity: Optional[FloatVector] = None,
        v_thresh: FloatVector = -55e-3,
        v_reset: FloatVector = -65e-3,
        v_rest: FloatVector = -65e-3,
        refractory: FloatVector = 1e-3,
        name: str = "unnamed",
        record: bool = False,
        num_cores: int = 1,
    ):
        """
        Construct a spiking feedforward layer with IAF neurons, with a NEST back-end. Inputs are continuous currents; outputs are spiking events

        :param np.ndarray weights:                  MxN FFwd weight matrix in nA
        :param FloatVector bias:                    Nx1 bias current vector in nA. Default: ``0.0``
        :param float dt:                            Time-step in seconds. Default: ``0.1 ms``
        :param FloatVector tau_mem:                 Nx1 vector of neuron time constants in seconds. Default: ``20 ms``
        :param Optional[FloatVector] capacity:      Nx1 vector of neuron membrance capacity in nF. Will be set to tau_mem (* 1 nS) if ``None``. Default: ``None``.
        :param FloatVector v_thresh:                Nx1 vector of neuron thresholds in Volt. Default: ``-55 mV``
        :param FloatVector v_reset:                 Nx1 vector of neuron reset potential in Volt. Default: ``-65 mV``
        :param FloatVector v_rest:                  Nx1 vector of neuron resting potential in Volt. Default: ``-65 mV``
        :param FloatVector refractory:              Refractory period after each spike in seconds. Default: ``1 ms``
        :param str name:                            Name for the layer. Default: ``'unnamed'``
        :param bool record:                         Record membrane potential during evolutions
        :param int num_cores:                       Number of CPU cores to use in simulation. Default: ``1``
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
            capacity = tau_mem
        else:
            capacity = self._expand_to_net_size(capacity, "capacity", allow_none=False)

        # - Record neuron parameters
        self._v_thresh = v_thresh
        self._v_reset = v_reset
        self._v_rest = v_rest
        self._tau_mem = tau_mem
        self._bias = bias
        self._capacity = capacity
        self._refractory = refractory
        # - Record layer settings and multiprocessing queues
        self._record = record
        self._num_cores = num_cores
        self.request_q = multiprocessing.Queue()
        self.result_q = multiprocessing.Queue()

        # - Start a process for running nest
        self._setup_nest()

    def _setup_nest(self):
        """
        Set up and start a nest process
        """

        self._nest_process = self.NestProcess(
            self.request_q,
            self.result_q,
            self._weights,
            self._bias,
            self._dt,
            self._tau_mem,
            self._capacity,
            self._v_thresh,
            self._v_reset,
            self._v_rest,
            self._refractory,
            self._record,
            self._num_cores,
        )
        self._nest_process.start()

    def reset_state(self):
        """
        Reset the internal state of the layer
        """

        self.request_q.put([COMMAND_SET, "V_m", V2mV(self._v_rest)])

    def randomize_state(self):
        """
        Randomize the internal state of the layer
        """
        v_range = abs(self._v_thresh - self._v_reset)
        randV = np.random.rand(self._size) * v_range + self._v_reset
        self.v_init = randV

        self.request_q.put([COMMAND_SET, "V_m", V2mV(randV)])

    def reset_time(self):
        """
        Reset the internal clock of this layer
        """

        warn(f"{self.name}: This function resets the whole network")

        self.request_q.put([COMMAND_RESET])
        self._timestep = 0

    def reset_all(self):
        """
        Resets time and state
        """

        self.request_q.put([COMMAND_RESET])
        self._timestep = 0

    # --- State evolution

    def _process_evolution_output(self, num_timesteps: int) -> TSEvent:
        """
        Internal method for processing recorded event data and neuron states after evolution into `.TSEvent` objects.
        :param int num_timesteps:  Evolution duration in timesteps
        :return `.TSEvent`: `.TSEvent` with the recorded events
        """
        if self.record:
            (
                event_time_out,
                event_channel_out,
                recorded_states_array,
            ) = self.result_q.get()

            self.recorded_states = TSContinuous(
                (np.arange(recorded_states_array.shape[1]) + self._timestep) * self.dt,
                recorded_states_array.T,
                t_start=self.t,
                name=f"{self.name} - recorded states",
            )
        else:
            event_time_out, event_channel_out, _ = self.result_q.get()

        # - Start and stop times for output time series
        t_start = self.t
        t_stop = (self._timestep + num_timesteps) * self.dt

        # - Shift event times to middle of time bins
        event_time_out -= 0.5 * self.dt

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

    def evolve(
        self,
        ts_input: Optional[TSContinuous] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Function to evolve the states of this layer given an input

        :param Optional[TSContinuous] ts_input: Input spike trian
        :param Optional[float] duration:        Simulation/Evolution time
        :param Optional[int] num_timesteps:     Number of evolution time steps
        :param bool verbose:                    Currently no effect, just for conformity

        :return `TSEvent`:                      Output spike series
        """
        if ts_input is not None:
            # - Make sure timeseries is of correct type
            if not isinstance(ts_input, TSContinuous):
                raise ValueError(
                    self.start_print + "This layer requires a `TSContinuous` as input."
                )
        # - Prepare time base
        time_base, input_steps, num_timesteps = self._prepare_input(
            ts_input, duration, num_timesteps
        )

        self.request_q.put([COMMAND_EVOLVE, time_base, input_steps, num_timesteps])

        return self._process_evolution_output(num_timesteps)

    def terminate(self):
        """
        Cleanly terminate underlying nest process
        """
        self.request_q.close()
        self.result_q.close()
        self.request_q.cancel_join_thread()
        self.result_q.cancel_join_thread()
        self._nest_process.terminate()
        self._nest_process.join()

    def to_dict(self) -> dict:
        """
        Convert parameters of this layer to a dict if they are relevant for reconstructing an identical layer

        :return Dict:   A dictionary that can be used to reconstruct the layer
        """

        config = {}
        config["weights"] = self.weights.tolist()
        config["bias"] = self.bias.tolist()
        config["dt"] = self.dt
        config["tau_mem"] = self.tau_mem.tolist()
        config["capacity"] = self.capacity.tolist()
        config["v_thresh"] = self.v_thresh.tolist()
        config["v_reset"] = self.v_reset.tolist()
        config["v_rest"] = self.v_rest.tolist()
        config["refractory"] = self.refractory.tolist()
        config["name"] = self.name
        config["record"] = self.record
        config["num_cores"] = self.num_cores
        config["class_name"] = "FFIAFNest"

        return config

    ### --- Properties

    @property
    def output_type(self):
        """(`.TSEvent`) Output time series class (`.TSEvent`)"""
        return TSEvent

    @property
    def weights(self):
        """(np.ndarray) Feedforward weights for this layer, in nA"""
        return SetterArray(self._weights, owner=self, name="weights")

    @property
    def weights_(self):
        """(np.ndarray) Feedforward weights matrix for this layer, in nA"""
        self.request_q.put([COMMAND_GET, "weights"])
        weights = nA2uA(self.result_q.get())
        return ImmutableArray(weights, name=self.start_print)

    @weights.setter
    def weights(self, new_weights: np.ndarray):
        Layer.weights.fset(self, new_weights)
        self.request_q.put([COMMAND_SET, "weights", uA2nA(self._weights)])

    @property
    def refractory(self):
        """(np.ndarray) Refractory parameter for this layer"""
        return SetterArray(self._refractory, owner=self, name="refractory")

    @refractory.setter
    def refractory(self, new_refractory):
        new_refractory = self._expand_to_net_size(
            new_refractory, "refractory", allow_none=False
        )
        self._refractory = new_refractory
        self.request_q.put([COMMAND_SET, "t_ref", s2ms(new_refractory)])

    @property
    def capacity(self):
        """Capacity of this layer (?)"""
        return SetterArray(self._capacity, owner=self, name="capaity")

    @capacity.setter
    def capacity(self, new_capacity):
        new_capacity = self._expand_to_net_size(
            new_capacity, "capacity", allow_none=False
        ).astype(float)
        self._capacity = new_capacity
        self.request_q.put([COMMAND_SET, "C_m", F2mF(new_capacity)])

    @property
    def Vmem(self):
        """(np.ndarray) Membrane potential $V_m$ of the neurons in this layer"""
        self.request_q.put([COMMAND_GET, "V_m"])
        vms = np.array(self.result_q.get())
        return SetterArray(mV2V(vms), owner=self, name="state")

    @Vmem.setter
    def Vmem(self, new_state):
        new_state = self._expand_to_net_size(new_state, "state", allow_none=False)
        self.request_q.put([COMMAND_SET, "V_m", V2mV(new_state)])

    @property
    def tau_mem(self):
        """(np.ndarray) Membrane time constants $\\tau_m$ of the neurons in this layer"""
        return SetterArray(self._tau_mem, owner=self, name="tau_mem")

    @tau_mem.setter
    def tau_mem(self, new_tau_mem):
        new_tau_mem = self._expand_to_net_size(new_tau_mem, "tau_mem", allow_none=False)
        self._tau_mem = new_tau_mem
        self.request_q.put([COMMAND_SET, "tau_m", s2ms(new_tau_mem)])

    @property
    def bias(self):
        """(np.ndarray) Bias currents of the neurons in this layer"""
        return SetterArray(self._bias, owner=self, name="bias")

    @bias.setter
    def bias(self, new_bias):
        new_bias = self._expand_to_net_size(new_bias, "bias", allow_none=False)
        self._bias = new_bias
        self.request_q.put([COMMAND_SET, "I_e", A2mA(new_bias)])

    @property
    def v_thresh(self):
        """(np.ndarray) Threshold potential $V_{thresh}$ of the neurons of this layer"""
        return SetterArray(self._v_thresh, owner=self, name="v_thresh")

    @v_thresh.setter
    def v_thresh(self, new_v_thresh):
        new_v_thresh = self._expand_to_net_size(
            new_v_thresh, "v_thresh", allow_none=False
        )
        self._v_thresh = new_v_thresh
        self.request_q.put([COMMAND_SET, "V_th", V2mV(new_v_thresh)])

    @property
    def v_reset(self):
        """(np.ndarray) Reset potential $v_{reset}$ of the neurons of this layer"""
        return SetterArray(self._v_reset, owner=self, name="v_reset")

    @v_reset.setter
    def v_reset(self, new_v_reset):
        new_v_reset = self._expand_to_net_size(new_v_reset, "v_reset", allow_none=False)
        self._v_reset = new_v_reset
        self.request_q.put([COMMAND_SET, "V_reset", V2mV(new_v_reset)])

    @property
    def v_rest(self):
        """(np.ndarray) Resting potential $V_{rest}$ of the neurons of this layer"""
        return SetterArray(self._v_rest, owner=self, name="v_rest")

    @v_rest.setter
    def v_rest(self, new_v_rest):
        new_v_rest = self._expand_to_net_size(new_v_rest, "v_rest", allow_none=False)
        self._v_rest = new_v_rest
        self.request_q.put([COMMAND_SET, "E_L", V2mV(new_v_rest)])

    @property
    def t(self):
        """(float) Current time of this layer"""
        return self._timestep * self.dt

    @Layer.dt.setter
    def dt(self, _):
        warn(self.start_print + "The `dt` property cannot be set for this layer")

    @property
    def record(self):
        """Record state of this layer"""
        return self._record

    @property
    def num_cores(self):
        """(int) Number of cores used by the NEST backend for this layer"""
        return self._num_cores


# - RecIAFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
class RecIAFSpkInNestV1(FFIAFNestV1):
    """
    Spiking recurrent layer with spiking in- and outputs. NEST backend.
    """

    class NestProcess(_BaseNestProcessSpkInRec):
        """Base class for running NEST in its own process (recurrent layers, spike input)"""

        def __init__(
            self,
            request_q,
            result_q,
            weights_in: np.ndarray,
            weights_rec: np.ndarray,
            delay_in: FloatVector,
            delay_rec: FloatVector,
            bias: FloatVector,
            dt: float,
            tau_mem: FloatVector,
            tau_syn_exc: FloatVector,
            tau_syn_inh: FloatVector,
            capacity: FloatVector,
            v_thresh: FloatVector,
            v_reset: FloatVector,
            v_rest: FloatVector,
            refractory: FloatVector,
            record: bool = False,
            num_cores: int = 1,
        ):
            """
            Initialise the Nest process

            :param multiprocessing.Queue request_q:     Queue for receiving RPC commands
            :param multiprocessing.Queue result_q:      Queue for sending results
            :param np.ndarray weights_in:               Weights of the input connections
            :param np.ndarray weights_rec:              Weights of the recurrent connections
            :param FloatVector delay_in:                Delays of the input connections
            :param FloatVector delay_rec:               Delays of the recurrent connections
            :param FloatVector bias:                    Biases (static input current)  of the neurons
            :param float dt:                            Time-step (resolution) of the simulation
            :param FloatVector tau_mem:                 Membrane time constant
            :param FloatVector tau_syn_exc:             Synaptic time constant of excitatory connections
            :param FloatVector tau_syn_inh:             Synaptic time constant of inhibitory connections
            :param FloatVector capacity:                Membrane capacity
            :param FloatVector v_thresh:                Threshold potential
            :param FloatVector v_reset:                 Reset potential
            :param FloatVector v_rest:                  Resting potential
            :param FloatVector refractory:              Refractory period
            :param bool record:                         If ``true`` neurons states are recorded
            :param int num_cores:                       Number of local threads to be used for this simulation
            """
            super().__init__(
                request_q=request_q,
                result_q=result_q,
                weights_in=weights_in,
                weights_rec=weights_rec,
                delay_in=delay_in,
                delay_rec=delay_rec,
                bias=bias,
                dt=dt,
                tau_syn_exc=tau_syn_exc,
                tau_syn_inh=tau_syn_inh,
                capacity=capacity,
                v_thresh=v_thresh,
                v_reset=v_reset,
                v_rest=v_rest,
                refractory=refractory,
                record=record,
                num_cores=num_cores,
                model="iaf_psc_exp",
            )

            # - Record layer-specific parameters
            self.tau_mem = s2ms(tau_mem)

        def generate_nest_params_list(self) -> List[Dict[str, np.ndarray]]:
            """
            Initialize nest neuron parameters and return as a list
            """
            params = super().generate_nest_params_list()
            for n in range(self.size):
                params[n]["tau_m"] = self.tau_mem[n]

            return params

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        delay_in: FloatVector = 0.1e-3,
        delay_rec: FloatVector = 0.1e-3,
        bias: FloatVector = 0.0,
        dt: float = 0.1e-3,
        tau_mem: FloatVector = 20e-3,
        tau_syn: Optional[Union[np.ndarray, float]] = 50e-3,
        tau_syn_exc: Optional[FloatVector] = None,
        tau_syn_inh: Optional[FloatVector] = None,
        v_thresh: FloatVector = -55e-3,
        v_reset: FloatVector = -65e-3,
        v_rest: FloatVector = -65e-3,
        capacity: Optional[FloatVector] = None,
        refractory: FloatVector = 1e-3,
        name: str = "unnamed",
        record: bool = False,
        num_cores: int = 1,
    ):
        """
        RecIAFSpkInNest - Construct a spiking recurrent layer with IAF neurons, with a NEST back-end. Spiking input and output

        :param FloatVector weights_in:              MxN input weight matrix in nA
        :param FloatVector weights_rec:             NxN recurrent weight matrix in nA
        :param FloatVector delay_in:                Input delay in s. Default: ``0.1 ms``
        :param FloatVector bias:                    Nx1 bias current vector in nA. Default 0.
        :param float dt:                            Time-step in seconds. Default: ``0.1 ms``
        :param FloatVector tau_mem:                 Nx1 vector of neuron time constants in seconds. Default: ``20 ms``
        :param FloatVector tau_syn:                 Nx1 vector of synapse time constants in seconds. Used instead of `tau_syn_exc` or `tau_syn_inh` if they are ``None``. Default: ``50 ms``
        :param Optional[FloatVector] tau_syn_exc:   Nx1 vector of excitatory synapse time constants in seconds. If ``None``, use ``tau_syn``. Default: ``None``
        :param Optional[FloatVector] tau_syn_inh:   Nx1 vector of inhibitory synapse time constants in seconds. If ``None``, use ``tau_syn``. Default: ``None``
        :param FloatVector v_thresh:                Nx1 vector of neuron thresholds in Volt. Default: -55 mV
        :param FloatVector v_reset:                 Nx1 vector of neuron reset potential in Volt. Default: -65 mV
        :param FloatVector v_rest:                  Nx1 vector of neuron resting potential in Volt. Default: -65 mV
        :param Optional[FloatVector] capacity:      Nx1 vector of neuron membrance capacity in nF. Will be set to ``tau_mem`` (* 1 nS) if ``None``. Default: ``None``.
        :param float refractory:                    Refractory period after each spike in seconds. Default: ``1 ms``
        :param str name:                            Name for the layer. Default: ``'unnamed'``
        :param bool record:                         Record membrane potential during evolutions
        """

        # - Determine layer size and name to run `_expand_to_net_size` method and store input weights
        self._weights_in = np.atleast_2d(weights_in)
        self._size_in, self._size = self._weights_in.shape
        self.name = name

        # - Handle synaptic time constants
        if tau_syn_exc is None:
            tau_syn_exc = tau_syn
        if tau_syn_inh is None:
            tau_syn_inh = tau_syn

        # - Convert parameters to arrays before passing to nest process
        self._weights_rec = self._expand_to_shape(
            weights_rec, (self.size, self.size), "weights_rec", allow_none=False
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

        super().__init__(
            weights=weights_in,
            bias=bias,
            dt=dt,
            tau_mem=tau_mem,
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
        """
        Set up and start a nest process for this layer
        """

        self._nest_process = self.NestProcess(
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
        self._nest_process.start()

    # --- State evolution

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        Function to evolve the states of this layer given an input

        :param Optional[TSEvent]ts_input:   Input spike trian
        :param Optional[float] duration:    Simulation/Evolution time
        :param Optional[int] num_timesteps: Number of evolution time steps
        :param bool verbose:                Currently no effect, just for conformity
        :return TSEvent:                    Output spike series

        """
        if ts_input is not None:
            # - Make sure timeseries is of correct type
            if not isinstance(ts_input, TSEvent):
                raise ValueError(
                    self.start_print + "This layer requires a `TSEvent` as input."
                )
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

        return self._process_evolution_output(num_timesteps)

    def to_dict(self) -> dict:
        """
        Convert parameters of this layer to a dict if they are relevant for reconstructing an identical layer

        :return Dict:   A dictionary that can be used to reconstruct the layer
        """
        config = super().to_dict()
        config.pop("weights")
        config["weights_in"] = self._weights_in.tolist()
        config["weights_rec"] = self._weights_rec.tolist()
        config["delay_in"] = self._delay_in.tolist()
        config["delay_rec"] = self._delay_rec.tolist()
        config["tau_syn_exc"] = self.tau_syn_exc.tolist()
        config["tau_syn_inh"] = self.tau_syn_inh.tolist()
        config["class_name"] = "RecIAFSpkInNest"

        return config

    ### --- Properties

    @property
    def input_type(self):
        """(`.TSEvent`) Input time series class for this layer (`.TSEvent`)"""
        return TSEvent

    @property
    def weights_in(self):
        """(np.ndarray) Input weights for this layer, in nA"""
        return SetterArray(self._weights_in, owner=self, name="weights_in")

    @property
    def weights_in_(self):
        """(np.ndarray) Input weights for this layer, in nA"""
        self.request_q.put([COMMAND_GET, "weights_in"])
        weights = nA2uA(self.result_q.get())
        return ImmutableArray(weights, name=self.start_print)

    @weights_in.setter
    def weights_in(self, new_weights):
        self._weights_in = self._expand_to_shape(
            new_weights, (self.size_in, self.size), "weights_in", allow_none=False
        ).astype(float)
        self.request_q.put([COMMAND_SET, "weights_in", uA2nA(self._weights_in)])

    @property
    def weights_rec(self):
        """(np.ndarray) Recurrent weights for this layer, in nA"""
        return SetterArray(self._weights_rec, owner=self, name="weights_rec")

    @property
    def weights_rec_(self):
        """(np.ndarray) Recurrent weights for this layer, in nA"""
        self.request_q.put([COMMAND_GET, "weights_rec"])
        weights = nA2uA(self.result_q.get())
        return ImmutableArray(weights, name=self.start_print)

    @weights_rec.setter
    def weights_rec(self, new_weights):
        self._weights_rec = self._expand_to_shape(
            new_weights, (self.size, self.size), "weights_rec", allow_none=False
        ).astype(float)
        self.request_q.put([COMMAND_SET, "weights_rec", uA2nA(self._weights_rec)])

    @property
    def delay_in(self):
        """(float) Input delay for this layer"""
        return SetterArray(self._delay_in, owner=self, name="delay_in")

    @delay_in.setter
    def delay_in(self, new_delay_in):
        raise AttributeError(
            self.start_print
            + "Updating delays is currently not supported. "
            + "Please contact the developer of this program if you need this feature."
        )

    @property
    def delay_rec(self):
        return SetterArray(self._delay_rec, owner=self, name="delay_rec")

    @delay_rec.setter
    def delay_rec(self, new_delay_rec):
        raise AttributeError(
            self.start_print
            + "Updating delays is currently not supported. "
            + "Please contact the developer of this program if you need this feature."
        )

    @property
    def tau_syn(self):
        """(np.ndarray) Synaptic time constant in s"""
        return SetterArray(self._tau_syn, owner=self, name="tau_syn")

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
        """(np.ndarray) Excitatory synaptic time constant in s"""
        return SetterArray(self._tau_syn_exc, owner=self, name="tau_syn_exc")

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
        """(np.ndarray) Inhibitory synaptic time constant in s"""
        return SetterArray(self._tau_syn_inh, owner=self, name="tau_syn_inh")

    @tau_syn_inh.setter
    def tau_syn_inh(self, new_tau_syn_inh):
        new_tau_syn_inh = self._expand_to_net_size(
            new_tau_syn_inh, "tau_syn_inh", allow_none=False
        )
        new_tau_syn_inh = new_tau_syn_inh.astype(float)
        self._tau_syn_inh = new_tau_syn_inh
        self._tau_syn = None
        self.request_q.put([COMMAND_SET, "tau_syn_in", s2ms(new_tau_syn_inh)])


@astimedmodule(
    parameters=[
        "weights",
        "bias",
        "tau_mem",
        "capacity",
        "v_thresh",
        "v_rest",
        "v_rest",
        "refractory",
        "capacity",
    ],
    simulation_parameters=["dt", "num_cores", "record"],
    states=["Vmem"],
)
class FFIAFNest(FFIAFNestV1):
    pass


@astimedmodule(
    parameters=[
        "weights_in",
        "weights_rec",
        "delay_in",
        "delay_rec",
        "bias",
        "tau_mem",
        "tau_syn_exc",
        "tau_syn_inh",
        "capacity",
        "v_thresh",
        "v_reset",
        "v_rest",
        "refractory",
    ],
    simulation_parameters=["dt", "record", "num_cores"],
    states=["Vmem"],
)
class RecIAFSpkInNest(RecIAFSpkInNestV1):
    pass
