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




# - RecAEIFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
class RecAEIFSpkInNest(Layer):
    """ RecAEIFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
    """

    class NestProcess(multiprocessing.Process):
        """ Class for running NEST in its own process """

        def __init__(
            self,
            request_q,
            result_q,
            weights_in: np.ndarray,
            weights_rec: np.ndarray,
            delay_in: Union[float, np.ndarray],
            delay_rec: Union[float, np.ndarray],
            vfBias: Union[float, np.ndarray],
            dt: float,
            vtTauN: Union[float, np.ndarray],
            vtTauS: Union[float, np.ndarray],
            vfCapacity: Union[float, np.ndarray],
            vfVThresh: Union[float, np.ndarray],
            vfVReset: Union[float, np.ndarray],
            vfVRest: Union[float, np.ndarray],
            tRefractoryTime,
            a: Union[float, np.ndarray],
            b: Union[float, np.ndarray],
            delta_t: Union[float, np.ndarray],
            vtTauW: Union[float, np.ndarray],
            bRecord: bool = False,
            num_cores: int = 1,
        ):
            """ initializes the process """

            multiprocessing.Process.__init__(self, daemon=True)

            self.request_q = request_q
            self.result_q = result_q

            # - Record neuron parameters
            self.dt = s2ms(dt)
            self.vfVThresh = V2mV(vfVThresh)
            self.vfVReset = V2mV(vfVReset)
            self.vfVRest = V2mV(vfVRest)
            self.vtTauN = s2ms(vtTauN)
            self.vtTauS = s2ms(vtTauS)
            self.vfBias = V2mV(vfBias)
            self.vfCapacity = vfCapacity
            self.weights_in = V2mV(weights_in)
            self.weights_rec = V2mV(weights_rec)
            self.delay_in = s2ms(delay_in)
            self.delay_rec = s2ms(delay_rec)
            self.tRefractoryTime = s2ms(tRefractoryTime)
            self.bRecord = bRecord
            self.size = np.shape(weights_rec)[0]
            self.num_cores = num_cores
            self.a = a
            self.b = b
            self.delta_t = delta_t
            self.vtTauW = s2ms(vtTauW)

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

                if type(self.vtTauS) is np.ndarray:
                    p["tau_syn_ex"] = self.vtTauS[n]
                    p["tau_syn_in"] = self.vtTauS[n]
                else:
                    p["tau_syn_ex"] = self.vtTauS
                    p["tau_syn_in"] = self.vtTauS

                if type(self.vtTauN) is np.ndarray:
                    if type(self.vfCapacity) == np.ndarray:
                        p["g_L"] = self.vfCapacity[n] / self.vtTauN[n]
                    else:
                        p["g_L"] = self.vfCapacity / self.vtTauN[n]
                else:
                    if type(self.vfCapacity) == np.ndarray:
                        p["g_L"] = self.vfCapacity[n] / self.vtTauN
                    else:
                        p["g_L"] = self.vfCapacity / self.vtTauN

                if type(self.vfVThresh) is np.ndarray:
                    p["V_th"] = self.vfVThresh[n]
                else:
                    p["V_th"] = self.vfVThresh

                if type(self.vfVReset) is np.ndarray:
                    p["V_reset"] = self.vfVReset[n]
                else:
                    p["V_reset"] = self.vfVReset

                if type(self.vfVReset) is np.ndarray:
                    p["E_L"] = self.vfVRest[n]
                    p["V_m"] = self.vfVRest[n]
                else:
                    p["E_L"] = self.vfVRest
                    p["V_m"] = self.vfVRest

                if type(self.tRefractoryTime) is np.ndarray:
                    p["t_ref"] = self.tRefractoryTime[n]
                else:
                    p["t_ref"] = self.tRefractoryTime

                if type(self.vfBias) is np.ndarray:
                    p["I_e"] = self.vfBias[n]
                else:
                    p["I_e"] = self.vfBias

                if type(self.vfCapacity) is np.ndarray:
                    p["C_m"] = self.vfCapacity[n]
                else:
                    p["C_m"] = self.vfCapacity

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

                if type(self.vtTauW) is np.ndarray:
                    p["tau_w"] = self.vtTauW[n]
                else:
                    p["tau_w"] = self.vtTauW

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

            for pre, row in enumerate(self.weights_in):
                for post, w in enumerate(row):
                    if w == 0:
                        continue
                    pres.append(self._sg[pre])
                    posts.append(self._pop[post])

            nest.Connect(pres, posts, "one_to_one")
            conns = nest.GetConnections(self._sg, self._pop)
            connsPrePost = np.array(nest.GetStatus(conns, ["source", "target"]))

            if not len(connsPrePost) == 0:
                connsPrePost[:, 0] -= np.min(self._sg)
                connsPrePost[:, 1] -= np.min(self._pop)

                weights = [self.weights_in[conn[0], conn[1]] for conn in connsPrePost]
                if type(self.delay_in) is np.ndarray:
                    delays = [self.delay_in[conn[0], conn[1]] for conn in connsPrePost]
                else:
                    delays = np.array([self.delay_in] * len(weights))

                delays = np.clip(delays, self.dt, np.max(delays))

                nest.SetStatus(
                    conns, [{"weight": w, "delay": d} for w, d in zip(weights, delays)]
                )

            t1 = time.time()
            # - Create recurrent connections
            pres = []
            posts = []

            for pre, row in enumerate(self.weights_rec):
                for post, w in enumerate(row):
                    if w == 0:
                        continue
                    pres.append(self._pop[pre])
                    posts.append(self._pop[post])

            nest.Connect(pres, posts, "one_to_one")

            conns = nest.GetConnections(self._pop, self._pop)
            connsPrePost = nest.GetStatus(conns, ["source", "target"])

            if not len(connsPrePost) == 0:
                connsPrePost -= np.min(self._pop)

                weights = [self.weights_rec[conn[0], conn[1]] for conn in connsPrePost]
                if type(self.delay_rec) is np.ndarray:
                    delays = [
                        self.delay_rec[conn[0], conn[1]] for conn in connsPrePost
                    ]
                else:
                    delays = np.array([self.delay_rec] * len(weights))

                delays = np.clip(delays, self.dt, np.max(delays))

                nest.SetStatus(
                    conns, [{"weight": w, "delay": d} for w, d in zip(weights, delays)]
                )

            if self.bRecord:
                # - Monitor for recording network potential
                self._mm = nest.Create(
                    "multimeter", 1, {"record_from": ["V_m"], "interval": 1.0}
                )
                nest.Connect(self._mm, self._pop)

            ######### DEFINE IPC COMMANDS ######

            def getParam(name):
                """ IPC command for getting a parameter """
                vms = nest.GetStatus(self._pop, name)
                return vms

            def setParam(name, value):
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
                vtEventTimes, vnEventChannels, num_timesteps: Optional[int] = None
            ):
                """ IPC command running the network for num_timesteps with mfInputStep as input """

                if len(vnEventChannels > 0):
                    # convert input index to NEST id
                    vnEventChannels += np.min(self._sg)

                    # NEST time starts with 1 (not with 0)
                    nest.SetStatus(
                        self._sg,
                        [
                            {"spike_times": s2ms(vtEventTimes[vnEventChannels == i])}
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
                if self.bRecord:
                    events = nest.GetStatus(self._mm, "events")[0]
                    vbUseEvent = events["times"] >= startTime

                    senders = events["senders"][vbUseEvent]
                    times = events["times"][vbUseEvent]
                    vms = events["V_m"][vbUseEvent]

                    mfRecordStates = []
                    u_senders = np.unique(senders)
                    for i, nid in enumerate(u_senders):
                        ind = np.where(senders == nid)[0]
                        _times = times[ind]
                        order = np.argsort(_times)
                        _vms = vms[ind][order]
                        mfRecordStates.append(_vms)

                    mfRecordStates = np.array(mfRecordStates)

                # - Build response TimeSeries
                events = nest.GetStatus(self._sd, "events")[0]
                vbUseEvent = events["times"] >= startTime
                vtEventTimeOutput = ms2s(events["times"][vbUseEvent])
                vnEventChannelOutput = events["senders"][vbUseEvent]

                # sort spiking response
                order = np.argsort(vtEventTimeOutput)
                vtEventTimeOutput = vtEventTimeOutput[order]
                vnEventChannelOutput = vnEventChannelOutput[order]

                # transform from NEST id to index
                vnEventChannelOutput -= np.min(self._pop)

                if self.bRecord:
                    return [
                        vtEventTimeOutput,
                        vnEventChannelOutput,
                        mV2V(mfRecordStates),
                    ]
                else:
                    return [vtEventTimeOutput, vnEventChannelOutput, None]

            IPC_switcher = {
                COMMAND_GET: getParam,
                COMMAND_SET: setParam,
                COMMAND_RESET: reset,
                COMMAND_EVOLVE: evolve,
            }

            # wait for an IPC command

            while True:
                req = self.request_q.get()

                func = IPC_switcher.get(req[0])

                result = func(*req[1:])

                if not result is None:
                    self.result_q.put(result)

    ## - Constructor
    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        delay_in=0.0001,
        delay_rec=0.0001,
        vfBias: np.ndarray = 0.0,
        dt: float = 0.0001,
        vtTauN: np.ndarray = 0.02,
        vtTauS: np.ndarray = 0.05,
        vfVThresh: np.ndarray = -0.055,
        vfVReset: np.ndarray = -0.065,
        vfVRest: np.ndarray = -0.065,
        vfCapacity: Union[float, np.ndarray] = 100.0,
        tRefractoryTime = 0.001,
        a: Union[float, np.ndarray] = 4.0,
        b: Union[float, np.ndarray] = 80.5,
        delta_t: Union[float, np.ndarray] = 2.,
        vtTauW: Union[float, np.ndarray] = 0.144,
        name: str = "unnamed",
        bRecord: bool = False,
        num_cores: int = 1,
    ):
        """
        RecAEIFSpkInNest - Construct a spiking recurrent layer with AEIF neurons, with a NEST back-end
                           in- and outputs are spiking events

        :param weights_in:           np.array MxN input weight matrix.
        :param weights_rec:          np.array NxN recurrent weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10.5mA

        :param dt:             float Time-step. Default: 0.1 ms

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param vtTauS:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron reset potential. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron resting potential. Default: -65mV

        :param vfCapacity:       np.array Nx1 vector of neuron membrance capacity. Default: 100 pF
        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param a:              float or np.ndarray scaling for subthreshold adaptation. Default: 4.
        :param b:              float or np.ndarray additive value for spike triggered adaptation. Default: 80.5
        :param vtTauW:          float or np.ndarray time constant for adaptation relaxation. Default: 144.0 ms
        :param delta_t:        float or np.ndarray scaling for exponential part of the activation function. Default: 2.


        :param name:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions
        """

        if type(weights_in) is list:
            weights_in = np.asarray(weights_in)

        if type(weights_rec) is list:
            weights_rec = np.asarray(weights_rec)

        if type(delay_in) is list:
            delay_in = np.asarray(delay_in)

        if type(delay_rec) is list:
            delay_rec = np.asarray(delay_rec)

        if type(vfBias) is list:
            vfBias = np.asarray(vfBias)

        if type(vtTauN) is list:
            vtTauN = np.asarray(vtTauN)

        if type(vtTauS) is list:
            vtTauS = np.asarray(vtTauS)

        if type(vfCapacity) is list:
            vfCapacity = np.asarray(vfCapacity)

        if type(vfVThresh) is list:
            vfVThresh = np.asarray(vfVThresh)

        if type(vfVReset) is list:
            vfVReset = np.asarray(vfVReset)

        if type(vfVRest) is list:
            vfVRest = np.asarray(vfVRest)

        if type(a) is list:
            a = np.asarray(a)

        if type(b) is list:
            b = np.asarray(b)

        if type(delta_t) is list:
            delta_t = np.asarray(delta_t)

        if type(vtTauW) is list:
            vtTauW = np.asarray(vtTauW)

        # - Call super constructor (`asarray` is used to strip units)

        # TODO this does not make much sense (weights <- weights_in)
        super().__init__(weights=np.asarray(weights_in), dt=dt, name=name)

        self.num_cores = num_cores

        self.request_q = multiprocessing.Queue()
        self.result_q = multiprocessing.Queue()

        self.nestProcess = self.NestProcess(
            self.request_q,
            self.result_q,
            weights_in=weights_in,
            weights_rec=weights_rec,
            delay_in=delay_in,
            delay_rec=delay_rec,
            vfBias=vfBias,
            dt=dt,
            vtTauN=vtTauN,
            vtTauS=vtTauS,
            vfCapacity=vfCapacity,
            vfVThresh=vfVThresh,
            vfVReset=vfVReset,
            vfVRest=vfVRest,
            tRefractoryTime=tRefractoryTime,
            bRecord=bRecord,
            num_cores=num_cores,
            a=a,
            b=b,
            delta_t=delta_t,
            vtTauW=vtTauW
        )

        self.nestProcess.start()

        # - Record neuron parameters
        self._vfVThresh = vfVThresh
        self._vfVReset = vfVReset
        self._vfVRest = vfVRest
        self._vtTauN = vtTauN
        self._vtTauS = vtTauS
        self._vfBias = vfBias
        self.vfCapacity = vfCapacity
        self.weights_in = weights_in
        self.weights_rec = weights_rec
        self._tRefractoryTime = tRefractoryTime
        self.bRecord = bRecord
        self._fA = a
        self._fB = b
        self._fDelta_T = delta_t
        self._vtTauW = vtTauW


    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """

        self.request_q.put([COMMAND_SET, "V_m", V2mV(self.vfVRest)])

    def randomize_state(self):
        """ .randomize_state() - Method: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        fRangeV = abs(self._vfVThresh - self._vfVReset)
        randV = np.random.rand(self.size) * fRangeV + self._vfVReset

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
        num_timesteps = self._determine_timesteps(ts_input, duration, num_timesteps)

        # - Generate discrete time base
        vtTimeBase = self._gen_time_trace(self.t, num_timesteps)

        # - Set spikes for spike generator
        if ts_input is not None:
            vtEventTimes, vnEventChannels = ts_input(
                vtTimeBase[0], vtTimeBase[-1] + self.dt
            )

        else:
            vtEventTimes = np.array([])
            vnEventChannels = np.array([])

        self.request_q.put(
            [COMMAND_EVOLVE, vtEventTimes, vnEventChannels, num_timesteps]
        )

        if self.bRecord:
            vtEventTimeOutput, vnEventChannelOutput, self.mfRecordStates = (
                self.result_q.get()
            )
        else:
            vtEventTimeOutput, vnEventChannelOutput, _ = self.result_q.get()

        # - Start and stop times for output time series
        tStart = self._timestep * self.dt
        tStop = (self._timestep + num_timesteps) * self.dt

        # - Update layer time step
        self._timestep += num_timesteps

        return TSEvent(
            np.clip(vtEventTimeOutput, tStart, tStop),
            vnEventChannelOutput,
            name="Layer spikes",
            num_channels=self.size,
            t_start=tStart,
            t_stop=tStop,
        )

    def terminate(self):
        self.request_q.close()
        self.result_q.close()
        self.request_q.cancel_join_thread()
        self.result_q.cancel_join_thread()
        self.nestProcess.terminate()
        self.nestProcess.join()

    ### --- Properties

    @property
    def input_type(self):
        return TSEvent

    @property
    def output_type(self):
        return TSEvent

    @property
    def tRefractoryTime(self):
        return self._tRefractoryTime

    @property
    def state(self):
        self.request_q.put([COMMAND_GET, "V_m"])
        vms = np.array(self.result_q.get())
        return mV2V(vms)

    @property
    def vW(self):
        self.request_q.put([COMMAND_GET, "w"])
        ws = np.array(self.result_q.get())
        return ws

    @state.setter
    def state(self, vNewState):
        self.request_q.put([COMMAND_SET, "V_m", V2mV(vNewState)])

    @property
    def vtTauN(self):
        return self._vtTauN

    @vtTauN.setter
    def vtTauN(self, vtNewTauN):
        self._vtTauN = vtNewTauN

        if type(vtNewTauN) is np.ndarray:
            gls = []
            for nid in range(self.vtNewTauN):
                if type(self.vfCapacity) == np.ndarray:
                    gls.append(self.vfCapacity[nid] / self.vtTauN[nid])
                else:
                    gls.append(self.vfCapacity / self.vtTauN[nid])
        else:
            if type(self.vfCapacity) == np.ndarray:
                gls = []
                for nid in range(self.vfCapacity):
                    gls.append(self.vfCapacity[nid] / self.vtTauN)
            else:
                gls = self.vfCapacity / self.vtTauN

        self.request_q.put([COMMAND_SET, "g_L", s2ms(gls)])

    @property
    def vtTauS(self):
        return self._vtTauS

    @vtTauS.setter
    def vtTauS(self, vtNewTauS):
        self._vtTauS = vtNewTauS

        self.request_q.put([COMMAND_SET, "tau_syn_ex", s2ms(vtNewTauS)])
        self.request_q.put([COMMAND_SET, "tau_syn_in", s2ms(vtNewTauS)])

    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias):
        self._vfBias = vfNewBias

        self.request_q.put([COMMAND_SET, "I_e", V2mV(vfNewBias)])

    @property
    def vfVThresh(self):
        return self._vfVThresh

    @vfVThresh.setter
    def vfVThresh(self, vfNewVThresh):
        self._vfVThresh = vfNewVThresh

        self.request_q.put([COMMAND_SET, "V_th", V2mV(vfNewVThresh)])

    @property
    def vfVReset(self):
        return self._vfVReset

    @vfVReset.setter
    def vfVReset(self, vfNewVReset):
        self._vfVReset = vfNewVReset

        self.request_q.put([COMMAND_SET, "V_reset", V2mV(vfNewVReset)])

    @property
    def vfVRest(self):
        return self._vfVRest

    @vfVRest.setter
    def vfVRest(self, vfNewVRest):
        self._vfVRest = vfNewVRest

        self.request_q.put([COMMAND_SET, "E_L", V2mV(vfNewVRest)])

    @property
    def a(self):
        return self._fA

    @a.setter
    def a(self, newFA):
        self._fA = newFA
        self.request_q.put([COMMAND_SET, "a", newFA])

    @property
    def b(self):
        return self._fB

    @b.setter
    def b(self, newFB):
        self._fB = newFB
        self.request_q.put([COMMAND_SET, "b", newFB])

    @property
    def delta_t(self):
        return self._fDelta_T

    @delta_t.setter
    def delta_t(self, newFDelta_T):
        self.delta_t = newFDelta_T
        self.request_q.put([COMMAND_SET, "Delta_T", newFDelta_T])

    @property
    def vtTauW(self):
        return self._vtTauW

    @vtTauW.setter
    def vtTauW(self, newVtTauW):
        self._vtTauW = newVtTauW
        self.request_q.put([COMMAND_SET, "tau_w", s2ms(newVtTauW)])


    @property
    def t(self):
        return self._timestep * self.dt

    @Layer.dt.setter
    def dt(self):
        raise ValueError("The `dt` property cannot be set for this layer")

    def to_dict(self):

        config = {}
        config["name"] = self.name
        config["weights_in"] = self.weights_in.tolist()
        config["weights_rec"] = self.weights_rec.tolist()
        config["vfBias"] = (
            self.vfBias if type(self.vfBias) is float else self.vfBias.tolist()
        )
        config["dt"] = self.dt if type(self.dt) is float else self.dt.tolist()
        config["vfVThresh"] = (
            self.vfVThresh if type(self.vfVThresh) is float else self.vfVThresh.tolist()
        )
        config["vfVReset"] = (
            self.vfVReset if type(self.vfVReset) is float else self.vfVReset.tolist()
        )
        config["vfVRest"] = (
            self.vfVRest if type(self.vfVRest) is float else self.vfVRest.tolist()
        )
        config["vfCapacity"] = (
            self.vfCapacity
            if type(self.vfCapacity) is float
            else self.vfCapacity.tolist()
        )
        config["tRef"] = (
            self.tRefractoryTime
            if type(self.tRefractoryTime) is float
            else self.tRefractoryTime.tolist()
        )
        config["num_cores"] = self.num_cores
        config["tauN"] = (
            self.vtTauN if type(self.vtTauN) is float else self.vtTauN.tolist()
        )
        config["tauS"] = (
            self.vtTauS if type(self.vtTauS) is float else self.vtTauS.tolist()
        )
        config["bRecord"] = self.bRecord

        config["a"] = (
            self._fA if type(self._fA) is float else self._fA.tolist()
        )

        config["b"] = (
            self._fB if type(self._fB) is float else self._fB.tolist()
        )

        config["delta_t"] = (
            self._fDelta_T if type(self._fDelta_T) is float else self._fDelta_T.tolist()
        )

        config["vtTauW"] = (
            self._vtTauW if type(self._vtTauW) is float else self._vtTauW.tolist()
        )

        config["ClassName"] = "RecAEIFSpkInNest"

        return config

    def save(self, config, filename):
        with open(filename, "w") as f:
            json.dump(config, f)

    @staticmethod
    def load_from_dict(config):

        return RecAEIFSpkInNest(
            weights_in=config["weights_in"],
            weights_rec=config["weights_rec"],
            vfBias=config["vfBias"],
            dt=config["dt"],
            vtTauN=config["tauN"],
            vtTauS=config["tauS"],
            vfCapacity=config["vfCapacity"],
            vfVThresh=config["vfVThresh"],
            vfVReset=config["vfVReset"],
            vfVRest=config["vfVRest"],
            tRefractoryTime=config["tRef"],
            name=config["name"],
            bRecord=config["bRecord"],
            num_cores=config["num_cores"],
            a=config['a'],
            b=config['b'],
            delta_t=config['delta_t'],
            vtTauW=config['vtTauW'],
        )

    @staticmethod
    def load_from_file(filename):
        with open(filename, "r") as f:
            config = json.load(f)

        return RecAEIFSpkInNest(
            weights_in=config["weights_in"],
            weights_rec=config["weights_rec"],
            vfBias=config["vfBias"],
            dt=config["dt"],
            vtTauN=config["tauN"],
            vtTauS=config["tauS"],
            vfCapacity=config["vfCapacity"],
            vfVThresh=config["vfVThresh"],
            vfVReset=config["vfVReset"],
            vfVRest=config["vfVRest"],
            tRefractoryTime=config["tRef"],
            name=config["name"],
            bRecord=config["bRecord"],
            num_cores=config["num_cores"],
            a=config['a'],
            b=config['b'],
            delta_t=config['delta_t'],
            vtTauW=config['vtTauW'],
        )
