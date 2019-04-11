import numpy as np
from ...timeseries import TSContinuous, TSEvent
import multiprocessing

from ..layer import Layer


from typing import Optional, Union, Tuple, List


def s2ms(t): return t * 1000.


def ms2s(t): return t / 1000.


def V2mV(v): return v * 1000.


def mV2V(v): return v / 1000.


COMMAND_GET = 0
COMMAND_SET = 1
COMMAND_RESET = 2
COMMAND_EVOLVE = 3


# - FFIAFNest- Class: define a spiking feedforward layer with spiking outputs
class FFIAFNest(Layer):
    """ FFIAFNest - Class: define a spiking feedforward layer with spiking outputs
    """

    class NestProcess(multiprocessing.Process):

        def __init__(self,
                     requestQ,
                     resultQ,
                     mfW: np.ndarray,
                     vfBias: Union[float, np.ndarray] = 15.,
                     tDt: float = 0.1,
                     vtTauN: Union[float, np.ndarray] = 20.,
                     vfCapacity: Union[float, np.ndarray] = 100.,
                     vfVThresh: Union[float, np.ndarray] = -55.,
                     vfVReset: Union[float, np.ndarray] = -65.,
                     vfVRest: Union[float, np.ndarray] = -65.,
                     tRefractoryTime=1.,
                     bRecord: bool = False,
                     numCores: int = 1):

            multiprocessing.Process.__init__(self, daemon=True)

            self.requestQ = requestQ
            self.resultQ = resultQ

            # - Record neuron parameters
            self.tDt = tDt
            self.vfVThresh = vfVThresh
            self.vfVReset = vfVReset
            self.vfVRest = vfVRest
            self.vtTauN = vtTauN
            self.vfBias = vfBias
            self.vfCapacity = vfCapacity
            self.mfW = mfW
            self.tRefractoryTime = tRefractoryTime
            self.bRecord = bRecord
            self.nSize = np.shape(mfW)[1]
            self.numCores = numCores

        def run(self):

            #### INITIALIZE NEST ####
            import nest

            numCPUs = multiprocessing.cpu_count()
            if self.numCores >= numCPUs:
                self.numCores = numCPUs

            nest.ResetKernel()
            nest.hl_api.set_verbosity("M_FATAL")
            nest.SetKernelStatus(
                {"resolution": s2ms(self.tDt), "local_num_threads": self.numCores})

            self._pop = nest.Create("iaf_psc_exp", self.nSize)

            params = []
            for n in range(self.nSize):
                p = {}

                if type(self.vtTauN) is np.ndarray:
                    p['tau_m'] = s2ms(self.vtTauN[n])
                else:
                    p['tau_m'] = s2ms(self.vtTauN)

                if type(self.vfVThresh) is np.ndarray:
                    p['V_th'] = self.vfVThresh[n]
                else:
                    p['V_th'] = self.vfVThresh

                if type(self.vfVReset) is np.ndarray:
                    p['V_reset'] = self.vfVReset[n]
                else:
                    p['V_reset'] = self.vfVReset

                if type(self.vfVReset) is np.ndarray:
                    p['E_L'] = self.vfVRest[n]
                    p['V_m'] = self.vfVRest[n]
                else:
                    p['E_L'] = self.vfVRest
                    p['V_m'] = self.vfVRest

                if type(self.tRefractoryTime) is np.ndarray:
                    p['t_ref'] = s2ms(self.tRefractoryTime[n])
                else:
                    p['t_ref'] = s2ms(self.tRefractoryTime)

                if type(self.vfBias) is np.ndarray:
                    p['I_e'] = V2mV(self.vfBias[n])
                else:
                    p['I_e'] = V2mV(self.vfBias)

                if type(self.vfCapacity) is np.ndarray:
                    p['C_m'] = self.vfCapacity[n]
                else:
                    p['C_m'] = self.vfCapacity

                params.append(p)

            nest.SetStatus(self._pop, params)

            # - Add spike detector to record layer outputs
            self._sd = nest.Create("spike_detector")
            nest.Connect(self._pop, self._sd)

            # - Add stimulation device
            self._scg = nest.Create(
                "step_current_generator", self.mfW.shape[0])
            nest.Connect(self._scg, self._pop, 'all_to_all',
                         {'weight': self.mfW.T})

            if self.bRecord:
                # - Monitor for recording network potential
                self._mm = nest.Create("multimeter", 1, {'record_from': [
                                       'V_m'], 'interval': s2ms(self.tDt)})
                nest.Connect(self._mm, self._pop)

            ######### DEFINE ICP COMMANDS ######

            def getParam(name):
                vms = nest.GetStatus(self._pop, name)
                return vms

            def setParam(name, value):
                params = []

                for n in range(self.nSize):
                    p = {}
                    if type(value) is np.ndarray:
                        p[name] = value[n]
                    else:
                        p[name] = value

                    params.append(p)

                nest.SetStatus(self._pop, params)

            def reset():
                """
                reset_all - resets time and state
                """

                nest.ResetNetwork()

            def evolve(vtTimeBase,
                       mfInputStep,
                       nNumTimeSteps: Optional[int] = None,
                       ):

                # NEST time starts with 1 (not with 0)

                vtTimeBase = s2ms(vtTimeBase) + 1

                nest.SetStatus(self._scg, [{'amplitude_times': vtTimeBase, 'amplitude_values': V2mV(
                    mfInputStep[:, i])} for i in range(len(self._scg))])

                if nest.GetKernelStatus("time") == 0:
                    # weird behavior of NEST; the recording stops a timestep before the simulation stops. Therefore
                    # the recording has one entry less in the first batch
                    nest.Simulate(nNumTimeSteps * s2ms(self.tDt) + 1)
                else:
                    nest.Simulate(nNumTimeSteps * s2ms(self.tDt))

                # - record states
                if self.bRecord:
                    events = nest.GetStatus(self._mm, 'events')[0]
                    vbUseEvent = events['times'] >= vtTimeBase[0]
                    vms = events['V_m'][vbUseEvent]
                    mfRecordStates = np.reshape(
                        vms, [self.nSize, nNumTimeSteps], order="F")

                # - Build response TimeSeries
                events = nest.GetStatus(self._sd, 'events')[0]
                vbUseEvent = events['times'] >= vtTimeBase[0]
                vtEventTimeOutput = ms2s(events['times'][vbUseEvent])
                vnEventChannelOutput = events['senders'][vbUseEvent]

                # sort spiking response
                order = np.argsort(vtEventTimeOutput)
                vtEventTimeOutput = vtEventTimeOutput[order]
                vnEventChannelOutput = vnEventChannelOutput[order]

                # transform from NEST id to index
                vnEventChannelOutput -= np.min(self._pop)

                if self.bRecord:
                    return [vtEventTimeOutput, vnEventChannelOutput, mfRecordStates]
                else:
                    return [vtEventTimeOutput, vnEventChannelOutput, None]

            ICP_switcher = {COMMAND_GET: getParam,
                            COMMAND_SET: setParam,
                            COMMAND_RESET: reset,
                            COMMAND_EVOLVE: evolve}

            # wait for an IPC command

            while True:
                req = self.requestQ.get()
                func = ICP_switcher.get(req[0])

                result = func(*req[1:])

                if not result is None:
                    self.resultQ.put(result)

    ## - Constructor
    def __init__(
        self,
        mfW: np.ndarray,
        vfBias: Union[float, np.ndarray] = 15.,
        tDt: float = 0.1,
        vtTauN: Union[float, np.ndarray] = 20.,
        vfCapacity: Union[float, np.ndarray] = 100.,
        vfVThresh: Union[float, np.ndarray] = -55.,
        vfVReset: Union[float, np.ndarray] = -65.,
        vfVRest: Union[float, np.ndarray] = -65.,
        tRefractoryTime=1.,
        strName: str = "unnamed",
        bRecord: bool = False,
        nNumCores=1,
    ):
        """
        FFIAFNest - Construct a spiking feedforward layer with IAF neurons, with a NEST back-end
                     Inputs are continuous currents; outputs are spiking events

        :param mfW:             np.array MxN weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10mA

        :param tDt:             float Time-step. Default: 0.1 ms

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms

        :param vfCapacity:       np.array Nx1 vector of neuron membrance capacity. Default: 100 pF

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron reset potential. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron resting potential. Default: -65mV

        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param strName:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions
        """

        if type(mfW) is list:
            mfW = np.asarray(mfW)

        if type(vfBias) is list:
            vfBias = np.asarray(vfBias)

        if type(vtTauN) is list:
            vtTauN = np.asarray(vtTauN)

        if type(vfCapacity) is list:
            vfCapacity = np.asarray(vfCapacity)

        if type(vfVThresh) is list:
            vfVThresh = np.asarray(vfVThresh)

        if type(vfVReset) is list:
            vfVReset = np.asarray(vfVReset)

        if type(vfVRest) is list:
            vfVRest = np.asarray(vfVRest)

        # - Call super constructor (`asarray` is used to strip units)
        super().__init__(
            mfW=np.asarray(mfW),
            tDt=np.asarray(tDt),
            strName=strName,
        )

        self.requestQ = multiprocessing.Queue()
        self.resultQ = multiprocessing.Queue()

        self.nestProcess = self.NestProcess(self.requestQ,
                                            self.resultQ,
                                            mfW,
                                            vfBias,
                                            tDt,
                                            vtTauN,
                                            vfCapacity,
                                            vfVThresh,
                                            vfVReset,
                                            vfVRest,
                                            tRefractoryTime,
                                            bRecord,
                                            nNumCores)

        self.nestProcess.start()

        # - Record neuron parameters
        self._vfVThresh = vfVThresh
        self._vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vfBias = vfBias
        self.vfCapacity = vfCapacity
        self.mfW = mfW
        self._tRefractoryTime = tRefractoryTime
        self.bRecord = bRecord

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """

        self.requestQ.put([COMMAND_SET, "V_m", self.vfVRest])

    def randomize_state(self):
        """ .randomize_state() - Method: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        fRangeV = abs(self.vfVThresh - self.vfVReset)
        randV = (np.random.rand(self._nSize) * fRangeV + self.vfVReset)

        self.requestQ.put([COMMAND_SET, "V_m", randV])

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        print("WARNING: This function resets the whole network")

        self.requestQ.put([COMMAND_RESET])
        self._nTimeStep = 0

    def reset_all(self):
        """
        reset_all - resets time and state
        """

        self.requestQ.put([COMMAND_RESET])
        self._nTimeStep = 0

    # --- State evolution

    def evolve(
        self,
        tsInput: Optional[TSContinuous] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:                TSEvent  output spike series

        """
        # - Prepare time base
        vtTimeBase, mfInputStep, nNumTimeSteps = self._prepare_input(
            tsInput, tDuration, nNumTimeSteps
        )

        self.requestQ.put([COMMAND_EVOLVE, vtTimeBase,
                           mfInputStep, nNumTimeSteps])

        if self.bRecord:
            vtEventTimeOutput, vnEventChannelOutput, self.mfRecordStates = self.resultQ.get()
        else:
            vtEventTimeOutput, vnEventChannelOutput, _ = self.resultQ.get()

        # - Start and stop times for output time series
        tStart = self._nTimeStep * np.asscalar(self.tDt)
        tStop = (self._nTimeStep + nNumTimeSteps) * np.asscalar(self.tDt)

        # - Update layer time step
        self._nTimeStep += nNumTimeSteps

        return TSEvent(
            np.clip(vtEventTimeOutput, tStart, tStop),
            vnEventChannelOutput,
            strName="Layer spikes",
            nNumChannels=self.nSize,
            tStart=tStart,
            tStop=tStop,
        )

    def terminate(self):
        self.nestProcess.terminate()
        self.nestProcess.join()

    ### --- Properties

    @property
    def cOutput(self):
        return TSEvent

    @property
    def tRefractoryTime(self):
        return self._tRefractoryTime

    @property
    def vState(self):
        self.requestQ.put([COMMAND_GET, "V_m"])
        vms = self.resultQ.get()
        return vms

    @vState.setter
    def vState(self, vNewState):

        self.requestQ.put([COMMAND_SET, "V_m", vNewState])

    @property
    def vtTauN(self):
        return self._vtTauN

    @vtTauN.setter
    def vtTauN(self, vtNewTauN):

        self.requestQ.put([COMMAND_SET, "tau_m", s2ms(vtNewTauN)])

    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias):

        self.requestQ.put([COMMAND_SET, "I_e", V2mV(vfNewBias)])

    @property
    def vfVThresh(self):
        return self._vfVThresh

    @vfVThresh.setter
    def vfVThresh(self, vfNewVThresh):

        self.requestQ.put([COMMAND_SET, "V_th", vfNewVThresh])

    @property
    def vfVReset(self):
        return self._vfVReset

    @vfVReset.setter
    def vfVReset(self, vfNewVReset):

        self.requestQ.put([COMMAND_SET, "V_reset", vfNewVReset])

    @property
    def vfVRest(self):
        return self._vfVReset

    @vfVReset.setter
    def vfVRest(self, vfNewVRest):

        self.requestQ.put([COMMAND_SET, "E_L", vfNewVRest])

    @property
    def t(self):
        return self._nTimeStep * np.asscalar(self.tDt)

    @Layer.tDt.setter
    def tDt(self, _):
        raise ValueError("The `tDt` property cannot be set for this layer")


# - RecIAFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
class RecIAFSpkInNest(Layer):
    """ RecIAFSpkInNest- Class: Spiking recurrent layer with spiking in- and outputs
    """

    class NestProcess(multiprocessing.Process):

        def __init__(self,
                     requestQ,
                     resultQ,
                     mfWIn: np.ndarray,
                     mfWRec: np.ndarray,
                     vfBias: Union[float, np.ndarray] = 15.,
                     tDt: float = 0.1,
                     vtTauN: Union[float, np.ndarray] = 20.,
                     vtTauS: Union[float, np.ndarray] = 10.,
                     vfCapacity: Union[float, np.ndarray] = 100.,
                     vfVThresh: Union[float, np.ndarray] = -55.,
                     vfVReset: Union[float, np.ndarray] = -65.,
                     vfVRest: Union[float, np.ndarray] = -65.,
                     tRefractoryTime=1.,
                     bRecord: bool = False,
                     numCores: int = 1):

            multiprocessing.Process.__init__(self, daemon=True)

            self.requestQ = requestQ
            self.resultQ = resultQ

            # - Record neuron parameters
            self.tDt = tDt
            self.vfVThresh = vfVThresh
            self.vfVReset = vfVReset
            self.vfVRest = vfVRest
            self.vtTauN = vtTauN
            self.vfBias = vfBias
            self.vfCapacity = vfCapacity
            self.mfWIn = mfWIn
            self.mfWRec = mfWRec
            self.tRefractoryTime = tRefractoryTime
            self.bRecord = bRecord
            self.nSize = np.shape(mfWRec)[0]
            self.numCores = numCores

        def run(self):

            #### INITIALIZE NEST ####
            import nest

            numCPUs = multiprocessing.cpu_count()
            if self.numCores >= numCPUs:
                self.numCores = numCPUs

            nest.ResetKernel()
            nest.hl_api.set_verbosity("M_FATAL")
            nest.SetKernelStatus(
                {"resolution": s2ms(self.tDt), "local_num_threads": self.numCores})

            self._pop = nest.Create("iaf_psc_exp", self.nSize)

            params = []
            for n in range(self.nSize):
                p = {}

                if type(self.vtTauN) is np.ndarray:
                    p['tau_m'] = s2ms(self.vtTauN[n])
                else:
                    p['tau_m'] = s2ms(self.vtTauN)

                if type(self.vfVThresh) is np.ndarray:
                    p['V_th'] = self.vfVThresh[n]
                else:
                    p['V_th'] = self.vfVThresh

                if type(self.vfVReset) is np.ndarray:
                    p['V_reset'] = self.vfVReset[n]
                else:
                    p['V_reset'] = self.vfVReset

                if type(self.vfVReset) is np.ndarray:
                    p['E_L'] = self.vfVRest[n]
                    p['V_m'] = self.vfVRest[n]
                else:
                    p['E_L'] = self.vfVRest
                    p['V_m'] = self.vfVRest

                if type(self.tRefractoryTime) is np.ndarray:
                    p['t_ref'] = s2ms(self.tRefractoryTime[n])
                else:
                    p['t_ref'] = s2ms(self.tRefractoryTime)

                if type(self.vfBias) is np.ndarray:
                    p['I_e'] = V2mV(self.vfBias[n])
                else:
                    p['I_e'] = V2mV(self.vfBias)

                if type(self.vfCapacity) is np.ndarray:
                    p['C_m'] = self.vfCapacity[n]
                else:
                    p['C_m'] = self.vfCapacity

                params.append(p)

            nest.SetStatus(self._pop, params)

            # - Add spike detector to record layer outputs
            self._sd = nest.Create("spike_detector")
            nest.Connect(self._pop, self._sd)

            # - Add stimulation device
            self._sg = nest.Create("spike_generator", self.mfWIn.shape[0])

            # - Create stimulation connections
            pres = []
            posts = []
            weights = []

            for pre, row in enumerate(self.mfWIn):
                for post, w in enumerate(row):
                    if w == 0:
                        continue
                    pres.append(self._sg[pre])
                    posts.append(self._pop[post])
                    weights.append(w)

            nest.Connect(pres, posts, 'one_to_one')
            nest.SetStatus(nest.GetConnections(self._sg, self._pop), [
                           {'weight': w, 'delay': s2ms(self.tDt)} for w in weights])

            # - Create recurrent connections
            pres = []
            posts = []
            weights = []

            for pre, row in enumerate(self.mfWRec):
                for post, w in enumerate(row):
                    if w == 0:
                        continue
                    pres.append(self._pop[pre])
                    posts.append(self._pop[post])
                    weights.append(w)

            nest.Connect(pres, posts, 'one_to_one')
            nest.SetStatus(nest.GetConnections(self._pop, self._pop), [
                           {'weight': w, 'delay': s2ms(self.tDt)} for w in weights])

            if self.bRecord:
                # - Monitor for recording network potential
                self._mm = nest.Create("multimeter", 1, {'record_from': [
                                       'V_m'], 'interval': s2ms(self.tDt)})
                nest.Connect(self._mm, self._pop)

            ######### DEFINE ICP COMMANDS ######

            def getParam(name):
                vms = nest.GetStatus(self._pop, name)
                return vms

            def setParam(name, value):
                params = []

                for n in range(self.nSize):
                    p = {}
                    if type(value) is np.ndarray:
                        p[name] = value[n]
                    else:
                        p[name] = value

                    params.append(p)

                nest.SetStatus(self._pop, params)

            def reset():
                """
                reset_all - resets time and state
                """

                nest.ResetNetwork()

            def evolve(vtEventTimes,
                       vnEventChannels,
                       nNumTimeSteps: Optional[int] = None,
                       ):

                if len(vnEventChannels > 0):
                    # convert input index to NEST id
                    vnEventChannels += np.min(self._sg)

                    # NEST time starts with 1 (not with 0)
                    nest.SetStatus(self._sg, [{'spike_times': s2ms(
                        vtEventTimes[vnEventChannels == i])} for i in self._sg])

                startTime = nest.GetKernelStatus("time")

                if startTime == 0:
                    # weird behavior of NEST; the recording stops a timestep before the simulation stops. Therefore
                    # the recording has one entry less in the first batch
                    nest.Simulate(nNumTimeSteps * s2ms(self.tDt) + 1)
                else:
                    nest.Simulate(nNumTimeSteps * s2ms(self.tDt))

                # - record states
                if self.bRecord:
                    events = nest.GetStatus(self._mm, 'events')[0]
                    vbUseEvent = events['times'] >= startTime
                    vms = events['V_m'][vbUseEvent]
                    mfRecordStates = np.reshape(
                        vms, [self.nSize, nNumTimeSteps], order="F")

                # - Build response TimeSeries
                events = nest.GetStatus(self._sd, 'events')[0]
                vbUseEvent = events['times'] >= startTime
                vtEventTimeOutput = ms2s(events['times'][vbUseEvent])
                vnEventChannelOutput = events['senders'][vbUseEvent]

                # sort spiking response
                order = np.argsort(vtEventTimeOutput)
                vtEventTimeOutput = vtEventTimeOutput[order]
                vnEventChannelOutput = vnEventChannelOutput[order]

                # transform from NEST id to index
                vnEventChannelOutput -= np.min(self._pop)

                if self.bRecord:
                    return [vtEventTimeOutput, vnEventChannelOutput, mfRecordStates]
                else:
                    return [vtEventTimeOutput, vnEventChannelOutput, None]

            ICP_switcher = {COMMAND_GET: getParam,
                            COMMAND_SET: setParam,
                            COMMAND_RESET: reset,
                            COMMAND_EVOLVE: evolve}

            # wait for an IPC command

            while True:
                req = self.requestQ.get()
                func = ICP_switcher.get(req[0])

                result = func(*req[1:])

                if not result is None:
                    self.resultQ.put(result)

    ## - Constructor
    def __init__(
            self,
            mfWIn: np.ndarray,
            mfWRec: np.ndarray,
            vfBias: np.ndarray = 10.5,
            tDt: float = 0.1,
            vtTauN: np.ndarray = 20.,
            vtTauS: np.ndarray = 50.,
            vfVThresh: np.ndarray = -55.,
            vfVReset: np.ndarray = -65.,
            vfVRest: np.ndarray = -65.,
            vfCapacity: Union[float, np.ndarray] = 100.,
            tRefractoryTime=0,
            strName: str = "unnamed",
            bRecord: bool = False,
            nNumCores: int = 1):
        """
        RecIAFSpkInNest - Construct a spiking recurrent layer with IAF neurons, with a NEST back-end
                           in- and outputs are spiking events

        :param mfWIn:           np.array MxN input weight matrix.
        :param mfWRec:          np.array NxN recurrent weight matrix.
        :param vfBias:          np.array Nx1 bias vector. Default: 10.5mA

        :param tDt:             float Time-step. Default: 0.1 ms

        :param vtTauN:          np.array Nx1 vector of neuron time constants. Default: 20ms
        :param vtTauS:          np.array Nx1 vector of synapse time constants. Default: 20ms

        :param vfVThresh:       np.array Nx1 vector of neuron thresholds. Default: -55mV
        :param vfVReset:        np.array Nx1 vector of neuron reset potential. Default: -65mV
        :param vfVRest:         np.array Nx1 vector of neuron resting potential. Default: -65mV

        :param vfCapacity:       np.array Nx1 vector of neuron membrance capacity. Default: 100 pF
        :param tRefractoryTime: float Refractory period after each spike. Default: 0ms

        :param strName:         str Name for the layer. Default: 'unnamed'

        :param bRecord:         bool Record membrane potential during evolutions
        """

        if type(mfWIn) is list:
            mfWIn = np.asarray(mfWIn)

        if type(mfWRec) is list:
            mfWRec = np.asarray(mfWRec)

        if type(vfBias) is list:
            vfBias = np.asarray(vfBias)

        if type(vtTauN) is list:
            vtTauN = np.asarray(vtTauN)

        if type(vfCapacity) is list:
            vfCapacity = np.asarray(vfCapacity)

        if type(vfVThresh) is list:
            vfVThresh = np.asarray(vfVThresh)

        if type(vfVReset) is list:
            vfVReset = np.asarray(vfVReset)

        if type(vfVRest) is list:
            vfVRest = np.asarray(vfVRest)

        # - Call super constructor (`asarray` is used to strip units)

        # TODO this does not make much sense (mfW <- mfWIn)
        super().__init__(
            mfW=np.asarray(mfWIn),
            tDt=tDt,
            strName=strName,
        )

        self.requestQ = multiprocessing.Queue()
        self.resultQ = multiprocessing.Queue()

        self.nestProcess = self.NestProcess(self.requestQ,
                                            self.resultQ,
                                            mfWIn,
                                            mfWRec,
                                            vfBias,
                                            tDt,
                                            vtTauN,
                                            vtTauS,
                                            vfCapacity,
                                            vfVThresh,
                                            vfVReset,
                                            vfVRest,
                                            tRefractoryTime,
                                            bRecord,
                                            nNumCores)

        self.nestProcess.start()

        # - Record neuron parameters
        self._vfVThresh = vfVThresh
        self._vfVReset = vfVReset
        self.vfVRest = vfVRest
        self.vtTauN = vtTauN
        self.vfBias = vfBias
        self.vfCapacity = vfCapacity
        self.mfWIn = mfWIn
        self.mfWRec = mfWRec
        self._tRefractoryTime = tRefractoryTime
        self.bRecord = bRecord

    def reset_state(self):
        """ .reset_state() - Method: reset the internal state of the layer
            Usage: .reset_state()
        """

        self.requestQ.put([COMMAND_SET, "V_m", self.vfVRest])

    def randomize_state(self):
        """ .randomize_state() - Method: randomize the internal state of the layer
            Usage: .randomize_state()
        """
        fRangeV = abs(self.vfVThresh - self.vfVReset)
        randV = (np.random.rand(self.nSize) * fRangeV + self.vfVReset)

        self.requestQ.put([COMMAND_SET, "V_m", randV])

    def reset_time(self):
        """
        reset_time - Reset the internal clock of this layer
        """

        print("WARNING: This function resets the whole network")

        self.requestQ.put([COMMAND_RESET])
        self._nTimeStep = 0

    def reset_all(self):
        """
        reset_all - resets time and state
        """

        self.requestQ.put([COMMAND_RESET])
        self._nTimeStep = 0

    # --- State evolution

    def evolve(
        self,
        tsInput: Optional[TSContinuous] = None,
        tDuration: Optional[float] = None,
        nNumTimeSteps: Optional[int] = None,
        bVerbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:      TSContinuous  Input spike trian
        :param tDuration:       float    Simulation/Evolution time
        :param nNumTimeSteps    int      Number of evolution time steps
        :param bVerbose:        bool     Currently no effect, just for conformity
        :return:                TSEvent  output spike series

        """

        # - Prepare time base
        nNumTimeSteps = self._determine_timesteps(
            tsInput, tDuration, nNumTimeSteps)

        # - Generate discrete time base
        vtTimeBase = self._gen_time_trace(self.t, nNumTimeSteps)

        # - Set spikes for spike generator
        if tsInput is not None:
            vtEventTimes, vnEventChannels, _ = tsInput.find(
                [vtTimeBase[0], vtTimeBase[-1] + self.tDt])

        else:
            vtEventTimes = np.array([])
            vnEventChannels = np.array([])

        self.requestQ.put([COMMAND_EVOLVE, vtEventTimes,
                           vnEventChannels, nNumTimeSteps])

        if self.bRecord:
            vtEventTimeOutput, vnEventChannelOutput, self.mfRecordStates = self.resultQ.get()
        else:
            vtEventTimeOutput, vnEventChannelOutput, _ = self.resultQ.get()

        # - Start and stop times for output time series
        tStart = self._nTimeStep * self.tDt
        tStop = (self._nTimeStep + nNumTimeSteps) * self.tDt

        # - Update layer time step
        self._nTimeStep += nNumTimeSteps

        return TSEvent(
            np.clip(vtEventTimeOutput, tStart, tStop),
            vnEventChannelOutput,
            strName="Layer spikes",
            nNumChannels=self.nSize,
            tStart=tStart,
            tStop=tStop,
        )

    def terminate(self):
        self.nestProcess.terminate()
        self.nestProcess.join()

    ### --- Properties

    @property
    def cInput(self):
        return TSEvent

    @property
    def cOutput(self):
        return TSEvent

    @property
    def tRefractoryTime(self):
        return self._tRefractoryTime

    @property
    def vState(self):
        self.requestQ.put([COMMAND_GET, "V_m"])
        vms = self.resultQ.get()
        return vms

    @vState.setter
    def vState(self, vNewState):

        self.requestQ.put([COMMAND_SET, "V_m", vNewState])

    @property
    def vtTauN(self):
        return self._vtTauN

    @vtTauN.setter
    def vtTauN(self, vtNewTauN):

        self.requestQ.put([COMMAND_SET, "tau_m", s2ms(vtNewTauN)])

    @property
    def vfBias(self):
        return self._vfBias

    @vfBias.setter
    def vfBias(self, vfNewBias):

        self.requestQ.put([COMMAND_SET, "I_e", V2mV(vfNewBias)])

    @property
    def vfVThresh(self):
        return self._vfVThresh

    @vfVThresh.setter
    def vfVThresh(self, vfNewVThresh):

        self.requestQ.put([COMMAND_SET, "V_th", vfNewVThresh])

    @property
    def vfVReset(self):
        return self._vfVReset

    @vfVReset.setter
    def vfVReset(self, vfNewVReset):

        self.requestQ.put([COMMAND_SET, "V_reset", vfNewVReset])

    @property
    def vfVRest(self):
        return self._vfVReset

    @vfVReset.setter
    def vfVRest(self, vfNewVRest):

        self.requestQ.put([COMMAND_SET, "E_L", vfNewVRest])

    @property
    def t(self):
        return self._nTimeStep * self.tDt

    @Layer.tDt.setter
    def tDt(self, _):
        raise ValueError("The `tDt` property cannot be set for this layer")
