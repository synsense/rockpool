# ----
# dynap_hw.py - Implementation of HW FF and Rec layers for DynapSE, via ctxCTL
# ----

from ...layer import Layer
from ....timeseries import TSEvent
from ....devices.dynapse_control_extd import DynapseControlExtd
from ....devices import dynapse_control as DC

import numpy as np
from warnings import warn
from typing import List, Optional, Generator
import time


# - Default timestep
DEF_TIMESTEP = 2e-5

# - Absolute tolerance, e.g. for comparing float values
ABS_TOLERANCE = 1e-9

# -- Define the HW layer class for recurrent networks
class RecDynapSE(Layer):
    """
    RecDynapSE - Recurrent layer implemented on DynapSE
    """

    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        neuron_ids: Optional[np.ndarray] = None,
        virtual_neuron_ids: Optional[np.ndarray] = None,
        dt: Optional[float] = DEF_TIMESTEP,
        max_trials_batch: Optional[float] = None,
        max_batch_dur: Optional[float] = None,
        max_num_timesteps: Optional[int] = None,
        max_num_events_batch: Optional[int] = None,
        l_input_core_ids: List[int] = [0],
        input_chip_id: int = 0,
        clearcores_list: Optional[list] = None,
        controller: DynapseControlExtd = None,
        name: Optional[str] = "unnamed",
        skip_weights: bool = False,
    ):
        """
        RecDynapSE - Recurrent layer implemented on DynapSE

        :param weights_in:               ndarray[int] MxN matrix of input weights from virtual to hardware neurons
        :param weights_rec:              ndarray[int] NxN matrix of weights between hardware neurons.
                                                 Supplied in units of synaptic connection. Negative elements
                                                 lead to inhibitory synapses
        :param neuron_ids:    ndarray  1D array of IDs of N hardware neurons that are to be used as layer neurons
        :param virtual_neuron_ids:  ndarray  1D array of IDs of M virtual neurons that are to be used as input neurons
        :param dt:                 float   Time-step.
        :param max_trials_batch:  int  Maximum number of trials (specified in input timeseries) per batch.
                                         Longer evolution periods will automatically split in smaller batches.
        :param max_batch_dur:        float  Maximum duration of single evolution batch.
        :param max_num_timesteps:    float  Maximum number of time steps of of single evolution batch.
        :param max_num_events_batch:  float  Maximum number of input events per evolution batch.
        :param l_input_core_ids:      array-like  IDs of the cores that contain neurons receiving external inputs.
                                                To avoid ID collisions neurons on these cores should not receive inputs
                                                from other neurons.
        :param input_chip_id:        int  ID of the chip with neurons that receive external input.
        :param clearcores_list:        list or None  IDs of chips where configurations should be cleared.
        :param controller:          DynapseControl object to interface the hardware
        :param name:             str     Layer name
        :param skip_weights:        bool    Do not upload weight configuration to chip. (Use carecully)
        """

        # - Instantiate DynapseControl
        if controller is None:
            if dt is None:
                raise ValueError(
                    "Layer `{}` Either dt or controller must be provided".format(name)
                )
            self.controller = DynapseControlExtd(dt, clearcores_list)
        else:
            self.controller = controller
            self.controller.fpga_isibase = dt
            self.controller.clear_connections(clearcores_list)

        # - Check supplied arguments
        assert (
            weights_rec.shape[0] == weights_rec.shape[1]
        ), "Layer `{}`: The recurrent weight matrix `mnWRec` must be square.".format(
            name
        )

        # - Initialise superclass
        super().__init__(
            weights=np.asarray(np.round(weights_in), "int"), dt=dt, name=name
        )
        print("Layer `{}`: Superclass initialized".format(name))

        # - Check weight matrices
        assert (
            weights_in.shape[1] == weights_rec.shape[0]
        ), "Layer `{}`: `mnWIn` and `mnWRec` must have compatible shapes: `mnWIn` is MxN, `mnWRec` is NxN.".format(
            name
        )

        # - Store weight matrices
        self.weights_in = weights_in
        self.weights_rec = weights_rec
        # - Record input core mask and chip ID
        self._input_coremask = int(np.sum([2 ** nID for nID in l_input_core_ids]))
        self._input_chip_id = input_chip_id
        # - Store evolution batch size limitations
        self.max_trials_batch = max_trials_batch
        self.max_num_events_batch = (
            self.controller.fpga_event_limit
            if max_num_events_batch is None
            else max_num_events_batch
        )
        if max_num_timesteps is not None:
            if max_batch_dur is not None:
                warn(
                    "Layer `{}`: Caution: If both `max_num_timesteps` and `max_batch_dur` are provided, only `max_num_timesteps` is considered.".format(
                        name
                    )
                )
            self.max_num_timesteps = max_num_timesteps
        else:
            self.max_batch_dur = max_batch_dur

        # - Allocate layer neurons
        self._hw_neurons, self._shadow_neurons = (
            self.controller.allocate_hw_neurons(self.size)
            if neuron_ids is None
            else self.controller.allocate_hw_neurons(neuron_ids)
        )
        # Make sure number of neurons is correct
        assert (
            self._hw_neurons.size == self.size
        ), "Layer `{}`: `neuron_ids` must be of size {} or None.".format(
            name, self.size
        )
        # - Keep list of neuron IDs
        self._neuron_ids = np.array([neuron.get_id() for neuron in self._hw_neurons])
        print("Layer `{}`: Layer neurons allocated".format(name))

        # - Allocate virtual neurons
        self._virtual_neurons = (
            self.controller.allocate_virtual_neurons(self.size_in)
            if virtual_neuron_ids is None
            else self.controller.allocate_virtual_neurons(virtual_neuron_ids)
        )
        # Make sure number of neurons is correct
        assert (
            self._virtual_neurons.size == self.size_in
        ), "Layer `{}`: `virtual_neuron_ids` must be of size {} or None.".format(
            name, self.size_in
        )
        # - Keep list of neuron IDs
        self._virtual_neuron_ids = np.array(
            [neuron.get_neuron_id() for neuron in self._virtual_neurons]
        )
        print("Layer `{}`: Virtual neurons allocated".format(name))

        # - Store recurrent weights
        self._weights_rec = np.asarray(np.round(weights_rec), int)

        if not skip_weights:
            # - Configure connectivity
            self._compile_weights_and_configure()

        print("Layer `{}` prepared.".format(self.name))

    def _batch_input_data(
        self, ts_input: TSEvent, num_timesteps: int, verbose: bool = False
    ) -> (np.ndarray, int):
        """_batch_input_data: Generator that returns the data in batches"""
        # - Time points of input trace in discrete layer time base
        vn_tpts_evts_inp = np.floor(ts_input.times / self.dt).astype(int)
        # - Make sure evolution is within correct interval
        start_idx_all = np.searchsorted(vn_tpts_evts_inp, self._timestep)
        end_idx_all = np.searchsorted(vn_tpts_evts_inp, self._timestep + num_timesteps)
        vn_tpts_evts_inp = vn_tpts_evts_inp[start_idx_all:end_idx_all]
        vn_channels_inp = ts_input.channels[start_idx_all:end_idx_all]
        # vn_channels_inp = ts_input.channels

        # - Check whether data for splitting by trial is available
        if hasattr(ts_input, "vtTrialStarts") and self.max_trials_batch is not None:
            ## -- Split by trials
            vn_trial_starts = np.floor(ts_input.vtTrialStarts / self.dt).astype(int)
            # - Make sure only trials within evolution period are considered
            vn_trial_starts = vn_trial_starts[
                np.logical_and(
                    self._timestep <= vn_trial_starts,
                    vn_trial_starts < self._timestep + num_timesteps,
                )
            ]
            # - Total number of trials
            num_trials = vn_trial_starts.size
            # - Array indices of ts_input.times and ts_input.channels where trials start
            v_trialstart_idcs = np.searchsorted(vn_tpts_evts_inp, vn_trial_starts)
            # - Count number of events for each trial (np.r_ to include last trial)
            vn_cumul_evts_trial = np.r_[v_trialstart_idcs, vn_tpts_evts_inp.size]

            # - First trial of current batch
            idx_current_trial = 0
            while idx_current_trial < num_trials:
                # - Cumulated numbers of events per trial for coming trials
                vn_cumul_next_evts = (
                    vn_cumul_evts_trial[idx_current_trial + 1 :]
                    - vn_cumul_evts_trial[idx_current_trial]
                )
                max_num_trials_e = np.searchsorted(
                    vn_cumul_next_evts, self.max_num_events_batch
                )
                if self.max_num_timesteps is not None:
                    # - Maximum number of trials before self.max_num_timesteps is exceeded
                    max_num_trials_mnts = np.searchsorted(
                        vn_cumul_next_evts, self.max_num_timesteps
                    )
                else:
                    max_num_trials_mnts = np.inf
                # - Number of trials to be used in current batch, considering max. number of trials per batch,
                #   events per batch and (if applies) time steps per batch
                num_trials_batch = min(
                    self.max_trials_batch, max_num_trials_e, max_num_trials_mnts
                )
                assert num_trials_batch > 0, (
                    "Layer `{}`: Cannot continue evolution. ".format(self.name)
                    + "Either too many timesteps or events in this trial."
                )
                # - Start and end time steps and indices (wrt vn_tpts_evts_inp) of this batch
                tstp_start_batch: int = vn_trial_starts[idx_current_trial]
                idx_start_batch: int = v_trialstart_idcs[idx_current_trial]
                try:
                    tstp_end_batch: int = vn_trial_starts[
                        idx_current_trial + num_trials_batch
                    ]
                    idx_end_batch: int = v_trialstart_idcs[
                        idx_current_trial + num_trials_batch
                    ]
                except IndexError as e:
                    # - If index error occurs because final batch is included
                    if idx_current_trial + num_trials_batch == v_trialstart_idcs.size:
                        idx_end_batch = vn_tpts_evts_inp.size
                        tstp_end_batch = num_timesteps + self._timestep
                    else:
                        raise e
                # - Event data to be sent to FPGA
                vn_tpts_evts_inp_batch = vn_tpts_evts_inp[idx_start_batch:idx_end_batch]
                vn_chnls_inp_batch = vn_channels_inp[idx_start_batch:idx_end_batch]
                num_tstps_batch = (
                    tstp_end_batch - tstp_start_batch
                )  # This is not the same as vn_tpts_evts_inp_batch.size as the latter only contains events and not the complete time base
                if verbose:
                    num_evts_batch = idx_end_batch - idx_start_batch
                    print(
                        "Layer `{}`: Current batch input: {} s ({} timesteps)".format(
                            self.name, num_tstps_batch * self.dt, num_tstps_batch
                        )
                        + ", {} events, trials {} to {} of {}".format(
                            num_evts_batch,
                            idx_current_trial + 1,
                            idx_current_trial + num_trials_batch,
                            num_trials,
                        )
                    )
                yield (
                    vn_tpts_evts_inp_batch,
                    vn_chnls_inp_batch,
                    tstp_start_batch,
                    num_tstps_batch * self.dt,
                )
                idx_current_trial += num_trials_batch
        else:
            ## -- Split by Maximum number of events and time steps
            # - Handle None for max_num_timesteps
            max_num_timesteps = (
                num_timesteps
                if self.max_num_timesteps is None
                else self.max_num_timesteps
            )
            # - Time step at which current batch starts
            tstp_start_batch = self._timestep
            # - Corresponding index wrt vn_tpts_evts_inp
            idx_start_batch = 0
            # - Time step after evolution ends
            tstp_end_evol = tstp_start_batch + num_timesteps
            while tstp_start_batch < tstp_end_evol:
                # - Endpoint of current batch
                tstp_end_batch = min(
                    tstp_start_batch + max_num_timesteps, tstp_end_evol
                )
                # - Corresponding intex wrt vn_tpts_evts_inp
                idx_end_batch = np.searchsorted(vn_tpts_evts_inp, tstp_end_batch)
                # - Correct if too many events are included
                if idx_end_batch - idx_start_batch > self.max_num_events_batch:
                    idx_end_batch = idx_start_batch + self.max_num_events_batch
                    tstp_end_batch = vn_tpts_evts_inp[idx_end_batch]
                # - Event data to be sent to FPGA
                vn_tpts_evts_inp_batch = vn_tpts_evts_inp[idx_start_batch:idx_end_batch]
                vn_chnls_inp_batch = vn_channels_inp[idx_start_batch:idx_end_batch]
                num_tstps_batch = (
                    tstp_end_batch - tstp_start_batch
                )  # This is not the same as vn_tpts_evts_inp_batch.size as the latter only contains events and not the complete time base
                if verbose:
                    num_evts_batch = idx_end_batch - idx_start_batch
                    print(
                        "Layer `{}`: Current batch input: {} s ({} timesteps)".format(
                            self.name, num_tstps_batch * self.dt, num_tstps_batch
                        )
                        + ", {} events, from {} s to {} s of {} s".format(
                            num_evts_batch,
                            tstp_start_batch * self.dt,
                            tstp_end_batch * self.dt,
                            num_timesteps * self.dt,
                        )
                    )
                yield (
                    vn_tpts_evts_inp_batch,
                    vn_chnls_inp_batch,
                    tstp_start_batch,
                    num_tstps_batch * self.dt,
                )
                tstp_start_batch = tstp_end_batch
                idx_start_batch = idx_end_batch

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = True,
    ) -> TSEvent:
        """
        evolve - Evolve the layer by queueing spikes, stimulating and recording

        :param ts_input:         TSEvent input time series, containing `self.size` channels
        :param duration:       float   Desired evolution duration, in seconds
        :param num_timesteps:   int     Desired evolution duration, in integer steps of `self.dt`
        :param verbose:        bool    Output information on evolution progress

        :return:                TSEvent spikes emitted by the neurons in this layer, during the evolution time
        """
        # - Compute duration for evolution
        if num_timesteps is None:
            # - Determine num_timesteps
            if duration is None:
                # - Determine duration
                assert (
                    ts_input is not None
                ), "Layer `{}`: One of `num_timesteps`, `ts_input` or `duration` must be supplied".format(
                    self.name
                )

                if ts_input.periodic:
                    # - Use duration of periodic TimeSeries, if possible
                    duration = ts_input.duration

                else:
                    # - Evolve until the end of the input TImeSeries
                    duration = ts_input.t_stop - self.t
                    assert duration > 0, (
                        "Layer `{}`: Cannot determine an appropriate evolution duration.".format(
                            self.name
                        )
                        + " `ts_input` finishes before the current evolution time."
                    )
            num_timesteps = int(np.floor((duration + ABS_TOLERANCE) / self.dt))
        else:
            assert isinstance(
                num_timesteps, int
            ), "Layer `{}`: num_timesteps must be of type int.".format(self.name)
            duration = num_timesteps * self.dt

        # - Lists for storing recorded data
        times = []
        channels = []

        # - Generator that splits inupt into batches
        input_gen: Generator = self._batch_input_data(
            # - Clip ts_input to required duration
            ts_input.clip([self.t, self.t + duration]),
            num_timesteps,
            verbose,
        )

        # - Iterate over input batches
        for (
            vn_tpts_evts_inp_batch,
            vn_chnls_inp_batch,
            tstp_start_batch,
            dur_batch,
        ) in input_gen:
            vtTimeTraceBatch, vnChannelsBatch = self._send_batch(
                timesteps=vn_tpts_evts_inp_batch - tstp_start_batch,
                vnChannels=vn_chnls_inp_batch,
                dur_batch=dur_batch,
            )

            channels.append(vnChannelsBatch)
            times.append(vtTimeTraceBatch + tstp_start_batch * self.dt)
            if verbose:
                print("Layer `{}`: Received event data".format(self.name))

        # - Flatten out times and channels
        times = [t for vThisTrace in times for t in vThisTrace]
        channels = [ch for vTheseChannels in channels for ch in vTheseChannels]

        # - Convert recorded events into TSEvent object
        tsResponse = TSEvent(
            times,
            channels,
            t_start=self.t,
            t_stop=self.t + self.dt * num_timesteps,
            num_channels=self.size,
            name="DynapSE spikes",
        )

        # - Set layer time
        self._timestep += num_timesteps

        if verbose:
            print("Layer `{}`: Evolution successful.".format(self.name))

        return tsResponse

    def _send_batch(
        self, timesteps: np.ndarray, vnChannels: np.ndarray, dur_batch: float
    ):
        try:
            vtTimeTraceOut, vnChannelsOut = self.controller.send_arrays(
                times=timesteps,
                channels=vnChannels,
                t_record=dur_batch,
                neuron_ids=self.virtual_neuron_ids,
                record_neur_ids=self.neuron_ids,
                targetcore_mask=self._input_coremask,
                targetchip_id=self._input_chip_id,
                periodic=False,
                record=True,
                return_ts=False,
            )
        # - It can happen that DynapseControl inserts dummy events to make sure ISI limit is not exceeded.
        #   This may result in too many events in single batch, in which case a MemoryError is raised.
        except ValueError:
            print(
                "Layer `{}`: Split current batch into two, due to large number of events.".format(
                    self.name
                )
            )
            ## -- Split the batch in two parts, then set it together
            # - Total number of time steps in batch
            nNumTSBatch = int(np.round(dur_batch / self.dt))
            # - Number of time steps and durations of first and second part:
            nNumTSPart1 = nNumTSBatch // 2
            nNumTSPart2 = nNumTSBatch - nNumTSPart1
            tplDurations = (self.dt * nNumTSPart1, self.dt * nNumTSPart2)
            # - Determine where to split arrays of input time steps and channels
            iSplit = np.searchsorted(timesteps, nNumTSPart1)
            # - Event time steps for each part
            tplvnTimeSteps = (timesteps[:iSplit], timesteps[iSplit:] - nNumTSPart1)
            # - Event channels for each part$
            tplvnChannels = (vnChannels[:iSplit], vnChannels[iSplit:])
            # - Evolve the two parts. lOutputs is list with two tuples, each tuple corresponding to one part
            lOutputs = [
                list(  # - Wrap in list to be able to modify elements (add time) later on
                    self._send_batch(timesteps=vnTS, vnChannels=vnC, dur_batch=tDur)
                )
                for vnTS, vnC, tDur in zip(tplvnTimeSteps, tplvnChannels, tplDurations)
            ]
            # - Correct time of second part
            lOutputs[-1][0] += tplDurations[0]
            # - Separate output into arrays
            vtTimeTraceOut, vnChannelsOut = [
                np.hstack(tplOut) for tplOut in zip(*lOutputs)
            ]

        return vtTimeTraceOut, vnChannelsOut

    def _compile_weights_and_configure(self):
        """
        _compile_weights_and_configure - Configure DynapSE weights from the weight matrices
        """

        # - Connect virtual neurons to hardware neurons
        self.controller.set_virtual_connections_from_weights(
            weights=self.weights_in,
            virtualneuron_ids=self.virtual_neuron_ids,
            hwneuron_ids=self.neuron_ids,
            syn_exc=self.controller.syn_exc_fast,
            syn_inh=self.controller.syn_inh_fast,
            apply_diff=False,
        )
        print(
            "Layer `{}`: Connections to virtual neurons have been set.".format(
                self.name
            )
        )

        ## -- Set connections wihtin hardware layer

        # - Infer which neurons are "input neurons" (i.e. neurons that receive input from a virtual neuron)
        vbInputNeurons = (self.weights_in != 0).any(axis=0)

        # - Connections from input neurons to remaining neurons
        mnWInToRec = self.weights.copy()
        mnWInToRec[vbInputNeurons == False] = 0
        self.controller.set_connections_from_weights(
            weights=mnWInToRec,
            hwneuron_ids=self.neuron_ids,
            syn_exc=self.controller.syn_exc_fast,
            syn_inh=self.controller.syn_inh_fast,
            apply_diff=False,
        )
        print(
            "Layer `{}`: Connections from input neurons to reservoir have been set.".format(
                self.name
            )
        )

        # - Connections going out from neurons that are not input neurons
        mnWRec = self.weights.copy()
        mnWRec[vbInputNeurons] = 0
        self.controller.set_connections_from_weights(
            weights=mnWRec,
            hwneuron_ids=self.neuron_ids,
            syn_exc=self.controller.syn_exc_slow,
            syn_inh=self.controller.syn_inh_fast,
            apply_diff=True,
        )
        print("Layer `{}`: Recurrent connections have been set.".format(self.name))

    @property
    def input_type(self):
        return TSEvent

    @property
    def output_type(self):
        return TSEvent

    @property
    def weights_in(self):
        return self._weights_in

    @weights_in.setter
    def weights_in(self, mfNewW):
        self._weights_in = np.round(
            self._expand_to_shape(
                mfNewW, (self.size_in, self.size), "weights_in", bAllowNone=False
            )
        ).astype(int)

    @property
    def weights_rec(self):
        return self._weights_rec

    @weights_rec.setter
    def weights_rec(self, mfNewW):
        self._weights_rec = np.round(
            self._expand_to_shape(
                mfNewW, (self.size, self.size), "weights_rec", bAllowNone=False
            )
        ).astype(int)

    # weights as alias for weights_rec
    @property
    def weights(self):
        return self.weights_rec

    @weights.setter
    def weights(self, mfNewW):
        self.weights_rec = mfNewW

    # _mfW as alias for _weights_rec
    @property
    def _mfW(self):
        return self._weights_rec

    @_mfW.setter
    def _mfW(self, mfNewW):
        self._weights_rec = mfNewW

    @property
    def max_trials_batch(self):
        return self._nMaxTrialsPerBatch

    @max_trials_batch.setter
    def max_trials_batch(self, nNewMax):
        assert nNewMax is None or (
            type(nNewMax) == int and 0 < nNewMax
        ), "Layer `{}`: max_trials_batch must be an integer greater than 0 or None.".format(
            self.name
        )
        self._nMaxTrialsPerBatch = nNewMax

    @property
    def max_num_timesteps(self):
        return self._nMaxNumTimeSteps

    @max_num_timesteps.setter
    def max_num_timesteps(self, nNewMax):
        assert nNewMax is None or (
            type(nNewMax) == int and 0 < nNewMax
        ), "Layer `{}`: max_num_timesteps must be an integer greater than 0 or None.".format(
            self.name
        )
        if nNewMax > self.controller.fpga_event_limit * self.controller.fpga_isi_limit:
            warn(
                "Layer `{}`: max_num_timesteps is larger than fpga_event_limit * fpga_isi_limit ({}).".format(
                    self.name,
                    self.controller.fpga_event_limit * self.controller.fpga_isi_limit,
                )
            )
        self._nMaxNumTimeSteps = nNewMax

    @property
    def max_batch_dur(self):
        return (
            None if self._nMaxNumTimeSteps is None else self._nMaxNumTimeSteps * self.dt
        )

    @max_batch_dur.setter
    def max_batch_dur(self, tNewMax):
        assert tNewMax is None or (
            type(tNewMax) == int and 0 < tNewMax
        ), "Layer `{}`: max_batch_dur must be an integer greater than 0 or None.".format(
            self.name
        )
        self._nMaxNumTimeSteps = (
            None if tNewMax is None else int(np.round(tNewMax / self.dt))
        )

    @property
    def max_num_events_batch(self):
        return self._nMaxEventsPerBatch

    @max_num_events_batch.setter
    def max_num_events_batch(self, nNewMax):
        assert (
            type(nNewMax) == int and 0 < nNewMax <= self.controller.fpga_event_limit
        ), "Layer `{}`: max_num_events_batch must be an integer between 0 and {}.".format(
            self.name, self.controller.fpga_event_limit
        )
        self._nMaxEventsPerBatch = nNewMax

    @property
    def virtual_neuron_ids(self):
        return self._virtual_neuron_ids

    @property
    def neuron_ids(self):
        return self._neuron_ids

    @property
    def l_input_core_ids(self):
        # - Core mask as reversed binary string
        strBinCoreMask = reversed(bin(self._input_coremask)[-4:])
        return [nCoreID for nCoreID, bMask in enumerate(strBinCoreMask) if int(bMask)]


# -- Define subclass of RecDynapSE to be used in demos with preloaded events
class RecDynapSEDemo(RecDynapSE):
    """
    RecDynapSE - Recurrent layer implemented on DynapSE, using preloaded events during evolution
    """

    def __init__(self, *args, **kwargs):

        # - Call parent initializer
        super().__init__(*args, **kwargs)

        # - Set up filter for recording spikes
        self.controller.add_buffered_event_filter(self.neuron_ids)

    def load_events(self, tsAS, vtRhythmStart, tTotalDuration: float):
        if tsAS.times.size > self.controller.sram_event_limit:
            raise MemoryError(
                "Layer `{}`: Can upload at most {} events. {} are too many.".format(
                    self.name, self.controller.sram_event_limit, tsAS.times.size
                )
            )

        # - Indices corresponding to first event of each rhythm
        viRhythmStarts = np.searchsorted(tsAS.times, vtRhythmStart)

        # - Durations of each rhythm
        self.vtRhythmDurations = np.diff(np.r_[vtRhythmStart, tTotalDuration])

        # - Convert timeseries to events for FPGA
        lEvents = self.controller._TSEvent_to_spike_list(
            series=tsAS,
            neuron_ids=self.virtual_neuron_ids,
            targetcore_mask=1,
            targetchip_id=0,
        )
        # - Set interspike interval of first event of each rhythm so that it corresponds
        #   to the rhythm start and not the last event from the previous rhythm
        for iRhythm, iEvent in enumerate(viRhythmStarts):
            lEvents[iEvent].isi = np.round(
                (tsAS.times[iEvent] - vtRhythmStart[iRhythm])
                / self.controller.fpga_isibase
            ).astype(int)

        print(
            "Layer `{}`: {} events have been generated.".format(self.name, len(lEvents))
        )

        # - Upload input events to processor
        iEvent = 0
        while iEvent < tsAS.times.size:
            self.controller.fpga_spikegen.set_base_addr(2 * iEvent)
            self.controller.fpga_spikegen.preload_stimulus(
                lEvents[
                    iEvent : min(
                        iEvent + self.controller.fpga_event_limit, len(lEvents)
                    )
                ]
            )
            iEvent += self.controller.fpga_event_limit
        print("Layer `{}`: Events have been loaded.".format(self.name))

        # - Fpga adresses where beats start
        self.vnRhythmAddress = 2 * (viRhythmStarts)
        # - Number of events per rhythm: do -1 for each rhythm, due to ctxctl syntax
        self.vnEventsPerRhythm = np.diff(np.r_[viRhythmStarts, len(lEvents)]) - 1

    # @profile
    def evolve(
        self, iRhythm: int, duration: Optional[float] = None, verbose: bool = True
    ) -> TSEvent:
        """
        evolve - Evolve the layer by playing back from the given base address and recording

        :param iRhythm:     int     Index of the rhythm to be played back
        :param duration:   float   Desired evolution duration, in seconds, use rhythm duration if None
        :param nNumEvents:  int     Number of input events to play back on FPGA starting from base address

        :return:                TSEvent spikes emitted by the neurons in this layer, during the evolution time
        """

        # - Determine evolution duration
        duration = self.vtRhythmDurations[iRhythm] if duration is None else duration
        num_timesteps = int(np.floor((duration + ABS_TOLERANCE) / self.dt))

        # - Instruct FPGA to spike
        # set new base adress and number of input events for stimulation
        self.controller.fpga_spikegen.set_base_addr(self.vnRhythmAddress[iRhythm])
        self.controller.fpga_spikegen.set_stim_count(self.vnEventsPerRhythm[iRhythm])

        # - Lists for storing collected events
        lnTimeStamps = []
        lnChannels = []
        lTriggerEvents = []

        # - Clear event filter
        self.controller.bufferedfilter.get_events()
        self.controller.bufferedfilter.get_special_event_timestamps()

        # - Time at which stimulation stops
        t_stop = time.time() + duration

        # Start stimulation
        self.controller.fpga_spikegen.start()

        # - Until duration is over, record events and process in quick succession
        while time.time() < t_stop:
            # - Collect events and possibly trigger events
            lTriggerEvents += (
                self.controller.bufferedfilter.get_special_event_timestamps()
            )
            lCurrentEvents = self.controller.bufferedfilter.get_events()

            vtTimeStamps, vnChannels = DC.event_data_to_channels(
                lCurrentEvents, self.neuron_ids
            )
            lnTimeStamps += list(vtTimeStamps)
            lnChannels += list(vnChannels)

        print(
            "Layer `{}`: Recorded {} event(s) and {} trigger event(s)".format(
                self.name, len(lnTimeStamps), len(lTriggerEvents)
            )
        )

        # - Post-processing of collected events
        vtTimeTrace = np.array(lnTimeStamps) * 1e-6
        vnChannels = np.array(lnChannels)

        # - Locate synchronisation timestamp
        vtStartTriggers = np.array(lTriggerEvents) * 1e-6
        viStartIndices = np.searchsorted(vtTimeTrace, vtStartTriggers)
        viEndIndices = np.searchsorted(vtTimeTrace, vtStartTriggers + duration)
        # - Choose first trigger where start and end indices not equal. If not possible, take first trigger
        try:
            iTrigger = np.argmax((viEndIndices - viStartIndices) > 0)
            print("\t\t Using trigger event {}".format(iTrigger))
        except ValueError:
            print("\t\t No Trigger found, using recording from beginning")
            iStartIndex = 0
            tStartTrigger = vtTimeTrace[0]
            iEndIndex = np.searchsorted(vtTimeTrace, vtTimeTrace[0] + duration)
        else:
            tStartTrigger = vtStartTriggers[iTrigger]
            iStartIndex = viStartIndices[iTrigger]
            iEndIndex = viEndIndices[iTrigger]
        # - Filter time trace
        vtTimeTrace = vtTimeTrace[iStartIndex:iEndIndex] - tStartTrigger + self.t
        vnChannels = vnChannels[iStartIndex:iEndIndex]
        print("Layer `{}`: Extracted event data".format(self.name))

        # - Generate TSEvent from recorded data
        tsResponse = TSEvent(
            vtTimeTrace,
            vnChannels,
            t_start=self.t,
            t_stop=self.t + duration,
            num_channels=self.size,
            name="DynapSEDemoBeat",
        )

        # - Set layer time
        self._timestep += num_timesteps

        if verbose:
            print("Layer `{}`: Evolution successful.".format(self.name))

        return tsResponse
