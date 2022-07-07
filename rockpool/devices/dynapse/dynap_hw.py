# ----
# dynap_hw.py - Implementation of HW FF and Rec layers for DynapSE, via ctxCTL
# ----

raise ImportError("This module needs to be ported to the v2 API")

from rockpool.nn.layers.layer import Layer
from rockpool.timeseries import TSEvent
from devices import DynapseControlExtd

import numpy as np
from warnings import warn
from typing import List, Tuple, Optional, Generator
import time


# - Default timestep
DEF_TIMESTEP = 2e-5

# - Absolute tolerance, e.g. for comparing float values
ABS_TOLERANCE = 1e-9

# -- Define the HW layer class for recurrent networks
class RecDynapSE(Layer):
    """
    Recurrent spiking layer implemented with a DynapSE backend.

    This class represents a recurrent layer of spiking neurons, implemented with a HW backend on DynapSE hardware from SynSense.

    """

    def __init__(
        self,
        weights_in: np.ndarray,
        weights_rec: np.ndarray,
        neuron_ids: Optional[np.ndarray] = None,
        virtual_neuron_ids: Optional[np.ndarray] = None,
        dt: Optional[float] = DEF_TIMESTEP,
        max_num_trials_batch: Optional[float] = None,
        max_batch_dur: Optional[float] = None,
        max_num_timesteps: Optional[int] = None,
        max_num_events_batch: Optional[int] = None,
        l_input_core_ids: List[int] = [0],
        input_chip_id: int = 0,
        clearcores_list: Optional[list] = None,
        controller: DynapseControlExtd = None,
        rpyc_port: Optional[int] = None,
        name: Optional[str] = "unnamed",
        skip_weights: bool = False,
        skip_neuron_allocation: bool = False,
        fastmode: bool = False,
        speedup: float = 1.0,
    ):
        """
        Recurrent spiking layer implemented with a DynapSE backend

        :param ndarray[int] weights_in:                     MxN matrix of input weights from virtual to hardware neurons. Supplied in units of synaptic connection.
        :param ndarray[int] weights_rec:                    NxN matrix of weights between hardware neurons. Supplied in units of synaptic connection. Negative elements give rise to inhibitory synapses.
        :param Optional[ndarray[int]] neuron_ids:           1xN array of IDs of hardware neurons that are to be used as layer neurons. Default: None
        :param Optional[ndarray[int]] virtual_neuron_ids:   1xM array of IDs of virtual neurons that are to be used as input neurons. Default: None
        :param Optional[float] dt:                          Layer time-step. Default: 2e-5 s
        :param Optional[int] max_num_trials_batch:          Maximum number of trials (specified in input timeseries) per batch.Longer evolution periods will automatically split in smaller batches.
        :param Optional[float] max_batch_dur:               Maximum duration of single evolution batch.
        :param Optional[int] max_num_timesteps:             Maximum number of time steps of of single evolution batch.
        :param Optional[float] max_num_events_batch:        Maximum number of input events per evolution batch.
        :param Optional[ArrayLike[int]] l_input_core_ids:   IDs of the cores that contain neurons receiving external inputs. To avoid ID collisions neurons on these cores should not receive inputs from other neurons.
        :param Optional[int] input_chip_id:                 ID of the chip with neurons that receive external input.
        :param Optional[List[int]] clearcores_list:         IDs of chips where configurations should be cleared.
        :param Optional[DynapseControl] controller:         DynapseControl object to use to interface with the DynapSE hardware.
        :param Optional[int] rpyc_port:                     Port at which RPyC connection should be established. Only considered if controller is ``None``.
        :param Optional[str] name:                          Layer name.
        :param Optional[bool] skip_weights:                 Do not upload weight configuration to chip. (Use carefully)
        :param Optional[bool] skip_neuron_allocation:       If ``True``, do not verify if neurons are usable. Default: ``False`` (check that specified neurons are usable).
        :param Optional[bool] fastmode:                     If ``True``, ``DynapseControl`` will not load buffered event filters when data is sent. Recording buffer is set to 0. (No effect with `RecDynapSEDemo` class). Default:``False``.
        :param Optional[float] speedup:                     If `fastmode`==True, speed up input events to Dynapse by this factor. (No effect with `RecDynapSEDemo` class). Default: 1.0 (no speedup)
        """

        # - Round weight matrices
        weights_in = np.asarray(np.round(weights_in), int)
        weights_rec = np.asarray(np.round(weights_rec), int)

        # - Check weight matrices
        if weights_in.shape[1] != weights_rec.shape[0]:
            raise ValueError(
                f"RecDynapSE `{name}`: `mnWIn` and `weights_rec` must have compatible "
                + "shapes: `mnWIn` is MxN, `weights_rec` is NxN."
            )
        if weights_rec.shape[0] != weights_rec.shape[1]:
            raise ValueError(
                f"RecDynapSE `{name}`: The recurrent weight matrix `weights_rec` must be square."
            )

        # - Initialise superclass
        super().__init__(
            weights=np.asarray(np.round(weights_in), "int"), dt=dt, name=name
        )
        print("RecDynapSE `{}`: Superclass initialized".format(name))

        # - Store weight matrices
        self.weights_in = weights_in
        self.weights_rec = weights_rec

        # - Store initialization arguments for `to_dict` method
        self._l_input_core_ids = l_input_core_ids
        self._clearcores_list = clearcores_list
        self._rpyc_port = rpyc_port
        self._skip_weights = skip_weights
        self.fastmode = fastmode
        self.speedup = speedup

        # - Record input core mask and chip ID
        self._input_coremask = int(np.sum([2**n_id for n_id in l_input_core_ids]))
        self._input_chip_id = input_chip_id

        # - Instantiate DynapseControl
        if controller is None:
            if dt is None:
                raise ValueError(
                    "Layer `{}` Either dt or controller must be provided".format(name)
                )
            self.controller = DynapseControlExtd(
                fpga_isibase=dt,
                clearcores_list=clearcores_list,
                rpyc_connection=rpyc_port,
            )
        else:
            self.controller = controller
            self.controller.fpga_isibase = dt
            self.controller.clear_connections(clearcores_list)

        # - Allocate layer neurons (and thereby cause initialization of required chips)
        if skip_neuron_allocation:
            neuron_ids = range(1, 1 + self.size) if neuron_ids is None else neuron_ids
            self._hw_neurons = np.array(
                [self.controller.hw_neurons[i] for i in neuron_ids]
            )
            self._shadow_neurons = np.array(
                [self.controller.shadow_neurons[i] for i in neuron_ids]
            )
        else:
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
        if skip_neuron_allocation:
            if virtual_neuron_ids is None:
                virtual_neuron_ids = range(1, 1 + self.size_in)
            self._virtual_neurons = np.array(
                [self.controller.virtual_neurons[i] for i in virtual_neuron_ids]
            )
        else:
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

        # - Store evolution batch size limitations
        self.max_num_trials_batch = max_num_trials_batch
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

        if not skip_weights:
            # - Configure connectivity
            self._compile_weights_and_configure()

        print("Layer `{}` prepared.".format(self.name))

    def _batch_input_data(
        self, ts_input: TSEvent, num_timesteps: int, verbose: bool = False
    ) -> (np.ndarray, int):
        """
        Generator that returns the data in batches

        :param TSEvent ts_input:        Input event time series of data to convert into batches
        :param int num_timesteps:       Number of time steps to return in each batch
        :param Optional[bool] verbose:  If ``True``, display information about the batch. Default: ``False``, do not display information

        :yields :   (vn_tpts_evts_inp_batch, vn_chnls_inp_batch, tstp_start_batch, num_tstps_batch * self.dt)
        """

        # - Time points of input trace in discrete layer time base
        vn_tpts_evts_inp = np.floor(ts_input.times / self.dt).astype(int)

        # - Make sure evolution is within correct interval
        start_idx_all = np.searchsorted(vn_tpts_evts_inp, self._timestep)
        end_idx_all = np.searchsorted(vn_tpts_evts_inp, self._timestep + num_timesteps)
        vn_tpts_evts_inp = vn_tpts_evts_inp[start_idx_all:end_idx_all]
        vn_channels_inp = ts_input.channels[start_idx_all:end_idx_all]
        # vn_channels_inp = ts_input.channels

        # - Check whether data for splitting by trial is available
        if (
            hasattr(ts_input, "trial_start_times")
            and self.max_num_trials_batch is not None
        ):
            ## -- Split by trials
            vn_trial_starts = np.floor(ts_input.trial_start_times / self.dt).astype(int)
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
                # - Number of trials to be used in current batch, considering max. number of trials per batch, events per batch and (if applies) time steps per batch
                num_trials_batch = min(
                    self.max_num_trials_batch, max_num_trials_e, max_num_trials_mnts
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
        Evolve the layer by queueing spikes, stimulating and recording

        :param Optional[TSEvent] ts_input:  input time series, containing `self.size` channels. Default: ``None``, just record for a specified duration
        :param Optional[float] duration:    Desired evolution duration, in seconds. Default: ``None``, use the duration implied by ``ts_input``
        :param Optional[int] num_timesteps: Desired evolution duration, in integer steps of `self.dt`. Default: ``None``, use ``duration`` or ``ts_input` to determine duration
        :param Optional[bool] verbose:      If ``True``, display information on evolution progress. Default: ``True``, display information during evolution

        :return TSEvent:                Output spikes emitted by the neurons in this layer, during the evolution time
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
            ts_input.clip(self.t, self.t + duration),
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
            times_batch, channels_batch = self._send_batch(
                timesteps=vn_tpts_evts_inp_batch - tstp_start_batch,
                channels=vn_chnls_inp_batch,
                dur_batch=dur_batch,
            )

            channels.append(channels_batch)
            times.append(times_batch + tstp_start_batch * self.dt)
            if verbose:
                print("Layer `{}`: Received event data".format(self.name))

        # - Flatten out times and channels
        times = [t for times_curr in times for t in times_curr]
        channels = [ch for channels_curr in channels for ch in channels_curr]

        # - Convert recorded events into TSEvent object
        ts_response = TSEvent(
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

        return ts_response

    def _send_batch(
        self, timesteps: np.ndarray, channels: np.ndarray, dur_batch: float
    ):
        """
        Send a batch of input events to the hardware

        :param ndarray timesteps:   1xT array of event times
        :param ndarray channels:    1xT array of event channels corresponding to event times in ``timesteps``
        :param float dur_batch:     Duration of this batch in seconds

        :return Tuple[times_out, channels_out]: Spike data emitted by the hardware during this batch
        """
        try:
            if self.fastmode:
                times_out, channels_out = self.controller.send_arrays(
                    timesteps=(timesteps.astype(float) / self.speedup).astype(int),
                    channels=channels,
                    t_record=dur_batch / self.speedup,
                    virtual_neur_ids=self.virtual_neuron_ids,
                    record_neur_ids=self.neuron_ids,
                    targetcore_mask=self._input_coremask,
                    targetchip_id=self._input_chip_id,
                    periodic=False,
                    record=True,
                    return_ts=False,
                    t_buffer=0.0,
                    fastmode=True,
                )
            else:
                times_out, channels_out = self.controller.send_arrays(
                    timesteps=timesteps,
                    channels=channels,
                    t_record=dur_batch,
                    virtual_neur_ids=self.virtual_neuron_ids,
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
            num_tstps_batch = int(np.round(dur_batch / self.dt))

            # - Number of time steps and durations of first and second part:
            num_tstps_part1 = num_tstps_batch // 2
            num_tstps_part2 = num_tstps_batch - num_tstps_part1
            tup_durations = (self.dt * num_tstps_part1, self.dt * num_tstps_part2)

            # - Determine where to split arrays of input time steps and channels
            idx_split = np.searchsorted(timesteps, num_tstps_part1)

            # - Event time steps for each part
            tup_v_tstps = (
                timesteps[:idx_split],
                timesteps[idx_split:] - num_tstps_part1,
            )
            # - Event channels for each part$
            tup_v_chnls = (channels[:idx_split], channels[idx_split:])

            # - Evolve the two parts. l_outputs is list with two tuples, each tuple corresponding to one part
            l_outputs: List[Tuple[np.ndarray]] = [
                list(  # - Wrap in list to be able to modify elements (add time) later on
                    self._send_batch(timesteps=tstp, channels=ch, dur_batch=dur)
                )
                for tstp, ch, dur in zip(tup_v_tstps, tup_v_chnls, tup_durations)
            ]
            # - Correct time of second part
            l_outputs[-1][0] += tup_durations[0]

            # - Separate output into arrays
            times_out, channels_out = [
                np.hstack(tup_out) for tup_out in zip(*l_outputs)
            ]

        return times_out, channels_out

    def _compile_weights_and_configure(self):
        """
        Configure DynapSE weights from the weight matrices

        Use the input and recurrent weight matrices to determine an approximate discretised synapse configuration consistent with the weights, and try to configure the DynapSE hardware with the configuration.
        """

        # - Connect virtual neurons to hardware neurons
        self.controller.set_connections_from_weights(
            weights=self.weights_in,
            neuron_ids=self.virtual_neuron_ids,
            neuron_ids_post=self.neuron_ids,
            virtual_pre=True,
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
        input_neurons = (self.weights_in != 0).any(axis=0)

        # - Connections from input neurons to remaining neurons
        weights_in_rec = self.weights.copy()
        weights_in_rec[input_neurons == False] = 0
        self.controller.add_connections_from_weights(
            weights=weights_in_rec,
            neuron_ids=self.neuron_ids,
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
        weights_rec = self.weights.copy()
        weights_rec[input_neurons] = 0
        self.controller.add_connections_from_weights(
            weights=weights_rec,
            neuron_ids=self.neuron_ids,
            syn_exc=self.controller.syn_exc_slow,
            syn_inh=self.controller.syn_inh_fast,
            apply_diff=True,
        )
        print("Layer `{}`: Recurrent connections have been set.".format(self.name))

    def to_dict(self):
        """
        Return a dictionary encapsulating the layer

        :return Dict:   A description of the layer that can be used to recreate the object
        """

        config = {}
        config["name"] = self.name
        config["weights_in"] = self._weights_in.tolist()
        config["weights_rec"] = self._weights_rec.tolist()
        config["neuron_ids"] = self.neuron_ids.tolist()
        config["virtual_neuron_ids"] = self.virtual_neuron_ids.tolist()

        config["dt"] = self.dt

        config["max_num_trials_batch"] = self.max_num_trials_batch
        config["max_batch_dur"] = self.max_batch_dur
        config["max_num_timesteps"] = self.max_num_timesteps

        config["l_input_core_ids"] = self._l_input_core_ids
        config["input_chip_id"] = self._input_chip_id
        config["clearcores_list"] = self._clearcores_list
        config["rpyc_port"] = self._rpyc_port
        config["skip_weights"] = self._skip_weights

        config["class_name"] = self.class_name

        return config

    @property
    def input_type(self):
        """
        (TimeSeries subclass)   The input class of this layer (:py:class:`TSEvent`)
        """
        return TSEvent

    @property
    def output_type(self):
        """
        (TimeSeries subclass)   The ouput class of this layer (:py:class:`TSEvent`)
        """
        return TSEvent

    @property
    def weights_in(self):
        """
        (ndarray[float])   MxN array of input weights from input neurons to recurrent neurons
        """
        return self._weights_in

    @weights_in.setter
    def weights_in(self, new_weights):
        self._weights_in = np.round(
            self._expand_to_shape(
                new_weights, (self.size_in, self.size), "weights_in", allow_none=False
            )
        ).astype(int)

    @property
    def weights_rec(self):
        """
        (ndarray[float])    NxN array of recurrent weights
        :return:
        """
        return self._weights_rec

    @weights_rec.setter
    def weights_rec(self, new_weights):
        self._weights_rec = np.round(
            self._expand_to_shape(
                new_weights, (self.size, self.size), "weights_rec", allow_none=False
            )
        ).astype(int)

    # weights as alias for weights_rec
    @property
    def weights(self):
        """
        (ndarray[float])    NxN array of recurrent weights. Alias for ``.weights_rec``.
        :return:
        """
        return self.weights_rec

    @weights.setter
    def weights(self, new_weights):
        self.weights_rec = new_weights

    # _weights as alias for _weights_rec
    @property
    def _weights(self):
        """
        (ndarray[float])    NxN array of recurrent weights. Alias for ``._weights_rec``.
        """
        return self._weights_rec

    @_weights.setter
    def _weights(self, new_weights):
        self._weights_rec = new_weights

    @property
    def max_num_trials_batch(self):
        """
        (int)   Maximum number of trials in a batch. ``None`` or ``int`` > 0.
        """
        return self._max_num_trials_batch

    @max_num_trials_batch.setter
    def max_num_trials_batch(self, new_max):
        assert new_max is None or (
            type(new_max) == int and 0 < new_max
        ), "Layer `{}`: max_num_trials_batch must be an integer greater than 0 or None.".format(
            self.name
        )
        self._max_num_trials_batch = new_max

    @property
    def max_num_timesteps(self):
        """
        (int)   Maxmimum number of timesteps. ``None`` or ``int`` > 0.
        """
        return self._max_num_timesteps

    @max_num_timesteps.setter
    def max_num_timesteps(self, new_max):
        assert new_max is None or (
            type(new_max) == int and 0 < new_max
        ), "Layer `{}`: max_num_timesteps must be an integer greater than 0 or None.".format(
            self.name
        )
        if new_max > self.controller.fpga_event_limit * self.controller.fpga_isi_limit:
            warn(
                "Layer `{}`: max_num_timesteps is larger than fpga_event_limit * fpga_isi_limit ({}).".format(
                    self.name,
                    self.controller.fpga_event_limit * self.controller.fpga_isi_limit,
                )
            )
        self._max_num_timesteps = new_max

    @property
    def max_batch_dur(self):
        """
        (int)   Maximum duration of a batch, in integer timesteps. ``None`` or ``int`` > 0
        """
        return (
            None
            if self._max_num_timesteps is None
            else self._max_num_timesteps * self.dt
        )

    @max_batch_dur.setter
    def max_batch_dur(self, new_max_t):
        assert new_max_t is None or (
            type(new_max_t) == int and 0 < new_max_t
        ), "Layer `{}`: max_batch_dur must be an integer greater than 0 or None.".format(
            self.name
        )
        self._max_num_timesteps = (
            None if new_max_t is None else int(np.round(new_max_t / self.dt))
        )

    @property
    def max_num_events_batch(self):
        """
        (int)   Maximum number of events in a batch. ``int`` >= 0
        """
        return self._max_num_events_batch

    @max_num_events_batch.setter
    def max_num_events_batch(self, new_max):
        assert (
            type(new_max) == int and 0 < new_max <= self.controller.fpga_event_limit
        ), "Layer `{}`: max_num_events_batch must be an integer between 0 and {}.".format(
            self.name, self.controller.fpga_event_limit
        )
        self._max_num_events_batch = new_max

    @property
    def virtual_neuron_ids(self):
        """
        (ndarray[int])  1xM Array of virtual neuron IDs implementing the input layer
        :return:
        """
        return self._virtual_neuron_ids

    @property
    def neuron_ids(self):
        """
        (ndarray[int])  1xN array of neuron IDs implementing the recurrent layer
        :return:
        """
        return self._neuron_ids

    @property
    def l_input_core_ids(self):
        """
        Core mask as a reversed binary string
        :return:
        """
        # - Core mask as reversed binary string
        bin_coremask = reversed(bin(self._input_coremask)[-4:])
        return [core_id for core_id, mask in enumerate(bin_coremask) if int(mask)]


# -- Define subclass of RecDynapSE to be used in demos with preloaded events
class RecDynapSEDemo(RecDynapSE):
    """
    Recurrent layer implemented on DynapSE, using preloaded events during evolution

    This layer plays back a set of provided events during evolution. Events are loaded on to the FPGA in the DynapSE box, then played back during evolution.

    .. rubric:: Usage
    Create the layer, then use the :py:`.load_events` method to provide a time series of events to load.
    """

    def __init__(self, *args, **kwargs):
        """
        Recurrent spiking layer, which plays back preloaded events during evolution

        :param ndarray[int] weights_in:                     MxN matrix of input weights from virtual to hardware neurons. Supplied in units of synaptic connection.
        :param ndarray[int] weights_rec:                    NxN matrix of weights between hardware neurons. Supplied in units of synaptic connection. Negative elements give rise to inhibitory synapses.
        :param Optional[ndarray[int]] neuron_ids:           1xN array of IDs of hardware neurons that are to be used as layer neurons. Default: None
        :param Optional[ndarray[int]] virtual_neuron_ids:   1xM array of IDs of virtual neurons that are to be used as input neurons. Default: None
        :param Optional[float] dt:                          Layer time-step. Default: 2e-5 s
        :param Optional[int] max_num_trials_batch:          Maximum number of trials (specified in input timeseries) per batch.Longer evolution periods will automatically split in smaller batches.
        :param Optional[float] max_batch_dur:               Maximum duration of single evolution batch.
        :param Optional[int] max_num_timesteps:             Maximum number of time steps of of single evolution batch.
        :param Optional[float] max_num_events_batch:        Maximum number of input events per evolution batch.
        :param Optional[ArrayLike[int]] l_input_core_ids:   IDs of the cores that contain neurons receiving external inputs. To avoid ID collisions neurons on these cores should not receive inputs from other neurons.
        :param Optional[int] input_chip_id:                 ID of the chip with neurons that receive external input.
        :param Optional[List[int]] clearcores_list:         IDs of chips where configurations should be cleared.
        :param Optional[DynapseControl] controller:         DynapseControl object to use to interface with the DynapSE hardware.
        :param Optional[int] rpyc_port:                     Port at which RPyC connection should be established. Only considered if controller is ``None``.
        :param Optional[str] name:                          Layer name.
        :param Optional[bool] skip_weights:                 Do not upload weight configuration to chip. (Use carefully)
        :param Optional[bool] skip_neuron_allocation:       If ``True``, do not verify if neurons are usable. Default: ``False`` (check that specified neurons are usable).
        :param Optional[bool] fastmode:                     If ``True``, ``DynapseControl`` will not load buffered event filters when data is sent. Recording buffer is set to 0. (No effect with `RecDynapSEDemo` class). Default:``False``.
        :param Optional[float] speedup:                     If `fastmode`==True, speed up input events to Dynapse by this factor. (No effect with `RecDynapSEDemo` class). Default: 1.0 (no speedup)
        """

        # - Call parent initializer
        super().__init__(*args, **kwargs)

        # - Set up filter for recording spikes
        self.controller.add_buffered_event_filter(self.neuron_ids)

    def load_events(self, ts_as: TSEvent, vt_rhythm_start, dur_total: float):
        """
        Load a series of events for later playback

        :param TSEvent ts_as:               Set of events to load to FPGA for later playback
        :param List[float] vt_rhythm_start: List of times that indicate the beginning of a playback trial
        :param float dur_total:             Total duration of event sequence, in seconds
        """
        if ts_as.times.size > self.controller.sram_event_limit:
            raise MemoryError(
                "Layer `{}`: Can upload at most {} events. {} are too many.".format(
                    self.name, self.controller.sram_event_limit, ts_as.times.size
                )
            )

        # - Indices corresponding to first event of each rhythm
        v_rhythm_start_idcs = np.searchsorted(ts_as.times, vt_rhythm_start)

        # - Durations of each rhythm
        self.v_rhythm_durs = np.diff(np.r_[vt_rhythm_start, dur_total])

        # - Convert timeseries to events for FPGA
        events: List = self.controller._TSEvent_to_spike_list(
            series=ts_as,
            neuron_ids=self.virtual_neuron_ids,
            targetcore_mask=1,
            targetchip_id=0,
        )
        # - Set interspike interval of first event of each rhythm so that it corresponds to the rhythm start and not the last event from the previous rhythm
        for idx_rhythm, idx_evt in enumerate(v_rhythm_start_idcs):
            events[idx_evt].isi = np.round(
                (ts_as.times[idx_evt] - vt_rhythm_start[idx_rhythm])
                / self.controller.fpga_isibase
            ).astype(int)

        print(
            "Layer `{}`: {} events have been generated.".format(self.name, len(events))
        )

        # - Upload input events to processor
        idx_evt = 0
        while idx_evt < ts_as.times.size:
            self.controller.fpga_spikegen.set_base_addr(2 * idx_evt)
            self.controller.fpga_spikegen.preload_stimulus(
                events[
                    idx_evt : min(
                        idx_evt + self.controller.fpga_event_limit, len(events)
                    )
                ]
            )
            idx_evt += self.controller.fpga_event_limit
        print("Layer `{}`: Events have been loaded.".format(self.name))

        # - Fpga adresses where beats start
        self.v_rhythm_addrs = 2 * (v_rhythm_start_idcs)

        # - Number of events per rhythm: do -1 for each rhythm, due to ctxctl syntax
        self.v_num_evts_rhythm = np.diff(np.r_[v_rhythm_start_idcs, len(events)]) - 1

    # @profile
    def evolve(
        self, idx_rhythm: int, duration: Optional[float] = None, verbose: bool = True
    ) -> TSEvent:
        """
        Evolve the layer by playing back from the given base address and recording

        :param int idx_rhythm:  Index of the rhythm to be played back
        :param float duration:  Desired evolution duration, in seconds, use rhythm duration if None
        :param int nNumEvents:  Number of input events to play back on FPGA starting from base address

        :return TSEvent:        Spikes emitted by the neurons in this layer, during the evolution time
        """

        # - Determine evolution duration
        duration = self.v_rhythm_durs[idx_rhythm] if duration is None else duration
        num_timesteps = int(np.floor((duration + ABS_TOLERANCE) / self.dt))

        # - Instruct FPGA to spike
        # set new base adress and number of input events for stimulation
        self.controller.fpga_spikegen.set_base_addr(self.v_rhythm_addrs[idx_rhythm])
        self.controller.fpga_spikegen.set_stim_count(self.v_num_evts_rhythm[idx_rhythm])

        # - Clear event filter
        self.controller.bufferedfilter.get_events()
        self.controller.bufferedfilter.get_special_event_timestamps()

        # - Time at which stimulation stops
        t_stop = time.time() + duration

        # Start stimulation
        self.controller.fpga_spikegen.start()

        # - Process recorded data
        times, channels = self.controller._record_and_process(
            t_stop=t_stop,
            record_neur_ids=self.neuron_ids,
            duration=duration,
            fastmode=True,
        )

        # - Generate TSEvent from recorded data
        ts_response = TSEvent(
            times + self.t,
            channels,
            t_start=self.t,
            t_stop=self.t + duration,
            num_channels=self.size,
            name="DynapSEDemoBeat",
        )

        # - Set layer time
        self._timestep += num_timesteps

        if verbose:
            print("Layer `{}`: Evolution successful.".format(self.name))

        return ts_response
