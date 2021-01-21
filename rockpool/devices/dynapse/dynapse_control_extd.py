# ----
# dynpase_control_extd.py - Subclass `DynapseControlExtd` of `DynapseControl` that
#                           provides handling of and functionality for `TSEvent` objects.
# ----

raise ImportError("This module needs to be ported to the v2 API")

from warnings import warn
from typing import List, Union, Optional, Tuple

import numpy as np

from rockpool.devices.dynapse.dynapse_control import DynapseControl
from rockpool.timeseries import TSEvent

__all__ = ["DynapseControlExtd"]


class DynapseControlExtd(DynapseControl):
    def _TSEvent_to_spike_list(
        self,
        series: TSEvent,
        neuron_ids: np.ndarray,
        targetcore_mask: int = 1,
        targetchip_id: int = 0,
    ) -> List:
        """
        _TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

        :param series:        TSEvent      Time series of events to send as input
        :param neuron_ids:     ArrayLike    IDs of neurons that should appear as sources of the events
        :param targetcore_mask: int          Mask defining target cores (sum of 2**core_id)
        :param targetchip_id:   int          ID of target chip
        :return:                list of FpgaSpikeEvent objects
        """
        # - Check that the number of channels is the same between time series and list of neurons
        assert series.num_channels <= np.size(
            neuron_ids
        ), "`series` contains more channels than the number of neurons in `neuron_ids`."

        # - Make sure neuron_ids is iterable
        neuron_ids = np.array(neuron_ids)

        # - Get events from this time series
        times, channels = series()

        # - Convert to ISIs
        t_start = series.t_start
        times = np.r_[t_start, times]
        times_discrete = (np.round(times / self.fpga_isibase)).astype("int")
        isi_array_discrete = np.diff(times_discrete)

        # - Convert events to an FpgaSpikeEvent
        print("DynapseControlExtd: Generating FPGA event list from TSEvent.")
        events: List = self.tools.generate_fpga_event_list(
            # Make sure that no np.int64 or other non-native type is passed
            [int(isi) for isi in isi_array_discrete],
            [int(neuron_ids[i]) for i in channels],
            int(targetcore_mask),
            int(targetchip_id),
        )
        # - Return a list of events
        return events

    def send_pulse(
        self,
        width: float = 0.1,
        frequency: float = 1000,
        t_record: float = 3,
        t_buffer: float = 0.5,
        virtual_neur_id: int = 0,
        record_neur_ids: Union[int, np.ndarray] = np.arange(1024),
        targetcore_mask: int = 15,
        targetchip_id: int = 0,
        periodic: bool = False,
        record: bool = False,
        return_ts: bool = False,
        inputneur_id=None,
    ) -> Union[None, Tuple[np.ndarray, np.ndarray], TSEvent]:
        """
        send_pulse - Send a pulse of periodic input events to the chip.
                     Return a TSEvent wih the recorded hardware activity.
        :param width:            float  Duration of the input pulse
        :param frequency:               float  Frequency of the events that constitute the pulse
        :param t_record:         float  Duration of the recording (including stimulus)
        :param t_buffer:         float  Record slightly longer than t_record to
                                        make sure to catch all relevant events
        :param virtual_neur_id:     int    ID of input neuron
        :param record_neur_ids:  array-like  ID(s) of neuron(s) to be recorded
        :param nChipID:          int  Target chip ID
        :param nCoreMask:        int  Target core mask
        :param periodic:         bool  Repeat the stimulus indefinitely
        :param record:           bool  Set up buffered event filter that records events
                                       from neurons defined in record_neur_ids
        :param return_ts:        bool    If True and record==True: output TSEvent instead of arrays of times and channels

        :return:
            if record==False:  None
            elif return_ts:    TSEvent object of recorded data
            else:              (times_out, channels_out)  np.ndarrays that contain recorded data
        """

        if inputneur_id is not None:
            warn(
                "DynapseControlExtd: The argument `inputneur_id` has been "
                + "renamed to 'virtual_neur_id`. The old name will not be "
                + "supported anymore in future versions."
            )
            if virtual_neur_id is None:
                virtual_neur_id = inputneur_id

        # - Stimulate and obtain recorded data if any
        recorded_data = super().send_pulse(
            width=width,
            frequency=frequency,
            t_record=t_record,
            t_buffer=t_buffer,
            virtual_neur_id=virtual_neur_id,
            record_neur_ids=record_neur_ids,
            targetcore_mask=targetcore_mask,
            targetchip_id=targetchip_id,
            periodic=periodic,
            record=record,
        )

        if recorded_data is not None:
            times_out, channels_out = recorded_data

            if return_ts:
                return TSEvent(
                    times_out,
                    channels_out,
                    t_start=0,
                    t_stop=t_record,
                    num_channels=(
                        np.amax(channels_out) + 1
                        if record_neur_ids is None
                        else np.size(record_neur_ids)
                    ),
                    name="DynapSE",
                )
            else:
                return times_out, channels_out

    def send_TSEvent(
        self,
        series,
        t_record: Optional[float] = None,
        t_buffer: float = 0.5,
        virtual_neur_ids: Optional[np.ndarray] = None,
        record_neur_ids: Optional[np.ndarray] = None,
        targetcore_mask: int = 15,
        targetchip_id: int = 0,
        periodic=False,
        record=False,
        return_ts=False,
        neuron_ids=None,
    ) -> Union[None, Tuple[np.ndarray, np.ndarray], TSEvent]:
        """
        send_TSEvent - Extract events from a TSEvent object and send them to FPGA.

        :param series:        TSEvent      Time series of events to send as input
        :param t_record:         float  Duration of the recording (including stimulus)
                                       If None, use series.duration
        :param t_buffer:         float  Record slightly longer than t_record to
                                       make sure to catch all relevant events
        :param virtual_neur_ids:     ArrayLike    IDs of neurons that should appear as sources of the events
                                             If None, use channels from series
        :param record_neur_ids: ArrayLike    IDs of neurons that should be recorded (if record==True)
                                               If None and record==True, record neurons in virtual_neur_ids
        :param targetcore_mask: int          Mask defining target cores (sum of 2**core_id)
        :param targetchip_id:   int          ID of target chip
        :param periodic:       bool         Repeat the stimulus indefinitely
        :param record:         bool         Set up buffered event filter that records events
                                             from neurons defined in record_neur_ids
        :param return_ts:        bool         If record: output TSEvent instead of arrays of times and channels

        :return:
            if record==False:  None
            elif return_ts:      TSEvent object of recorded data
            else:               (times_out, channels_out)  np.ndarrays that contain recorded data
        """

        if neuron_ids is not None:
            warn(
                "DynapseControlExtd: The argument `neuron_ids` has been "
                + "renamed to 'virtual_neur_ids`. The old name will not be "
                + "supported anymore in future versions."
            )
            if virtual_neur_ids is None:
                virtual_neur_ids = neuron_ids

        # - Process input arguments
        virtual_neur_ids = (
            np.arange(series.num_channels)
            if virtual_neur_ids is None
            else np.array(virtual_neur_ids)
        )
        record_neur_ids = (
            virtual_neur_ids if record_neur_ids is None else record_neur_ids
        )
        t_record = series.duration if t_record is None else t_record

        # - Prepare event list
        events = self._TSEvent_to_spike_list(
            series,
            neuron_ids=virtual_neur_ids,
            targetcore_mask=targetcore_mask,
            targetchip_id=targetchip_id,
        )
        print(
            "DynapseControlExtd: Stimulus prepared from TSEvent `{}`.".format(
                series.name
            )
        )

        # - Stimulate and obtain recorded data if any
        recorded_data = self._send_stimulus_list(
            events=events,
            duration=t_record,
            t_buffer=t_buffer,
            record_neur_ids=record_neur_ids,
            periodic=periodic,
            record=record,
        )

        if recorded_data is not None:
            times_out, channels_out = recorded_data

            if return_ts:
                return TSEvent(
                    times_out,
                    channels_out,
                    t_start=0,
                    t_stop=t_record,
                    num_channels=(
                        np.amax(channels_out) + 1
                        if record_neur_ids is None
                        else np.size(record_neur_ids)
                    ),
                    name="DynapSE",
                )
            else:
                return times_out, channels_out

    def send_arrays(
        self,
        channels: np.ndarray,
        timesteps: Optional[np.ndarray] = None,
        times: Optional[np.ndarray] = None,
        t_record: Optional[float] = None,
        t_buffer: float = 0.5,
        virtual_neur_ids: Optional[np.ndarray] = None,
        record_neur_ids: Optional[np.ndarray] = None,
        targetcore_mask: int = 15,
        targetchip_id: int = 0,
        periodic=False,
        record=False,
        return_ts=False,
        fastmode: bool = False,
        neuron_ids=None,
    ) -> Union[None, Tuple[np.ndarray, np.ndarray], TSEvent]:
        """
        send_arrays - Send events defined in timetrace and channel arrays to FPGA.

        :param channels:      np.ndarray  Event channels
        :param vnTimeSeops:     np.ndarray  Event times in Fpga time base (overwrites times if not None)
        :param times:     np.ndarray  Event times in seconds
        :param t_record:         float  Duration of the recording (including stimulus)
                                       If None, use times[-1]
        :param t_buffer:         float  Record slightly longer than t_record to
                                       make sure to catch all relevant events
        :param virtual_neur_ids:     ArrayLike    IDs of neurons that should appear as sources of the events
                                             If None, use channels from channels
        :param record_neur_ids: ArrayLike    IDs of neurons that should be recorded (if record==True)
                                               If None and record==True, record neurons in virtual_neur_ids
        :param targetcore_mask: int          Mask defining target cores (sum of 2**core_id)
        :param targetchip_id:   int          ID of target chip
        :param periodic:       bool         Repeat the stimulus indefinitely
        :param record:         bool         Set up buffered event filter that records events
                                             from neurons defined in record_neur_ids
        :param return_ts:        bool         If record: output TSEvent instead of arrays of times and channels

        :return:
            if record==False:  None
            elif return_ts:      TSEvent object of recorded data
            else:               (times_out, channels_out)  np.ndarrays that contain recorded data
        """

        if neuron_ids is not None:
            warn(
                "DynapseControlExtd: The argument `neuron_ids` has been "
                + "renamed to 'virtual_neur_ids`. The old name will not be "
                + "supported anymore in future versions."
            )
            if virtual_neur_ids is None:
                virtual_neur_ids = neuron_ids

        # - Stimulate and obtain recorded data if any
        recorded_data = super().send_arrays(
            channels=channels,
            timesteps=timesteps,
            times=times,
            t_record=t_record,
            t_buffer=t_buffer,
            virtual_neur_ids=virtual_neur_ids,
            record_neur_ids=record_neur_ids,
            targetcore_mask=targetcore_mask,
            targetchip_id=targetchip_id,
            periodic=periodic,
            record=record,
            fastmode=fastmode,
        )

        if recorded_data is not None:
            times_out, channels_out = recorded_data

            if return_ts:
                return TSEvent(
                    times_out,
                    channels_out,
                    t_start=0,
                    t_stop=t_record,
                    num_channels=(
                        np.amax(channels_out) + 1
                        if record_neur_ids is None
                        else np.size(record_neur_ids)
                    ),
                    name="DynapSE",
                )
            else:
                return times_out, channels_out

    def record(
        self,
        neuron_ids: Union[np.ndarray, List[int], int],
        duration: Optional[float] = None,
        return_ts: bool = False,
    ) -> Union[Tuple[np.array, np.array], TSEvent]:
        """
        record - Record spiking activity of given neurons. Either record for
                 given duration or until `self.stop_recording` is called
        :param neuron_ids:  Array-like with IDs of neurons that should be recorded
        :param duration:    Recording duration in seconds. If None, will record
                            until `self.stop_recording` is called.
        :param return_ts:      If True, return TSEvent instead of arrays.
        """
        times, channels = super().record(neuron_ids, duration)
        if return_ts:
            return TSEvent(
                times,
                channels,
                t_start=0,
                t_stop=duration,
                num_channels=len(neuron_ids),
                name="DynapSE",
            )
        else:
            return times, channels

    def stop_recording(
        self, since_trigger: bool = False, return_ts: bool = False
    ) -> Union[TSEvent, Tuple[np.ndarray, np.ndarray]]:
        """
        stop_recording - Stop recording and return recorded events as arrays.
        :param since_trigger:  If True, only use events recorded after first
                               trigger event in buffer.
        :param return_ts:      If True, return TSEvent instead of arrays.
        """
        try:
            self.bufferedfilter.clear()
        except AttributeError:
            warn("DynapseControlExtd: No recording has been started.")
            return np.array([]), np.array([])
        else:
            if return_ts:
                return self._recorded_data_to_TSEvent(None, None, since_trigger)
            else:
                return self._recorded_data_to_arrays(None, None, since_trigger)

    def _recorded_data_to_TSEvent(
        self,
        neuron_ids: Union[np.ndarray, None],
        t_record: float,
        since_trigger: bool = True,
    ) -> TSEvent:
        # - Retrieve recorded data and convert to arrays
        times_out, channels_out = super()._recorded_data_to_arrays(
            neuron_ids=neuron_ids, t_record=t_record, since_trigger=since_trigger
        )

        return TSEvent(
            times_out,
            channels_out,
            t_start=0,
            t_stop=t_record,
            num_channels=(
                np.amax(channels_out) + 1 if neuron_ids is None else np.size(neuron_ids)
            ),
            name="DynapSE",
        )
