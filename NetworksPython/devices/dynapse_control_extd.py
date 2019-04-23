# ----
# dynpase_control_extd.py - Subclass `DynapseControlExtd` of `DynapseControl` that
#                           provides handling of and functionality for `TSEvent` objects.
# ----

from typing import List, Union, Optional, Tuple

import numpy as np

from .dynapse_control import DynapseControl, generate_fpga_event_list
from ..timeseries import TSEvent

__all__ = ["DynapseControlExtd"]


class DynapseControlExtd(DynapseControl):
    def _TSEvent_to_spike_list(
        self,
        tsSeries: TSEvent,
        vnNeuronIDs: np.ndarray,
        nTargetCoreMask: int = 1,
        nTargetChipID: int = 0,
    ) -> List:
        """
        _TSEvent_to_spike_list - Convert a TSEvent object to a ctxctl spike list

        :param tsSeries:        TSEvent      Time series of events to send as input
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :return:                list of FpgaSpikeEvent objects
        """
        # - Check that the number of channels is the same between time series and list of neurons
        assert tsSeries.num_channels <= np.size(
            vnNeuronIDs
        ), "`tsSeries` contains more channels than the number of neurons in `vnNeuronIDs`."

        # - Make sure vnNeuronIDs is iterable
        vnNeuronIDs = np.array(vnNeuronIDs)

        # - Get events from this time series
        vtTimes, vnChannels = tsSeries()

        # - Convert to ISIs
        tStartTime = tsSeries.t_start
        vtISIs = np.diff(np.r_[tStartTime, vtTimes])
        vnDiscreteISIs = (np.round(vtISIs / self.tFpgaIsiBase)).astype("int")

        # - Convert events to an FpgaSpikeEvent
        print("dynapse_control: Generating FPGA event list from TSEvent.")
        lEvents = generate_fpga_event_list(
            # Make sure that no np.int64 or other non-native type is passed
            [int(nISI) for nISI in vnDiscreteISIs],
            [int(vnNeuronIDs[i]) for i in vnChannels],
            int(nTargetCoreMask),
            int(nTargetChipID),
        )
        # - Return a list of events
        return lEvents

    def send_pulse(
        self,
        tWidth: float = 0.1,
        fFreq: float = 1000,
        tRecord: float = 3,
        tBuffer: float = 0.5,
        nInputNeuronID: int = 0,
        vnRecordNeuronIDs: Union[int, np.ndarray] = np.arange(1024),
        nTargetCoreMask: int = 15,
        nTargetChipID: int = 0,
        bPeriodic: bool = False,
        bRecord: bool = False,
        bTSEvent: bool = False,
    ) -> Union[None, Tuple[np.ndarray, np.ndarray], TSEvent]:
        """
        send_pulse - Send a pulse of periodic input events to the chip.
                     Return a TSEvent wih the recorded hardware activity.
        :param tWidth:              float  Duration of the input pulse
        :param fFreq:               float  Frequency of the events that constitute the pulse
        :param tRecord:             float  Duration of the recording (including stimulus)
        :param tBuffer:             float  Record slightly longer than tRecord to
                                           make sure to catch all relevant events
        :param nInputNeuronID:      int    ID of input neuron
        :param vnRecordNeuronIDs:   array-like  ID(s) of neuron(s) to be recorded
        :param nChipID:     int  Target chip ID
        :param nCoreMask:   int  Target core mask
        :param bPeriodic:   bool    Repeat the stimulus indefinitely
        :param bRecord:     bool    Set up buffered event filter that records events
                                    from neurons defined in vnRecordNeuronIDs
        :param bTSEvent:    bool    If True and bRecord==True: output TSEvent instead of arrays of times and channels

        :return:
            if bRecord==False:  None
            elif bTSEvent:      TSEvent object of recorded data
            else:               (vtTimeTrace, vnChannels)  np.ndarrays that contain recorded data
        """
        # - Stimulate and obtain recorded data if any
        recorded_data = super().send_pulse(
            tWidth=tWidth,
            fFreq=fFreq,
            tRecord=tRecord,
            tBuffer=tBuffer,
            nInputNeuronID=nInputNeuronID,
            vnRecordNeuronIDs=vnRecordNeuronIDs,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,
            bPeriodic=bPeriodic,
            bRecord=bRecord,
        )

        if recorded_data is not None:
            times, channels = recorded_data

            if bTSEvent:
                return TSEvent(
                    times,
                    channels,
                    t_start=0,
                    t_stop=tRecord,
                    num_channels=(
                        np.amax(channels)
                        if vnRecordNeuronIDs is None
                        else np.size(vnRecordNeuronIDs)
                    ),
                    name="DynapSE",
                )
            else:
                return times, channels

    def send_TSEvent(
        self,
        tsSeries,
        tRecord: Optional[float] = None,
        tBuffer: float = 0.5,
        vnNeuronIDs: Optional[np.ndarray] = None,
        vnRecordNeuronIDs: Optional[np.ndarray] = None,
        nTargetCoreMask: int = 15,
        nTargetChipID: int = 0,
        bPeriodic=False,
        bRecord=False,
        bTSEvent=False,
    ) -> Union[None, Tuple[np.ndarray, np.ndarray], TSEvent]:
        """
        send_TSEvent - Extract events from a TSEvent object and send them to FPGA.

        :param tsSeries:        TSEvent      Time series of events to send as input
        :param tRecord:         float  Duration of the recording (including stimulus)
                                       If None, use tsSeries.duration
        :param tBuffer:         float  Record slightly longer than tRecord to
                                       make sure to catch all relevant events
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
                                             If None, use channels from tsSeries
        :param vnRecordNeuronIDs: ArrayLike    IDs of neurons that should be recorded (if bRecord==True)
                                               If None and bRecord==True, record neurons in vnNeuronIDs
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :param bPeriodic:       bool         Repeat the stimulus indefinitely
        :param bRecord:         bool         Set up buffered event filter that records events
                                             from neurons defined in vnNeuronIDs
        :param bTSEvent:        bool         If bRecord: output TSEvent instead of arrays of times and channels

        :return:
            if bRecord==False:  None
            elif bTSEvent:      TSEvent object of recorded data
            else:               (vtTimeTrace, vnChannels)  np.ndarrays that contain recorded data
        """

        # - Process input arguments
        vnNeuronIDs = (
            np.arange(tsSeries.num_channels)
            if vnNeuronIDs is None
            else np.array(vnNeuronIDs)
        )
        vnRecordNeuronIDs = (
            vnNeuronIDs if vnRecordNeuronIDs is None else vnRecordNeuronIDs
        )
        tRecord = tsSeries.duration if tRecord is None else tRecord

        # - Prepare event list
        lEvents = self._TSEvent_to_spike_list(
            tsSeries,
            vnNeuronIDs=vnNeuronIDs,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,
        )
        print(
            "DynapseControl: Stimulus prepared from TSEvent `{}`.".format(
                tsSeries.strName
            )
        )

        # - Stimulate and obtain recorded data if any
        recorded_data = self._send_stimulus_list(
            lEvents=lEvents,
            tDuration=tRecord,
            tBuffer=tBuffer,
            vnRecordNeuronIDs=vnRecordNeuronIDs,
            bPeriodic=bPeriodic,
            bRecord=bRecord,
        )

        if recorded_data is not None:
            times, channels = recorded_data

            if bTSEvent:
                return TSEvent(
                    times,
                    channels,
                    t_start=0,
                    t_stop=tRecord,
                    num_channels=(
                        np.amax(channels)
                        if vnRecordNeuronIDs is None
                        else np.size(vnRecordNeuronIDs)
                    ),
                    name="DynapSE",
                )
            else:
                return times, channels

    def send_arrays(
        self,
        vnChannels: np.ndarray,
        vnTimeSteps: Optional[np.ndarray] = None,
        vtTimeTrace: Optional[np.ndarray] = None,
        tRecord: Optional[float] = None,
        tBuffer: float = 0.5,
        vnNeuronIDs: Optional[np.ndarray] = None,
        vnRecordNeuronIDs: Optional[np.ndarray] = None,
        nTargetCoreMask: int = 15,
        nTargetChipID: int = 0,
        bPeriodic=False,
        bRecord=False,
        bTSEvent=False,
    ) -> Union[None, Tuple[np.ndarray, np.ndarray], TSEvent]:
        """
        send_arrays - Send events defined in timetrace and channel arrays to FPGA.

        :param vnChannels:      np.ndarray  Event channels
        :param vnTimeSeops:     np.ndarray  Event times in Fpga time base (overwrites vtTimeTrace if not None)
        :param vtTimeTrace:     np.ndarray  Event times in seconds
        :param tRecord:         float  Duration of the recording (including stimulus)
                                       If None, use vtTimeTrace[-1]
        :param tBuffer:         float  Record slightly longer than tRecord to
                                       make sure to catch all relevant events
        :param vnNeuronIDs:     ArrayLike    IDs of neurons that should appear as sources of the events
                                             If None, use channels from vnChannels
        :param vnRecordNeuronIDs: ArrayLike    IDs of neurons that should be recorded (if bRecord==True)
                                               If None and bRecord==True, record neurons in vnNeuronIDs
        :param nTargetCoreMask: int          Mask defining target cores (sum of 2**core_id)
        :param nTargetChipID:   int          ID of target chip
        :param bPeriodic:       bool         Repeat the stimulus indefinitely
        :param bRecord:         bool         Set up buffered event filter that records events
                                             from neurons defined in vnNeuronIDs
        :param bTSEvent:        bool         If bRecord: output TSEvent instead of arrays of times and channels

        :return:
            if bRecord==False:  None
            elif bTSEvent:      TSEvent object of recorded data
            else:               (vtTimeTrace, vnChannels)  np.ndarrays that contain recorded data
        """

        # - Stimulate and obtain recorded data if any
        recorded_data = super().send_arrays(
            vnChannels=vnChannels,
            vnTimeSteps=vnTimeSteps,
            vtTimeTrace=vtTimeTrace,
            tRecord=tRecord,
            tBuffer=tBuffer,
            vnNeuronIDs=vnNeuronIDs,
            vnRecordNeuronIDs=vnRecordNeuronIDs,
            nTargetCoreMask=nTargetCoreMask,
            nTargetChipID=nTargetChipID,
            bPeriodic=bPeriodic,
            bRecord=bRecord,
        )

        if recorded_data is not None:
            times, channels = recorded_data

            if bTSEvent:
                return TSEvent(
                    times,
                    channels,
                    t_start=0,
                    t_stop=tRecord,
                    num_channels=(
                        np.amax(channels)
                        if vnRecordNeuronIDs is None
                        else np.size(vnRecordNeuronIDs)
                    ),
                    name="DynapSE",
                )
            else:
                return times, channels

    def _recorded_data_to_TSEvent(
        self, vnNeuronIDs: np.ndarray, tRecord: float
    ) -> TSEvent:
        # - Retrieve recorded data and convert to arrays
        vtTimeTrace, vnChannels = super()._recorded_data_to_arrays(
            vnNeuronIDs=vnNeuronIDs, tRecord=tRecord
        )

        return TSEvent(
            vtTimeTrace,
            vnChannels,
            t_start=0,
            t_stop=tRecord,
            num_channels=(
                np.amax(vnChannels) if vnNeuronIDs is None else np.size(vnNeuronIDs)
            ),
            name="DynapSE",
        )
