"""
Dynap-SE samna backend bridge
Handles the low-level hardware configuration under the hood and provide easy-to-use access to the user

Note : Existing modules are reconstructed considering consistency with Xylo support.

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com

15/09/2022
[] TODO : configure FPGA inside ?
[] TODO : It's done when it's done
"""

from __future__ import annotations

from typing import Any, Dict, Optional, List, Tuple
import time

import numpy as np
from rockpool.devices.dynapse.interface.utils import (
    aer_to_raster,
    event_generator,
    raster_to_aer,
    capture_events_from_device,
)


# - Rockpool imports
from rockpool.nn.modules.module import Module
from rockpool.devices.dynapse.samna_alias.dynapse2 import (
    Dynapse2Destination,
    NormalGridEvent,
    Dynapse2Interface,
    Dynapse2Model,
    DeviceInfo,
)

# Try to import samna for device interfacing
try:
    import samna
    import samna.dynapse2 as se2
except:
    samna = Any
    se2 = Any
    print(
        "Device interface requires `samna` package which is not installed on the system"
    )

# - Configure exports
__all__ = ["DynapseSamna"]
DT_FPGA = 1e-6


class DynapseSamna(Module):
    def __init__(
        self,
        shape: Tuple[int],
        board: Dynapse2Interface,
        dt: float = 1e-3,
    ):

        if board is None:
            raise ValueError("`device` must be a valid, opened Xylo HDK device.")

        # - Initialise the superclass
        super().__init__(shape=shape, spiking_input=True, spiking_output=True)

        self.board = board
        self.dt = dt
        self.dt_fpga = DT_FPGA

    def evolve(
        self,
        input_data: np.ndarray,
        channel_map: Optional[Dict[int, Dynapse2Destination]] = None,
        read_timeout: float = None,
        offset_fpga: bool = True,
        offset: float = 100e-3,
        record: bool = False,
    ) -> Tuple[np.ndarray]:
        None

        # Read Current FPGA timestamp, offset the events accordingly
        if offset_fpga:
            offset += self.current_timestamp()

        # Convert the input data to aer sequence
        event_sequence = raster_to_aer(
            input_data,
            start_time=offset,
            channel_map=channel_map,
            return_samna=True,
            dt=self.dt,
            dt_fpga=self.dt_fpga,
        )

        # Write AER packages to the bus
        self.board.grid_bus_write_events(event_sequence)
        output_events = capture_events_from_device(self.board, 5.0)
        spikes_out = aer_to_raster(output_events)
        return spikes_out, {}, {}

    def reset_time(self) -> bool:
        return self.board.reset_fpga()

    def current_timestamp(
        self,
        reading_interval: float = 10e-3,
        number_of_events: int = 10,
        retry: int = 20,
    ) -> float:
        """
        current_timestamp bounces a dummy event from FPGA to get the exact FPGA time at that moment.

        :param reading_interval: minimum time to wait for the event to bounce back, defaults to 10e-3
        :type reading_interval: float, optional
        :param number_of_events: the number of dummy events to bounce to dispatch, defaults to 10
        :type number_of_events: int, optional
        :param retry: number of retrials in the case that event is not returned back. Each time double the reading interval, defaults to 20
        :type retry: int, optional
        :raises TimeoutError: "FPGA could not respond, increase number of trials or reading interval!"
        :return: the current FPGA time in seconds
        :rtype: float
        """    

        # Flush the buffers
        self.board.output_read()

        # Generate dummy events
        events = [
            event_generator(ts, dt_fpga=self.dt_fpga)
            for ts in np.arange(
                0, reading_interval, reading_interval / number_of_events
            )
        ]

        # Send dummy event sequence to the device
        self.board.input_interface_write_events(0, events)
        time.sleep(reading_interval)

        # Try to catch them and read the last timestamp
        for __break in range(retry):
            evs = self.board.output_read()
            if len(evs) > 0:
                return evs[-1] * self.dt_fpga
            else:
                time.sleep(reading_interval)
                reading_interval *= 2

        raise TimeoutError(
            f"FPGA could not respond, increase number of trials or reading interval!"
        )
