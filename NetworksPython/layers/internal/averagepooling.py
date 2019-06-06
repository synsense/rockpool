###
# averagepooling.py - Class implementing a pooling layer.
###

import numpy as np
import skimage.measure
from warnings import warn
from typing import Optional, Union, List, Tuple

# from typing import Optional, Union, List, Tuple
from ...timeseries import TSEvent

from .iaf_cl import CLIAF
from ...weights import CNNWeight

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9

# - Type alias for array-like objects

ArrayLike = Union[np.ndarray, List, Tuple]


class AveragePooling2D(CLIAF):
    """
    AveragePooling: Implements average pooling by simply merging inputs. So this is more of sum than average pooling.
    """

    def __init__(
        self,
        inp_shape: tuple,
        pool_size: tuple = (1, 1),
        img_data_format="channels_last",
        dt: float = 1,
        name: str = "unnamed",
    ):
        """
        :param inp_shape:     tuple Input shape
        :param pool_size:   tuple Pooling width along each dimension
        :param img_data_format: str 'channels_first' or 'channels_last'
        :param name:     str  Name of this layer.
        :param dt:         float  Time step for simulations
        """

        if img_data_format == "channels_last":
            kernels = inp_shape[-1]
        elif img_data_format == "channels_first":
            kernels = inp_shape[0]
        self.img_data_format = img_data_format
        weights = CNNWeight(
            inp_shape=inp_shape,
            kernels=kernels,
            kernel_size=pool_size,
            strides=pool_size,
            img_data_format=img_data_format,
        )  # Simple hack

        # Call parent constructor
        super().__init__(weights_in=weights, dt=dt, name=name)
        self.pool_size = pool_size
        self.reset_state()

    def _prepare_input(
        self, ts_input: Optional[TSEvent] = None, num_timesteps: int = 1
    ) -> np.ndarray:
        """
        Prepare input stream and return a binarized vector of spikes
        """
        # - End time of evolution
        t_final = self.t + num_timesteps * self.dt
        # - Extract spike timings and channels
        if ts_input is not None:
            if ts_input.isempty():
                # Return an empty list with all zeros
                spike_raster = np.zeros((self.size_in), bool)
            else:
                # Ensure number of channels is atleast as many as required
                try:
                    assert ts_input.num_channels >= self.size_in
                except AssertionError as err:
                    warn(
                        self.name
                        + ": Expanding input dimensions to match layer size."
                    )
                    ts_input.num_channels = self.size_in

                # Extract spike data from the input variable
                spike_raster = ts_input.xraster(
                    dt=self.dt, t_start=self.t, t_stop=t_final
                )

                ## - Make sure size is correct
                # spike_raster = spike_raster[:num_timesteps, :]
                # assert spike_raster.shape == (num_timesteps, self.size_in)
                yield from spike_raster  # Yield a single time step
                return
        else:
            # Return an empty list with all zeros
            spike_raster = np.zeros((self.size_in), bool)

        while True:
            yield spike_raster

    def evolve(
        self,
        ts_input: Optional[TSEvent] = None,
        duration: Optional[float] = None,
        num_timesteps: Optional[int] = None,
        verbose: bool = False,
    ) -> TSEvent:
        """
        evolve : Function to evolve the states of this layer given an input

        :param tsSpkInput:  TSEvent  Input spike trian
        :param duration:   float    Simulation/Evolution time
        :param verbose:    bool Currently no effect, just for conformity
        :return:          TSEvent  output spike series

        """

        # Compute number of simulation time steps
        if num_timesteps is None:
            num_timesteps = int((duration + tol_abs) // self.dt)

        # - Generate input in rasterized form, get actual evolution duration
        inp_spike_raster = np.array(list(self._prepare_input(ts_input, num_timesteps)))

        # Reshape input data
        inp_spike_raster = inp_spike_raster.reshape((-1, *self.weights.inp_shape))

        # Do average pooling here
        if self.img_data_format == "channels_last":
            out_raster = skimage.measure.block_reduce(
                inp_spike_raster, (1, *self.pool_size, 1), np.sum
            )
        elif self.img_data_format == "channels_first":
            out_raster = skimage.measure.block_reduce(
                inp_spike_raster, (1, 1, *self.pool_size, 1), np.sum
            )

        # Reshape the output to flat indices
        out_raster = out_raster.reshape((-1, self.size))

        # Convert raster to indices
        spike_times, spike_ids = np.nonzero(out_raster)

        # Convert arrays to TimeSeries objects
        event_out = TSEvent(
            times=spike_times, channels=spike_ids, num_channels=self.size
        )

        # Update time
        self._t += self.dt * num_timesteps

        return event_out

    @property
    def weights(self):
        return self._weights_in

    @weights.setter
    def weights(self, new_w):
        self.weights_in = new_w
