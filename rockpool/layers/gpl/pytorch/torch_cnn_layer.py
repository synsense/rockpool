##
# torch_cnn_layer.py - Torch implementation of a FF rockpool layer using convolutional weights
##

from .iaf_conv2d import TorchSpikingConv2dLayer
import numpy as np
import torch

# Internal class dependencies
from ....timeseries import TSEvent
from ....weights.internal.spiking_conv2d_torch import CNNWeightTorch
from ..iaf_cl import FFCLIAF

from typing import Optional, Union, List, Tuple
from warnings import warn

# - Type alias for array-like objects
ArrayLike = Union[np.ndarray, List, Tuple]

# - Absolute tolerance, e.g. for comparing float values
tol_abs = 1e-9


class FFCLIAFCNNTorch(FFCLIAF):
    """
    FFCLIAFTorch - Feedforward layer of integrate and fire neurons with constant leak
    Implemented using pytorch for speed and is meant to be used for convolutions
    """

    def __init__(
        self,
        weights: CNNWeightTorch,
        bias: float = 0,
        fVThresh: float = 8,
        fVReset: float = 0,
        fVSubtract: float = 8,
        dt: float = 1,
        monitor_id: Union[bool, int, None, ArrayLike] = [],
        name: str = "unnamed",
    ):
        """
        FFCLIAFTorch - Feedforward layer of integrate and fire neurons with constant leak

        :param weights:         array-like  Input weight matrix
        :param bias:     array-like  Constant bias to be added to state at each time step
        :param v_thresh:   array-like  Spiking threshold
        :param v_reset:    array-like  Reset potential after spike (also see param bSubtract)
        :param v_subtract: array-like  If not None, subtract provided values
                                        from neuron state after spike. Otherwise will reset.
        :monitor_id:       array-like  IDs of neurons to be recorded
        :param name:     str  Name of this layer.
        """
        
        warn("FFCLIAFCNNTorch: This layer is deprecated.")
        
        # Call parent constructor
        FFCLIAF.__init__(
            self,
            weights=weights,
            bias=bias,
            v_thresh=fVThresh,
            v_reset=fVReset,
            v_subtract=fVSubtract,
            dt=dt,
            monitor_id=monitor_id,
            name=name,
        )

        self.fVThresh = fVThresh
        self.fVSubtract = fVSubtract
        self.fVReset = fVReset

        # Placeholder variable
        self._lyrTorch = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.reset_state()

    @property
    def lyrTorch(self):
        if self._lyrTorch is None:
            self._init_torch_layer()
        return self._lyrTorch

    @lyrTorch.setter
    def lyrTorch(self, lyrNewTorch):
        self._lyrTorch = lyrNewTorch

    def _init_torch_layer(self):
        # Initialize torch layer
        self.lyrTorch = TorchSpikingConv2dLayer(
            nInChannels=self.weights.nInChannels,
            nOutChannels=self.weights.kernels,
            kernel_size=self.weights.kernel_size,
            strides=self.weights.strides,
            padding=self.weights.padding,
            fVThresh=self.fVThresh,
            fVSubtract=self.fVSubtract,
            fVReset=self.fVReset,
        )
        # Set biases
        self.lyrTorch.conv.bias.data = torch.zeros(self.weights.kernels)
        # Set torch weights
        if self.weights.img_data_format == "channels_first":
            self.lyrTorch.conv.weight.data = torch.from_numpy(self.weights.data).float()
        elif self.weights.img_data_format == "channels_last":
            weights = self.weights.data
            self.lyrTorch.conv.weight.data = torch.from_numpy(
                weights.transpose((3, 2, 0, 1))
            ).float()
        else:
            raise Exception(
                "img_data_format(={}) not understood".format(
                    self.weights.img_data_format
                )
            )

        # Transfer layer to appropriate device
        self.lyrTorch.to(self.device)
        return

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
                        self.name + ": Expanding input dimensions to match layer size."
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

        :param tsSpkInput:      TSEvent  Input spike trian
        :param duration:       float    Simulation/Evolution time
        :param num_timesteps    int      Number of evolution time steps
        :param verbose:        bool     Currently no effect, just for conformity
        :return:            TSEvent  output spike series

        """
        # Compute number of simulation time steps
        if num_timesteps is None:
            num_timesteps = int((duration + tol_abs) // self.dt)

        # - Generate input in rasterized form
        inp_spike_raster = self._prepare_input(ts_input, num_timesteps=num_timesteps)

        # Convert input to torch tensors
        inp_spike_raster = [next(inp_spike_raster) for i in range(num_timesteps)]
        print(sum(inp_spike_raster))
        tsrIn = torch.from_numpy(np.array(inp_spike_raster, np.uint8)).type(torch.float)
        # Reshape flat data to images and channels
        tsrInReshaped = tsrIn.reshape(-1, *self.weights.inp_shape)
        print(tsrInReshaped.shape)
        # Restructure input
        if self.weights.img_data_format == "channels_last":
            tsrInReshaped = tsrInReshaped.permute((0, 3, 1, 2))
        elif self.weights.img_data_format == "channels_first":
            pass

        # Process data
        tsrInReshaped = tsrInReshaped.to(self.device)
        tsrOut = self.lyrTorch(tsrInReshaped)

        # Reshape data again to the class's format
        if self.weights.img_data_format == "channels_last":
            tsrOut = tsrOut.permute((0, 2, 3, 1))
        elif self.weights.img_data_format == "channels_first":
            pass
        # Flatten output and return
        out_raster = tsrOut.cpu().numpy()
        out_raster = out_raster.reshape((num_timesteps, -1))

        # Create time series from raster
        vnTimeSteps, vnChannels = np.nonzero(out_raster)
        times = self.t + (vnTimeSteps + 1) * self.dt

        # Update time
        self._timestep += num_timesteps

        out_events = TSEvent(times, vnChannels, num_channels=self.size, name=self.name)
        return out_events
