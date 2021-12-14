"""
Input layer implementation for DynapSE modules

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
07/12/2021
"""
from __future__ import annotations

from typing import (
    Optional,
    Tuple,
    Any,
    Dict,
)
from rockpool.typehints import FloatVector

from jax.lax import scan
from jax import numpy as np

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter

from rockpool.devices.dynapse.simconfig import DynapSE1SimBoard
from rockpool.devices.dynapse.adexplif_jax import poisson_weight_matrix
from rockpool.devices.dynapse.router import Router, NeuronKey

_SAMNA_AVAILABLE = True

try:
    from samna.dynapse1 import Dynapse1Configuration
except ModuleNotFoundError as e:
    Dynapse1Configuration = Any
    print(
        e,
        "\DynapSEFPGA module can only be constructed manually."
        "automatic factory methods depends on samna!",
    )
    _SAMNA_AVAILABLE = False

_NETGEN_AVAILABLE = True

try:
    from netgen import (
        NetworkGenerator,
    )
except ModuleNotFoundError as e:
    NetworkGenerator = Any
    print(
        e,
        "\nDynapSEFPGA object factory from the NetworkGenerator object is not possible!",
    )
    _NETGEN_AVAILABLE = False


class DynapSEFPGA(JaxModule):
    """
    DynapSEFPGA implements the input layer for a `DynapSEAdExpLIFJax` module. It's similar to LinearJax module but since
    the DynapSEAdExpLIFJax modules have a specific requirement that each computational unit gets 4 input spikes,
    The DynapSEFPGA is necessary. In addition, the input weights can be obtained from a samna configuration object or a INI netgen object.

    :param shape: Should be 2 dimensional first representing the number of virtual neurons and the second representing the real neurons in device (Nin,Nrec), defaults to None
    :type shape: Optional[Tuple], optional
    :param w_in: Initial input weights defining the connections from virtual FPGA neurons to real device neurons. It must be a rectangular matrix with shape ``(Nin, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types.
    :type w_in: Optional[FloatVector], optional

        Let's say 3 virtual neurons allocated to send events to 5 device neurons

        # Gb Ga N  A
        [[0, 0, 0, 1],  # pre = 0 (virtual) post = 0 (device)
         [0, 0, 0, 1],  #                   post = 1 (device)
         [0, 0, 0, 0],  #                   post = 2 (device)
         [0, 0, 0, 0],  #                   post = 3 (device)
         [0, 0, 0, 1]], #                   post = 4 (device)

        [[0, 0, 0, 0],  # pre = 1 (virtual)
         [0, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 0, 1, 0],
         [0, 0, 0, 1]],

        [[0, 0, 0, 0],  # pre = 3 (virtual)
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 0, 0, 0],
         [0, 1, 0, 0]]],

    Virtual(External Input)

        AMPA : 1 from s0 to n0, 1 from s0 to n1, 1 from s0 to n4, 1 from s1 to n4
        NMDA : 1 from s1 to n2, 1 from s1 to n3
        GABA_A : 1 from s2 to n4
        GABA_B : -

    :param idx_map: a dictionary of the mapping between matrix indexes of the virtual neurons and their global unique neuron keys.
    :type idx_map: Dict[int, NeuronKey]

        idx_map = {
            0: (0, 1, 20),
            1: (0, 1, 36),
            2: (0, 1, 60),
        }

    :param spiking_input: Whether this module receives spiking input, defaults to True
    :type spiking_input: bool, optional
    :param spiking_output: Whether this module produces spiking output, defaults to True
    :type spiking_output: bool, optional

    :raises ValueError: `shape` must specify input and output sizes (Nin,Nrec) for DynapSEFPGA
    :raises ValueError: The size of index map is different than number of virtual neurons defined

    [] TODO: parametric fill rate, different initialization methods
    """

    __doc__ += "\nJaxModule" + JaxModule.__doc__

    def __init__(
        self,
        shape: Optional[Tuple] = None,
        w_in: Optional[FloatVector] = None,
        idx_map: Optional[Dict[int, NeuronKey]] = None,
        spiking_input: bool = True,
        spiking_output: bool = True,
        *args,
        **kwargs,
    ):
        """
        __init__ Initialize ``DynapSEFPGA`` module. Parameters are explained in the superclass docstring.
        """

        # Check the parameters and initialize to default if necessary
        if shape is None or len(shape) != 2:
            raise ValueError(
                "`shape` must specify input and output sizes (Nin,Nrec*4) for DynapSEFPGA! shape={shape}"
            )

        # Check the network size and the recurrent weight vector accordingly
        syn_size_check = lambda s: s == (s // 4) * 4  # 4 synapse per neuron for sure

        # Check if input dimension meets the 4 synapse per neuron criteria
        if not syn_size_check(shape[1]):
            raise ValueError(
                f"Output dimension ({shape[1]},..) should have been multiples of 4! (Go for {shape[1]//4}, {(shape[1]+4)//4}, or {shape[1]*4}) \n"
                f"Each neuron holds 4 synaptic state, which means 4 input gates per neuron!\n"
                f"i.e. ({(shape[1]//4)*4},..) means {shape[1]//4} neurons with 4 synapses\n"
                f"i.e. ({((shape[1]+4)//4)*4},..) means {(shape[1]+4)//4} neurons with 4 synapses\n"
                f"i.e. ({shape[1]*4},..) means {shape[1]} neurons with 4 synapses\n"
            )

        super().__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        if idx_map is None:
            idx_map = dict(
                zip(range(self.size_in), map(Router.decode_UID, range(self.size_in)))
            )

        weight_init = lambda s: poisson_weight_matrix(s)

        # - Specify weight parameter
        self.w_in = Parameter(
            w_in,
            shape=(self.size_in, self.size_out // 4, 4),
            init_func=weight_init,
            family="weights",
        )

        # Check the index_map
        if len(idx_map.keys()) != self.size_in:
            raise ValueError(
                "The size of index map is different than number of virtual neurons defined"
            )
        DynapSE1SimBoard.check_neuron_id_order(list(idx_map.keys()))
        self.idx_map = idx_map

    def evolve(
        self, input_data: np.ndarray, record: bool = False
    ) -> Tuple[np.ndarray, None, None]:
        """
        evolve iterates through the input spike train and delivers the input spikes to the respective device neurons
        It takes a 2 dimensional spike raster in discrete time (T,N) and produces a 3 dimensional matrix delivering the
        spikes to synaptic gates [GABA_B, GABA_A, NMDA, AMPA]. It's because events are sent to the neurons but they process
        the events through their synapses. If a spike reaches to neruon 0, and it accepts from AMPA and NMDA, then AMPA and NMDA
        should process the exact same event.

        :param input_data: Input array of shape ``(T, Nin)`` to evolve over. Represents number of spikes at that timebin
        :type input_data: np.ndarray
        :param record: record the each timestep of evolution or not, defaults to False
        :type record: bool, optional
        :return: spikes_ts, state, record_dict
            :spikes_ts: spikes delivered to synaptic gates of the device neurons in shape ``(T, Nrec, 4)``
            :states: empty dictionary {}
            :record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True`` else empty dictionary {}
        :rtype: Tuple[np.ndarray, None, None]
        """

        def forward(
            state: Any,
            spike_inputs_ts: np.ndarray,
        ) -> np.ndarray:
            """
            forward implements single time-step delivery of input spikes to device neuron's synaptic gates

            :param state: does not have any effect, just for convenince
            :type state: Any
            :param spike_inputs_ts: incoming spike raster to be used as an axis with shape [Nin]
            :type spike_inputs_ts: np.ndarray
            :return: state, spikes
                :states: None
                :spikes: logical spiking raster for each synaptic gate of each neuron over time with shape [Nrec, 4]
            :rtype: np.ndarray
            """

            spikes = np.dot(self.w_in.T, spike_inputs_ts).T
            return state, spikes

        # --- Evolve over spiking inputs --- #
        _, spikes_ts = scan(forward, None, input_data)

        record_dict = {}

        return spikes_ts, {}, record_dict

    @classmethod
    def from_config(
        cls,
        config: Dynapse1Configuration,
        *args,
        **kwargs,
    ) -> DynapSEFPGA:
        """
        from_config is a class factory method depending on a samna device configuration object. Using this,
        the virtual connections and/or input weights `w_in` should be obtained easily.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :return: `DynapSEFPGA` simulator input layer object
        :rtype: DynapSEFPGA
        """
        w_in, idx_map = Router.w_in_from_config(config, return_maps=True)
        in_shape = w_in.shape  # size_in, size_out // 4, 4
        shape = (in_shape[0], in_shape[1] * in_shape[2])
        mod = cls(shape, w_in, idx_map, *args, **kwargs)
        return mod

    @classmethod
    def from_netgen(cls, netgen: NetworkGenerator, *args, **kwargs) -> DynapSEFPGA:
        """
        from_netgen is a class factory which makes it easier to get a `DynapSEFPGA` object using the `NetworkGenerator` object

        :param netgen: network generator object defined in samna/ctxctl_contrib/netgen
        :type netgen: NetworkGenerator
        :return: `DynapSEFPGA` simulator input layer object
        :rtype: DynapSEFPGA
        """
        config = netgen.make_dynapse1_configuration()
        mod = cls.from_config(config, *args, **kwargs)
        return mod
