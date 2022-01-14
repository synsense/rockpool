"""
Input layer implementation for DynapSE modules

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
07/12/2021
"""

from __future__ import annotations
import logging

from typing import Any, Dict, Optional, Tuple

from jax.lax import scan

from jax import numpy as jnp

from rockpool.nn.modules.jax.jax_module import JaxModule
from rockpool.parameters import Parameter

from rockpool.devices.dynapse.infrastructure.router import Router
from rockpool.devices.dynapse.config.simconfig import DynapSE1SimBoard
from rockpool.devices.dynapse.base import DynapSE, NeuronKey

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


class DynapSEFPGA(JaxModule):
    """
    DynapSEFPGA implements the input layer for a `DynapSEAdExpLIFJax` module. It's similar to LinearJax module but since
    the DynapSEAdExpLIFJax modules have a specific requirement that each computational unit gets 4 input spikes,
    The DynapSEFPGA is necessary.

    :param shape: Should be 2 dimensional first representing the number of virtual neurons and the second representing the real neurons in device (Nin,Nrec), defaults to None
    :type shape: Optional[Tuple], optional
    :param w_in: Initial input weights defining the connections from virtual FPGA neurons to real device neurons. It must be a rectangular matrix with shape ``(Nin, Nrec, 4)``. The last 4 holds a weight matrix for 4 different synapse types.
    :type w_in: Optional[jnp.DeviceArray], optional

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
        sim_config: Optional[DynapSE1SimBoard] = None,
        w_in: Optional[jnp.DeviceArray] = None,
        idx_map: Optional[Dict[int, NeuronKey]] = None,
        spiking_input: bool = True,
        spiking_output: bool = False,
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

        super(DynapSEFPGA, self).__init__(
            shape=shape,
            spiking_input=spiking_input,
            spiking_output=spiking_output,
            *args,
            **kwargs,
        )

        if sim_config is None:
            logging.warning("A new simconfig object is created for FPGA input module!")
            sim_config = DynapSE1SimBoard(size=self.size_out // 4)

        if len(sim_config) != self.size_out // 4:
            raise ValueError(
                f"The simulation configuration object size {len(sim_config)} and number of device neruons {self.size_out} does not match!"
            )

        if idx_map is None:
            idx_map = dict(
                zip(range(self.size_in), map(Router.decode_UID, range(self.size_in)))
            )

        weight_init = lambda s: sim_config.weight_matrix(DynapSE.poisson_CAM(s))

        if w_in is not None:
            w_in = jnp.array(w_in, dtype=jnp.float32)

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
        self, input_data: jnp.DeviceArray, record: bool = False
    ) -> Tuple[jnp.DeviceArray, None, None]:
        """
        evolve iterates through the input spike train and delivers the input spikes to the respective device neurons
        It takes a 2 dimensional spike raster in discrete time (T,N) and produces a 3 dimensional matrix delivering the
        spikes to synaptic gates [GABA_B, GABA_A, NMDA, AMPA]. It's because events are sent to the neurons but they process
        the events through their synapses. If a spike reaches to neruon 0, and it accepts from AMPA and NMDA, then AMPA and NMDA
        should process the exact same event.

        :param input_data: Input array of shape ``(T, Nin)`` to evolve over. Represents number of spikes at that timebin
        :type input_data: jnp.DeviceArray
        :param record: record the each timestep of evolution or not, defaults to False
        :type record: bool, optional
        :return: spikes_ts, state, record_dict
            :spikes_ts: spikes delivered to synaptic gates of the device neurons in shape ``(T, Nrec, 4)``
            :states: empty dictionary {}
            :record_dict: is a dictionary containing the recorded state variables during the evolution at each time step, if the ``record`` argument is ``True`` else empty dictionary {}
        :rtype: Tuple[jnp.DeviceArray, None, None]
        """

        def forward(state: Any, spike_inputs_ts: jnp.DeviceArray,) -> jnp.DeviceArray:
            """
            forward implements single time-step delivery of input spikes to device neuron's synaptic gates

            :param state: does not have any effect, just for convenince
            :type state: Any
            :param spike_inputs_ts: incoming spike raster to be used as an axis with shape [Nin]
            :type spike_inputs_ts: jnp.DeviceArray
            :return: state, spikes
                :states: None
                :spikes: logical spiking raster for each synaptic gate of each neuron over time with shape [Nrec, 4]
            :rtype: jnp.DeviceArray
            """

            spikes = jnp.dot(self.w_in.T, spike_inputs_ts).T
            return state, spikes

        # --- Evolve over spiking inputs --- #
        _, spikes_ts = scan(forward, None, input_data)

        record_dict = {}

        return spikes_ts, {}, record_dict

    @classmethod
    def from_config(
        cls,
        config: Dynapse1Configuration,
        sim_config: Optional[DynapSE1SimBoard] = None,
        default_bias: bool = True,
        *args,
        **kwargs,
    ) -> DynapSEFPGA:
        """
        from_config is a class factory method depending on a samna device configuration object. Using this,
        the virtual connections and/or input weights `w_in` should be obtained easily.

        :param config: samna Dynapse1 configuration object used to configure a network on the chip
        :type config: Dynapse1Configuration
        :param sim_config: Dynap-SE1 bias currents and simulation configuration parameters, it can be provided explicitly, or created using default settings, or can be extracted from the config bias currents. defaults to None
        :type sim_config: Optional[DynapSE1SimBoard], optional
        :param default_bias: use default bias values or get the bias parameters from the samna config, defaults to True
        :type default_bias: bool
        :return: `DynapSEFPGA` simulator input layer object
        :rtype: DynapSEFPGA
        """
        CAM_in, idx_map = Router.CAM_in_from_config(config, return_maps=True)

        # CAM_shape: size_in, size_out // 4, 4
        CAM_shape = CAM_in.shape  # N_pre, N_post, 4(syn_type)
        mod_shape = (CAM_shape[0], CAM_shape[1] * CAM_shape[2])

        if sim_config is None:
            _, idx_map_rec = Router.CAM_rec_from_config(config, return_maps=True)
            if not default_bias:
                sim_config = DynapSE1SimBoard.from_config(config, idx_map_rec)

            else:
                sim_config = DynapSE1SimBoard.from_idx_map(idx_map_rec)

        w_in = sim_config.weight_matrix(CAM_in)
        mod = cls(mod_shape, sim_config, w_in, idx_map, *args, **kwargs)
        return mod
