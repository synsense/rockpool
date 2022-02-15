"""
An LIF spiking neuron with a Jax backend, implementing an explicit ODE solver
"""

from typing import Tuple

from .lif_jax import LIFJax, step_pwl, sigmoid

import jax
from jax import numpy as np
from jax.lax import scan
import jax.random as rand


class LIFODEJax(LIFJax):
    """
    An LIF spiking neuron module, implementing an explicit ODE system with a Jax backend

    """

    def evolve(
        self,
        input_data: np.ndarray,
        record: bool = False,
    ) -> Tuple[np.ndarray, dict, dict]:
        """

        Args:
            input_data (np.ndarray): Input array of shape ``(T, Nin)`` to evolve over
            record (bool): If ``True``,

        Returns:
            (np.ndarray, dict, dict): output, new_state, record_state
            ``output`` is an array with shape ``(T, Nout)`` containing the output data produced by this module. ``new_state`` is a dictionary containing the updated module state following evolution. ``record_state`` will be a dictionary containing the recorded state variables for this evolution, if the ``record`` argument is ``True``.
        """
        # - Get input shapes, add batch dimension if necessary
        input_data, (vmem, spikes, isyn) = self._auto_batch(
            input_data,
            (self.vmem, self.spikes, self.isyn),
            (
                (self.size_out,),
                (self.size_out,),
                (self.size_out, self.n_synapses),
            ),
        )
        n_batches, n_timesteps, _ = input_data.shape

        # - Reshape data over separate input synapses
        input_data = input_data.reshape(
            n_batches, n_timesteps, self.size_out, self.n_synapses
        )

        # - Get evolution constants
        noise_zeta = self.noise_std * np.sqrt(self.dt)

        # - Generate membrane noise trace
        key1, subkey = rand.split(self.rng_key)
        noise_ts = noise_zeta * rand.normal(
            subkey, shape=(n_batches, n_timesteps, self.size_out)
        )

        # - Single-step LIF dynamics
        def forward(
            state: Tuple[np.ndarray, np.ndarray, np.ndarray],
            inputs_t: Tuple[np.ndarray, np.ndarray],
        ) -> (
            Tuple[np.ndarray, np.ndarray, np.ndarray],
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            np.ndarray,
        ):
            """
            Single-step LIF dynamics for a recurrent LIF layer

            :param LayerState state:
            :param Tuple[np.ndarray, np.ndarray] inputs_t: (spike_inputs_ts, current_inputs_ts)

            :return: (state, Irec_ts, output_ts, surrogate_ts, spikes_ts, Vmem_ts, Isyn_ts)
                state:          (Tuple[np.ndarray, np.ndarray, np.ndarray]) Layer state at end of evolution
                Irec_ts:        (np.ndarray) Recurrent input received at each neuron over time [T, N]
                output_ts:      (np.ndarray) Weighted output surrogate over time [T, O]
                surrogate_ts:   (np.ndarray) Surrogate time trace for each neuron [T, N]
                spikes_ts:      (np.ndarray) Logical spiking raster for each neuron [T, N]
                Vmem_ts:        (np.ndarray) Membrane voltage of each neuron over time [T, N]
                Isyn_ts:        (np.ndarray) Synaptic input current received by each neuron over time [T, N]
            """
            # - Unpack inputs
            (sp_in_t, noise_in_t) = inputs_t

            # - Unpack state
            spikes, isyn, vmem = state

            # - Apply synaptic and recurrent input
            d_isyn = -isyn + sp_in_t
            irec = np.dot(spikes, self.w_rec).reshape(self.size_out, self.n_synapses)
            d_isyn = d_isyn + irec
            isyn = isyn + d_isyn * self.dt / self.tau_syn

            # - Integrate membrane potentials
            d_vmem = -vmem + isyn.sum(1) + noise_in_t + self.bias
            vmem = vmem + d_vmem * self.dt / self.tau_mem

            # - Detect next spikes (with custom gradient)
            spikes = step_pwl(vmem, self.threshold, 0.5, self.max_spikes_per_dt)

            # - Apply subtractive membrane reset
            vmem = vmem - spikes * self.threshold

            # - Return state and outputs
            return (spikes, isyn, vmem), (irec, spikes, vmem, isyn)

        # - Map over batches
        @jax.vmap
        def scan_time(spikes, isyn, vmem, input_data, noise_ts):
            return scan(forward, (spikes, isyn, vmem), (input_data, noise_ts))

        # - Evolve over spiking inputs
        state, (irec_ts, spikes_ts, vmem_ts, isyn_ts) = scan_time(
            spikes, isyn, vmem, input_data, noise_ts
        )

        # - Generate output surrogate
        surrogate_ts = sigmoid(vmem_ts * 20.0, self.threshold)

        # - Generate return arguments
        outputs = spikes_ts
        states = {
            "spikes": spikes_ts[0, -1],
            "isyn": isyn_ts[0, -1],
            "vmem": vmem_ts[0, -1],
            "rng_key": key1,
        }

        record_dict = {
            "irec": irec_ts,
            "spikes": spikes_ts,
            "isyn": isyn_ts,
            "vmem": vmem_ts,
            "U": surrogate_ts,
        }

        # - Return outputs
        return outputs, states, record_dict
