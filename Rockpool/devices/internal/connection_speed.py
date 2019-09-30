from pathlib import Path
import numpy as np
from NetworksPython.devices import DynapseControl

filedir = Path(__file__).parent

# - Setup
dc = DynapseControl(init_chips=[0, 1], clearcores_list=list(range(8)))
biasdir = filedir.parent.parent / "Documentation" / "Reservoir tutorial"
dc.load_biases(biasdir / "biases.py")
dc.copy_biases(0, range(4, 8))

# - Connect 1 neuron to external input
dc.add_connections_from_weights(
    weights=[1], neuron_ids=[1], neuron_ids_post=[2], virtual_pre=True, apply_diff=True
)

# - Start sending spikes
dc.start_cont_stim(40, 1)

# - Connect to neurons
for n_neurons in [1, 10, 100, 500, 1000, 2000]:
    for i in range(5):
        print(n_neurons)
        dc.add_connections_from_weights(
            weights=np.ones((1, n_neurons)),
            neuron_ids=[2],
            neuron_ids_post=list(range(3, n_neurons + 3)),
            virtual_pre=False,
            apply_diff=True,
        )
    # dc.remove_all_connections_from([2])
    dc.reset_all()
