#include <torch/extension.h>
#include <iostream>

#include "threshold.cpp"
#include "bitshift.cpp"

tensor_list lif_forward(torch::Tensor data,
                        torch::Tensor vmem,
                        torch::Tensor isyn,
                        torch::Tensor alpha_mem,
                        torch::Tensor alpha_syn,
                        torch::Tensor tau_mem,
                        torch::Tensor tau_syn,
                        double threshold,
                        double window,
                        bool record)
{
    int time_steps = data.size(0);
    int n_batches = data.size(1);
    int n_synapses = data.size(2);
    int n_neurons = data.size(3);

    auto options = data.options();
    torch::Tensor out_spikes = torch::ones({time_steps, n_batches, n_neurons}, options);

    torch::Tensor vmem_rec;
    torch::Tensor isyn_rec;

    if (record)
    {
        vmem_rec = torch::zeros({time_steps, n_batches, n_neurons}, options);
        isyn_rec = torch::zeros({time_steps, n_batches, n_synapses, n_neurons}, options);
    }
    else
    {
        vmem_rec = torch::zeros({1, n_batches, n_neurons}, options);
        isyn_rec = torch::zeros({1, n_batches, n_synapses, n_neurons}, options);
    }
        
    torch::Tensor dash_mem = torch::tensor({2.}, options).pow(alpha_mem);
    torch::Tensor dash_syn = torch::tensor({2.}, options).pow(alpha_syn);

    // TODO the use of accessors could speed things up?!
    //auto data_accessor = data.packed_accessor64<int, 4>();
    //auto out_spikes_accessor = out_spikes.packed_accessor64<int, 3>();

    for (int t = 0; t < time_steps; ++t)
    {
        // Spike generation
        auto out = ThresholdSubtract::apply(vmem, threshold, window);
        out_spikes[t] = out;

        // Membrane reset
        vmem = vmem - out*threshold;

        if (record)
        {
            // record
            vmem_rec[t] = vmem;
            isyn_rec[t] = isyn;
        }

        // Integrate input
        isyn = isyn + data[t];

        vmem = BitshiftDecay::apply(vmem, dash_mem, tau_mem);
        isyn = BitshiftDecay::apply(isyn, dash_syn, tau_syn);
        
        // State propagation
        vmem = vmem + isyn.sum(1);
    }                             
 
    if (!record)
    {
        // just record the last vmem and isyn
        vmem_rec[0] = vmem;
        isyn_rec[0] = isyn;
    }    
    return {out_spikes, vmem_rec, isyn_rec};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lif_forward, "LIF forward");
}


