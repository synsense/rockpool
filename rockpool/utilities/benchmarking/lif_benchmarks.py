"""
Define benchmark functions for LIF layers
"""

__all__ = [
    "jax_lif_nojit_benchmark",
    "jax_lif_jit_cpu_benchmark",
    "jax_lif_jit_gpu_benchmark",
    "jax_lif_jit_tpu_benchmark",
    "lif_benchmark",
    "lif_torch_benchmark",
    "lif_torch_cuda_benchmark",
    "lif_torch_cuda_graph_benchmark",
    "lif_exodus_cuda_benchmark",
    "all_lif_benchmarks",
]


def jax_lif_nojit_benchmark():
    from rockpool.nn.modules import LIFJax
    import numpy as np

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = LIFJax(layer_size)
        input_static = np.random.rand(batch_size, time_steps, layer_size)

        mod(input_static)

        bench_obj = (layer_size, mod, input_static)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, _, _) = bench_obj
        LIFJax(layer_size)

    def evolve_fn(bench_obj):
        (_, mod, input_static) = bench_obj
        mod(input_static)

    benchmark_title = f"LIFJax, no JIT"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


def jax_lif_jit_cpu_benchmark():
    from rockpool.nn.modules import LIFJax
    import numpy as np
    import jax

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = jax.jit(LIFJax(layer_size), backend="cpu")
        input_static = np.random.rand(batch_size, time_steps, layer_size)

        mod(input_static)

        bench_obj = (layer_size, mod, input_static)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, mod, input_static) = bench_obj
        jax.jit(LIFJax(layer_size), backend="cpu")

    def evolve_fn(bench_obj):
        (layer_size, mod, input_static) = bench_obj
        mod(input_static)

    benchmark_title = f"LIFJax, with CPU JIT compilation"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


def jax_lif_jit_gpu_benchmark():
    from rockpool.nn.modules import LIFJax
    import numpy as np
    import jax

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = jax.jit(LIFJax(layer_size), backend="gpu")
        input_static = np.random.rand(batch_size, time_steps, layer_size)

        mod(input_static)

        bench_obj = (layer_size, mod, input_static)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, _, _) = bench_obj
        jax.jit(LIFJax(layer_size), backend="gpu")

    def evolve_fn(bench_obj):
        (_, mod, input_static) = bench_obj
        mod(input_static)

    benchmark_title = f"LIFJax, with GPU JIT compilation"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


def jax_lif_jit_tpu_benchmark():
    from rockpool.nn.modules import LIFJax
    import numpy as np
    import jax

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = jax.jit(LIFJax(layer_size), backend="tpu")
        input_static = np.random.rand(batch_size, time_steps, layer_size)

        mod(input_static)

        bench_obj = (layer_size, mod, input_static)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, _, _) = bench_obj
        jax.jit(LIFJax(layer_size), backend="tpu")

    def evolve_fn(bench_obj):
        (_, mod, input_static) = bench_obj
        mod(input_static)

    benchmark_title = f"LIFJax, with TPU JIT compilation"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


def lif_benchmark():
    from rockpool.nn.modules import LIF
    import numpy as np

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = LIF(layer_size)
        input_static = np.random.rand(batch_size, time_steps, layer_size)

        mod(input_static)

        bench_obj = (layer_size, mod, input_static)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, _, _) = bench_obj
        LIF(layer_size)

    def evolve_fn(bench_obj):
        (_, mod, input_static) = bench_obj
        mod(input_static)

    benchmark_title = f"LIF with no acceleration (numpy backend)"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


def lif_torch_benchmark():
    from rockpool.nn.modules import LIFTorch
    import torch

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = LIFTorch(layer_size)
        input_static = torch.randn(batch_size, time_steps, layer_size)

        mod(input_static)

        bench_obj = (layer_size, mod, input_static)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, _, _) = bench_obj
        LIFTorch(layer_size)

    def evolve_fn(bench_obj):
        (_, mod, input_static) = bench_obj
        mod(input_static)

    benchmark_title = f"LIFTorch on CPU"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


def lif_torch_cuda_benchmark():
    from rockpool.nn.modules import LIFTorch
    import torch

    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = LIFTorch(layer_size).cuda()
        input_static = torch.randn(batch_size, time_steps, layer_size, device="cuda")

        mod(input_static)

        bench_obj = (layer_size, mod, input_static)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, _, _) = bench_obj
        LIFTorch(layer_size).cuda()

    def evolve_fn(bench_obj):
        (_, mod, input_static) = bench_obj
        mod(input_static)

    benchmark_title = f"LIFTorch on a CUDA device"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


def lif_torch_cuda_graph_benchmark():
    from rockpool.nn.modules import LIFTorch
    import torch

    class StepPWL(torch.autograd.Function):
        """
        Heaviside step function with piece-wise linear surrogate to use as spike-generation surrogate
        """

        @staticmethod
        def forward(
            ctx,
            x,
            threshold=torch.tensor(1.0),
            window=torch.tensor(0.5),
            max_spikes_per_dt=torch.tensor(2.0**16),
        ):
            ctx.save_for_backward(x, threshold)
            ctx.window = window
            nr_spikes = ((x >= threshold) * torch.floor(x / threshold)).float()
            # nr_spikes[nr_spikes > max_spikes_per_dt] = max_spikes_per_dt.float()
            clamp_bool = (nr_spikes > max_spikes_per_dt).float()
            nr_spikes -= (nr_spikes - max_spikes_per_dt.float()) * clamp_bool
            return nr_spikes

        @staticmethod
        def backward(ctx, grad_output):
            x, threshold = ctx.saved_tensors
            grad_x = grad_threshold = grad_window = grad_max_spikes_per_dt = None

            mask = x >= (threshold - ctx.window)
            if ctx.needs_input_grad[0]:
                grad_x = grad_output / threshold * mask

            if ctx.needs_input_grad[1]:
                grad_threshold = -x * grad_output / (threshold**2) * mask

            return grad_x, grad_threshold, grad_window, grad_max_spikes_per_dt

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = LIFTorch(
            layer_size, spike_generation_fn=StepPWL, max_spikes_per_dt=2.0**16
        ).cuda()
        input_static = torch.randn(batch_size, time_steps, layer_size, device="cuda")

        # - Warm up the CUDA stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                y_pred, _, _ = mod(input_static)

        # - Capture the graph
        g = torch.cuda.CUDAGraph()

        with torch.cuda.graph(g):
            static_y_pred, _, _ = mod(input_static)

        bench_obj = (layer_size, mod, input_static, g, static_y_pred)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, mod, input_static, g, static_y_pred) = bench_obj
        mod = LIFTorch(
            layer_size, spike_generation_fn=StepPWL, max_spikes_per_dt=2.0**16
        ).cuda()

        # - Warm up the CUDA stream
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for i in range(3):
                y_pred, _, _ = mod(input_static)

    def evolve_fn(bench_obj):
        (layer_size, mod, input_static, g, static_y_pred) = bench_obj
        g.replay()

    benchmark_title = f"LIFTorch using CUDA graph replay acceleration"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


def lif_exodus_cuda_benchmark():
    from rockpool.nn.modules import LIFExodus
    import torch

    assert torch.cuda.is_available(), "CUDA is required for this benchmark"

    def prepare_fn(batch_size, time_steps, layer_size):
        mod = LIFExodus(layer_size).cuda()
        input_static = torch.randn(batch_size, time_steps, layer_size, device="cuda")

        mod(input_static)

        bench_obj = (layer_size, mod, input_static)

        return bench_obj

    def create_fn(bench_obj):
        (layer_size, _, _) = bench_obj
        LIFExodus(layer_size).cuda()

    def evolve_fn(bench_obj):
        (_, mod, input_static) = bench_obj
        mod(input_static)

    benchmark_title = f"LIFExodus on a CUDA device"

    return prepare_fn, create_fn, evolve_fn, benchmark_title


all_lif_benchmarks = [
    lif_exodus_cuda_benchmark,
    lif_benchmark,
    lif_torch_cuda_graph_benchmark,
    lif_torch_cuda_benchmark,
    lif_torch_benchmark,
    jax_lif_nojit_benchmark,
    jax_lif_jit_cpu_benchmark,
    jax_lif_jit_gpu_benchmark,
    jax_lif_jit_tpu_benchmark,
]
