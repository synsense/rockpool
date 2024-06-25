"""
Utilities for performing benchmarking
"""

from typing import Callable, Tuple, List, Optional

from time import time
import warnings

try:
    from tqdm.auto import tqdm
except:

    def tqdm(obj):
        return obj


__all__ = ["timeit", "benchmark_neurons", "plot_benchmark_results"]


def timeit(
    callable: Callable,
    min_runs: int = 3,
    max_runs=5000,
    min_time: float = 2.0,
    warmup_calls: int = 1,
) -> List[float]:
    """
    Time the execution of a callable

    :func:`timeit` will run the function handle ``callable`` several times, and collect execution times. It first performs warm-up by calling the function one or more times (argument ``warmup_calls``). It then repeatedly executes the function until at least ``min_time`` seconds of execution time have elapsed, calling the function at least ``min_runs`` times, and at most ``max_runs`` times.

    :func:`timeit` returns the full list of collected execution times.

    Arguments:
        callable (Callable): A function to time the execution of. Must accept no arguments.
        min_runs (int): The minimum number of calls to make. Default: ``3``
        max_runs (int): The maximum number of calls to make. Default: ``5000``
        min_time (float): The minimum total execution time, in seconds. Default: ``2.``
        warmup_calls (int): The number of warm-up calls to make. Default: ``1``
    """
    # - Warmup
    for _ in range(warmup_calls):
        callable()

    # - Take at least min_time seconds, at least min_runs runs
    exec_count = 0
    t_total = 0.0
    collected_times = []
    while ((t_total <= min_time) or (exec_count < min_runs)) and (
        exec_count < max_runs
    ):
        # - Time a single run
        t_start = time()
        callable()
        collected_times.append(time() - t_start)

        exec_count += 1
        t_total = sum(collected_times)

    return collected_times


def benchmark_neurons(
    prepare_fn: Callable,
    create_fn: Callable,
    evolve_fn: Callable,
    benchmark_desc: Optional[str] = None,
    layer_sizes: List[int] = [1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000, 5000, 10000],
    num_batches: int = 10,
    num_timesteps: int = 1000,
) -> Tuple[List, List, List]:
    """
    Benchmark the creation and evolution of neuron layers

    A benchmark is defined as three functions :func:`prepare_fn`, :func:`create_fn`, :func:`evolve_fn`.
    :func:`prepare_fn` is called once per layer parameter set, to initialise the benchmark.
    It has the signature ``def prepare_fn(num_batches: int, num_timesteps: int, layer_size: int) -> object``.
    :func:`prepare_fn` may return an arbitrary python object ``bench_obj``, which is passed to the other benchmarking functions.
    :func:`prepare_fn` is not benchmarked, and can execute arbitrary code --- whatever is required to set up the benchmark.


    :func:`create_fn` is used to benchmark the creation of a neuron layer. It should do as little as possible, ideally just calling the layer creation constructor.
    It has the signature ``def create_fn(bench_obj: object) -> None``.
    :func:`create_fn` will be called many times as part of the benchmarking process.

    :func:`evolve_fn` is used to benchmark the evolution of the layer.
    It has the signature ``def evolve_fn(bench_obj: object) -> None``.
    :func:`evolve_fn` should do as little as possible besides evolve the layer.
    Random static data and layer creation should ideally be performed in :func:`prepare_fn` and passed as elements of ``bench_obj``.
    :func:`evolve_fn` will be called many times as part of the benchmarking process.

    :func:`.benchmark_neurons` will run these functions many times over increasing layer sizes.
    It will return lists of collected benchmark times (excluding warm-up) for layer construction and evolution, as well as a list of the layer sizes that were benchmarked: ``creation_times, evolution_times, layer_sizes``.

    You can conveniently plot the results of a benchmark with ``matplotlib.pyplot.boxplot(evolution_times, labels = layer_sizes)``.
    You can also use the convenience function :func:`plot_benchmark_results`.

    Arguments:
        prepare_fn (Callable): A callable which prepares a benchmark. Signature: ``def prepare_fn(num_batches: int, num_timesteps: int, layer_size: int) -> object``
        create_fn (Callable): A callable which creates a layer. Signature: ``def create_fn(bench_obj: object) -> None``
        evolve_fn (Callable): A callable which evolves a layer. Signature: ``def evolve_fn(bench_obj: object) -> None``
        benchmark_desc (str): A description of the benchmark, which will be returned
        layer_sizes (List[int]): A list of layer sizes which should be benchmarked. Default: ``[1, 2, 5, 10, 20, 50, 100, 500, 1000, 2000]``
        num_batches (int): The number of batches to test in evolution. Default: ``10``
        num_timesteps (int): The number of timesteps to test in evolution. Default: ``1000``

    Returns:
        (creation_times, evolution_times, layer_sizes)
    """
    creation_times = []
    evolution_times = []

    # - Perform a benchmark for each layer size
    for l_size in tqdm(layer_sizes):
        try:
            # - Prepare benchmark
            bench_obj = prepare_fn(num_batches, num_timesteps, l_size)

            # - Benchmark layer creation
            creation_times.append(timeit(lambda: create_fn(bench_obj)))

            # - Benchmark evolution
            evolution_times.append(timeit(lambda: evolve_fn(bench_obj)))

        except Exception as e:
            # - Fail nicely with a warning if a benchmark dies
            warnings.warn(
                f"Benchmarking for layer size {l_size} failed with error {str(e)}."
            )

            # - No results for this run
            creation_times.append([])
            evolution_times.append([])

    # - Build a description of the benchmark
    benchmark_desc = f"{benchmark_desc}; " if benchmark_desc is not None else ""
    benchmark_desc = f"{benchmark_desc}B = {num_batches}, T = {num_timesteps}"

    # - Return benchmark results
    return creation_times, evolution_times, layer_sizes, benchmark_desc


def plot_benchmark_results(
    creation_times: List,
    evolution_times: List,
    layer_sizes: List[int],
    benchmark_title: str,
    axes: Optional[List] = None,
) -> None:
    """
    Convenience function to plot benchmark results

    Arguments:
        creation_times (List): Benchmark layer creation times
        evolution_times (List): Benchmark layer evolution times
        layer_sizes (List[int]): Layer sizes corresponding to the benchmark lists
        axes (Optional[List]): Plot into the specified axes, if provided, otherwise create new axes
    """
    import matplotlib.pyplot as plt

    ax = plt.axes() if axes is None else axes[0]
    ax.boxplot(creation_times, labels=layer_sizes)
    ax.set_xlabel("Num. LIF neurons")
    ax.set_ylabel("Creation time (s)")
    ax.set_yscale("log")
    ax.set_ylim([1e-5, 1e1])
    ax.yaxis.grid(linestyle="--")
    ax.set_title(benchmark_title)

    ax = plt.axes() if axes is None else axes[1]
    ax.boxplot(evolution_times, labels=layer_sizes)
    ax.set_xlabel("Num. LIF neurons")
    ax.set_ylabel("Evolution time (s)")
    ax.set_yscale("log")
    ax.set_ylim([1e-4, 1e2])
    ax.yaxis.grid(linestyle="--")
    ax.set_title(benchmark_title)
