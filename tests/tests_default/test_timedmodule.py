def test_imports():
    from rockpool.nn.modules.timed_module import TimedModule, TimedModuleWrapper


def test_TimedModule():
    from rockpool.nn.modules.timed_module import TimedModule
    from rockpool.timeseries import TimeSeries, TSContinuous, TSEvent
    from rockpool.parameters import State, Parameter, SimulationParameter

    from typing import Any

    import numpy as np

    class test_timedmod(TimedModule):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape=shape, *args, **kwargs)

            self.activation = State(shape=self.size_out, init_func=np.zeros)

        def evolve(
            self,
            ts_input: TimeSeries = None,
            duration: float = None,
            num_timesteps: int = None,
            kwargs_timeseries: Any = None,
            record: bool = False,
            *args,
            **kwargs,
        ):
            # - Rasterise input
            time_base, input_raster, num_timesteps = self._prepare_input(
                ts_input, duration, num_timesteps
            )

            output = input_raster
            new_state = {"activation": self.activation}
            record_dict = {}

            # - Return and wrap outputs
            return (
                self._gen_timeseries(output),
                new_state,
                record_dict,
            )

    N = 100
    dt = 1e-3

    mod = test_timedmod(N, dt=dt)
    print(mod)

    # - Test evolution
    input_ts = TSContinuous.from_clocked(
        np.random.rand(1000, N), dt, periodic=True, name="Input signal"
    )

    # - Evolve by input TS
    output_ts, ns, _ = mod(input_ts)
    mod.set_attributes(ns)

    print(output_ts)

    # - Evolve by duration
    output_ts, ns, _ = mod(duration=2.0)

    # - Evolve by time steps
    output_ts, ns, _ = mod(num_timesteps=25)


def test_submodules():
    from rockpool.nn.modules.timed_module import TimedModule
    from rockpool.timeseries import TimeSeries, TSContinuous, TSEvent
    from rockpool.parameters import State, Parameter, SimulationParameter

    from typing import Any

    import numpy as np

    class simplemod(TimedModule):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

            self.param1 = Parameter(shape=self.size_out, init_func=np.zeros)
            self.param2 = Parameter(shape=(2, self.size_out), init_func=np.zeros)
            self.state1 = State(shape=self.size_out, init_func=np.zeros)

        def evolve(
            self,
            ts_input: TimeSeries = None,
            duration: float = None,
            num_timesteps: int = None,
            kwargs_timeseries: Any = None,
            record: bool = False,
            *args,
            **kwargs,
        ):
            # - Rasterise input
            time_base, input_raster, num_timesteps = self._prepare_input(
                ts_input, duration, num_timesteps
            )

            output = input_raster
            new_state = {"state1": self.state1}
            record_dict = {}

            # - Return and wrap outputs
            return (
                self._gen_timeseries(output),
                new_state,
                record_dict,
            )

    class net_mod(TimedModule):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape=shape, *args, **kwargs)

            self.submod1 = simplemod(dt=self.dt, shape=self.size_in)
            self.submod2 = simplemod(dt=self.dt * 2, shape=self.size_out)

        def evolve(
            self,
            ts_input: TimeSeries = None,
            duration: float = None,
            num_timesteps: int = None,
            kwargs_timeseries: Any = None,
            record: bool = False,
            *args,
            **kwargs,
        ):
            new_state = {}
            record_dict = {}

            ts_data, substate, subrec = self.submod1(
                ts_input, duration, num_timesteps, record=record
            )
            new_state.update(substate)
            if record:
                record_dict.update(subrec)
                record_dict.update({"submod1_output": ts_data})

            ts_data, substate, subrec = self.submod2(
                ts_data, duration, num_timesteps, record=record
            )
            new_state.update(substate)
            if record:
                record_dict.update(subrec)

            return ts_data, new_state, record_dict

    N_in = 20
    N_out = 40
    dt = 1e-3
    mod = net_mod((N_in, N_out), dt=dt)

    print(mod)

    # - Test evolution
    input_ts = TSContinuous.from_clocked(
        np.random.rand(1000, N_in), dt, periodic=True, name="Input signal"
    )

    # - Evolve by input TS
    output_ts, ns, _ = mod(input_ts, record=True)
    mod.set_attributes(ns)

    print(output_ts)

    # - Evolve by duration
    output_ts, ns, _ = mod(duration=2.0)

    # - Evolve by time steps
    output_ts, ns, _ = mod(num_timesteps=25)


def test_wrapper():
    from rockpool.nn.modules.timed_module import TimedModuleWrapper
    from rockpool.nn.modules.jax.jax_module import JaxModule
    from rockpool.nn.modules.jax.rate_jax import RateEulerJax
    from rockpool.parameters import Parameter, SimulationParameter, State
    from rockpool.timeseries import TSContinuous

    import jax.numpy as jnp
    import numpy as np

    class net_mod(JaxModule):
        def __init__(
            self, shape, dt: float = 1e-3,
        ):
            super().__init__(shape=shape)

            self.dt = SimulationParameter(dt)
            self.weight = Parameter(
                shape=shape, init_func=np.random.standard_normal, family="weights"
            )
            self.relu = RateEulerJax(self.size_out)

        def evolve(
            self, input_data, record: bool = False,
        ):
            return self.relu(jnp.dot(input_data, self.weight), record=record)

    N_in = 10
    N_out = 20
    mod = net_mod((N_in, N_out))
    tnm = TimedModuleWrapper(mod)

    # - Test raw module
    T = 1000
    input_data = np.random.rand(T, N_in)

    output_data, new_state, record_dict = mod(input_data)

    output_ts, new_state, record_dict = tnm(
        TSContinuous.from_clocked(input_data, dt=tnm.dt)
    )
    tnm.set_attributes(new_state)


def test_v1_conversion():
    from rockpool.nn.layers import Layer
    from rockpool.nn.modules.timed_module import astimedmodule
    import numpy as np

    # - Define a minimal wrapped layer
    @astimedmodule(parameters=["weights"])
    class TestLayer(Layer):
        def __init__(self, weights, *args, **kwargs):
            super().__init__(weights, *args, **kwargs)

        def evolve(self, *args, **kwargs):
            super().evolve(*args, **kwargs)

        def to_dict(self):
            return super().to_dict()

    # - Test constructing a layer
    Nin = 2
    Nout = 2
    tmod = TestLayer(np.zeros((Nin, Nout)))

    # - Test that we got the correct attributes
    assert hasattr(tmod, "dt")
    assert hasattr(tmod, "t")
    assert hasattr(tmod, "weights")

    # - Test assigning weights directly
    w = np.array([[1, 2], [3, 4]])
    tmod.weights = w
    assert np.all(tmod.weights == w)
    assert np.all(tmod._module.weights == w)

    # - Test assigning weights through .set_attributes()
    w2 = np.array([[0, 1], [1, 0]])
    p = tmod.parameters()
    p["weights"] = w2
    tmod.set_attributes(p)
    assert np.all(tmod.weights == w2)
    assert np.all(tmod._module.weights == w2)
