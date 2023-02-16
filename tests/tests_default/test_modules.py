import pytest


def test_imports():
    from rockpool.nn.modules.module import Module


def test_module():
    from rockpool.nn.modules.module import Module
    from rockpool.parameters import Parameter, State, SimulationParameter
    import numpy as np

    class my_module(Module):
        def __init__(self, shape: tuple, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)

            self.tau = Parameter(
                shape=self.shape[-1],
                family="taus",
                init_func=lambda s: np.ones(s) * 100e-3,
            )
            self.bias = Parameter(
                shape=self.shape[-1], family="bias", init_func=np.zeros
            )
            self.v_mem = State(shape=self.shape[-1], init_func=np.zeros)
            self.dt = SimulationParameter(1e-3)

        def evolve(self, input, record: bool = False):
            if record:
                record_dict = {
                    "v_mem": [],
                    "rec_input": [],
                    "input": input,
                }

            for input_t in input:
                self.v_mem += (input_t + self.bias) / self.tau * self.dt

                if record:
                    record_dict["v_mem"].append(self.v_mem)

            if record:
                record_dict["v_mem"] = np.array(record_dict["v_mem"])
                record_dict["rec_input"] = np.array(record_dict["rec_input"])
            else:
                record_dict = {}

            return np.clip(self.v_mem, 0, np.inf), {"v_mem": self.v_mem}, record_dict

    # - Test instantiation
    N = 5
    mod = my_module((N,))

    # - Test evolution
    input = np.random.rand(10, N)
    mod.evolve(input)
    output, new_state, rec = mod(input)

    # - Test setting attributes
    mod.set_attributes(new_state)

    # - Test recording
    output, new_state, rec = mod(input, record=True)
    print(rec)


def test_submodules():
    from rockpool.nn.modules.module import Module
    from rockpool.parameters import Parameter, SimulationParameter, State
    import numpy as np

    class IAF(Module):
        def __init__(
            self,
            dt: float = 1e-3,
            tau_syn=None,
            tau_mem=None,
            bias=None,
            shape=None,
            *args,
            **kwargs,
        ):
            # - Work out the shape of this module
            if shape is None:
                assert (
                    tau_syn is not None
                ), "You must provide either `shape` or else specify parameters."
                shape = tau_syn.shape

            # - Call the superclass initialiser
            super().__init__(shape, *args, **kwargs)

            # - Provide defaults for parameters
            tau_syn = np.zeros(self.shape) if tau_syn is None else tau_syn
            tau_mem = np.zeros(self.shape) if tau_mem is None else tau_mem
            bias = np.empty(self.shape) if bias is None else bias

            self.tau_syn = Parameter(tau_syn, "taus")
            self.tau_mem = Parameter(tau_mem, "taus")
            self.bias = Parameter(bias, "bias")

            self.dt = SimulationParameter(dt)

            self.v_mem = State(np.zeros(self.shape))
            self.i_syn = State(np.zeros(self.shape))

        def evolve(self, input, record: bool = False):
            new_state = {
                "v_mem": self.v_mem,
                "i_syn": self.i_syn,
            }

            return (
                input,
                new_state,
                {},
            )

    # - Instantiate an IAF module
    iaf = IAF(shape=(2, 3))
    print("IAF state:", iaf.state())

    p = iaf.parameters()
    p["tau_syn"] = np.random.rand(2, 3)

    # - Set parameters
    iaf.set_attributes(p)

    # - Define a module with nested submodules
    class my_ffwd_net(Module):
        def __init__(self, shape, *args, **kwargs):
            super().__init__(shape, *args, **kwargs)

            for index, (N_in, N_out) in enumerate(zip(self.shape[:-1], self.shape[1:])):
                setattr(
                    self,
                    f"weight_{index}",
                    Parameter(np.random.rand(N_in, N_out), "weights"),
                )
                setattr(self, f"iaf_{index}", IAF(shape=(N_in, N_out)))

        def evolve(self, input, record: bool = False):
            new_state = {}
            record_dict = {}
            for layer in range(len(self._shape) - 1):
                w = getattr(self, f"weight_{layer}")
                mod_name = f"iaf_{layer}"
                iaf = getattr(self, mod_name)

                input, substate, subrec = iaf(np.dot(input, w), record)
                new_state.update({mod_name: substate})
                record_dict.update({mod_name: subrec})

            return input, new_state, record_dict

    net = my_ffwd_net([2, 3, 2])
    print("Repr:", net)

    print("Evolve as call:", net(4))

    print('All "weights" family:', net.parameters("weights"))

    # - Test an operation on collected weights
    np.sum([np.sum(v**2) for v in net.parameters("weights").values()])
