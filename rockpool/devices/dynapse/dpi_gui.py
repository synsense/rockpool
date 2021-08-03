"""
Graphic User Interface to help finding the best parameter set.
One can create a custom or a random input pattern
Then the effect on DPI circuitry can be observed.  

Note : The package needs an notebook environment to run.

Example:
    ```
    from dpi_gui import ResponseGUI
    res = ResponseGUI((2,), "input", "response")
    res.display_widgets()
    ```

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
02/08/2021
"""

import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

from dataclasses import dataclass
from IPython.display import display
from rockpool import TSEvent, TSContinuous
from rockpool.nn.modules import TimedModuleWrapper

from dynapse1_neuron_synapse_jax import (
    DynapSE1NeuronSynapseJax,
    DynapSE1Layout,
    DPIParameters,
    DynapSE1Parameters,
)

from utils import spike_to_pulse, custom_spike_train, random_spike_train

from typing import (
    Optional,
)

plt.rcParams["figure.figsize"] = [12, 4]
plt.rcParams["figure.dpi"] = 300


@dataclass
class SpikeParams:
    base: str = "0,1,0,0,0,0,0,0,0,0"
    kernel: str = "1,0,0,0,0,0"
    duration: float = 1e-1
    dt: float = 1e-6
    pulse_width: float = 1e-5
    rate: float = 40
    amplitude: float = 1.0


@dataclass
class DPIParams:
    Capacitor: float = 1e-9
    Itau: float = 1e-9
    Ith: float = 1e-9
    Iw: float = 1e-6
    Io: float = 5e-12


class ResponseGUI:
    def __init__(
        self,
        shape: tuple,
        spike_params: Optional[SpikeParams] = None,
        dpi_params: Optional[DPIParams] = None,
        input_tab: Optional[str] = "Input Spikes",
        response_tab: Optional[str] = "DPI Response",
    ):
        self.shape = shape
        self.input_tab = input_tab
        self.response_tab = response_tab

        if spike_params is None:
            spike_params = SpikeParams()

        if dpi_params is None:
            dpi_params = DPIParams()

        self._create_input_widgets(spike_params)
        self._create_response_widgets(dpi_params)
        self._create_gui()

    def _create_input_widgets(self, params):

        self.select_type = widgets.ToggleButtons(
            options=["custom", "random"],
            description="Spike Gen.:",
            value="custom",
        )
        self.select_type.observe(self._on_toggle_changed, "value")

        self.base = widgets.Text(
            value=params.base,
            description="Carrier:",
            disabled=False,
        )

        self.kernel = widgets.Text(
            value=params.kernel,
            description="Kernel:",
            disabled=False,
        )

        self.duration = widgets.FloatLogSlider(
            min=-6,
            max=3,
            value=params.duration,
            description="Duration (s):",
            disabled=False,
            readout_format=".1e",
        )

        self.dt = widgets.FloatLogSlider(
            min=-6,
            max=-1,
            value=params.dt,
            description="dt (s):",
            disabled=False,
            readout_format=".1e",
        )

        self.pulse_width = widgets.FloatLogSlider(
            min=-5,
            max=-1,
            value=params.pulse_width,
            description="PW (s):",
            disabled=False,
            readout_format=".1e",
        )

        self.rate = widgets.FloatSlider(
            min=0,
            max=1000,
            value=params.rate,
            description="Rate (Hz):",
            disabled=False,
        )

        self.amplitude = widgets.FloatSlider(
            min=0,
            max=5,
            value=params.amplitude,
            description="Amp. (V):",
            disabled=False,
        )

        self.name = widgets.Text(value="Input", description="Name:", disabled=False)

        self.hit = widgets.Button(description="Hit Me!", disabled=True)
        self.hit.on_click(self._on_hit_clicked)

        self.interact = widgets.interactive(
            self.plot_custom_spike_train,
            base=self.base,
            kernel=self.kernel,
            duration=self.duration,
            pulse_width=self.pulse_width,
            amplitude=self.amplitude,
            dt=self.dt,
            name=self.name,
        )

        self.input_out = widgets.Output()

    def _create_response_widgets(self, params):
        self.Capacitor = widgets.FloatLogSlider(
            min=-12,
            max=-3,
            value=params.Capacitor,
            description="Capacitor (F):",
            disabled=False,
            readout_format=".1e",
        )

        self.Itau = widgets.FloatLogSlider(
            min=-12,
            max=-3,
            value=params.Itau,
            description="Itau (A):",
            disabled=False,
            readout_format=".1e",
        )

        self.Ith = widgets.FloatLogSlider(
            min=-12,
            max=-3,
            value=params.Ith,
            description="Ith (A):",
            disabled=False,
            readout_format=".1e",
        )

        self.Iw = widgets.FloatLogSlider(
            min=-12,
            max=-3,
            value=params.Iw,
            description="Iw (A):",
            disabled=False,
            readout_format=".1e",
        )

        self.Io = widgets.FloatLogSlider(
            min=-13,
            max=-6,
            value=params.Io,
            description="Io (A):",
            disabled=False,
            readout_format=".1e",
        )

        self.run = widgets.Button(description="Run")
        self.run.on_click(self._on_run_clicked)

        self.response_out = widgets.Output()

    def plot_input(self, st, dt, pulse_width, amplitude, name):
        plt.figure()
        st.plot()

        self.vin = spike_to_pulse(
            input_spike=st,
            dt=dt,
            pulse_width=pulse_width,
            amplitude=amplitude,
            name=name,
        )

        plt.figure()
        self.vin.plot(stagger=self.vin.max * 1.2)
        plt.show()

    def plot_custom_spike_train(
        self, base, kernel, duration, dt, pulse_width, amplitude, name
    ):
        try:
            base = list(map(int, base.split(",")))
            kernel = list(map(int, kernel.split(",")))
        except:
            None

        if isinstance(base, list) and isinstance(kernel, list):
            steps = int(np.round(duration / dt))
            len_base = len(base)
            kernel_size = int(np.round(steps / len_base))
            len_kernel = len(kernel)
            if len_kernel < kernel_size:
                extend = kernel_size - len_kernel
                kernel = kernel + [0] * extend

            self.st = custom_spike_train(
                channels=self.shape[0], base=base, kernel=kernel, dt=dt, name=name
            )
            self.plot_input(self.st, dt, pulse_width, amplitude, name)

        else:
            print("Base and Kernel is not given properly!")

    def plot_random_spike_train(self, duration, dt, pulse_width, amplitude, rate, name):

        try:
            self.st = random_spike_train(
                duration=duration, channels=self.shape[0], rate=rate, dt=dt, name=name
            )
            self.plot_input(self.st, dt, pulse_width, amplitude, name)

        except ValueError as e:
            print(e)

        except:
            print("Fatal error occured! Comment out the try except block!")

    def plot_synaptic_response(self, rd):
        input_data = TSContinuous.from_clocked(
            rd["input_data"], dt=self.dt.value, name="$V_{in}$"
        )

        I_syn = TSContinuous.from_clocked(
            rd["Inmda"], dt=self.dt.value, name="$I_{syn}$"
        )

        plt.figure()
        input_data.plot(stagger=input_data.max * 1.2)

        plt.figure()
        I_syn.plot(stagger=I_syn.max * 1.2)
        plt.show()

    def _on_hit_clicked(self, button):
        self.interact.update()

    def _on_run_clicked(self, button):
        self.response_out.clear_output()
        with self.response_out:

            # Create a new DynapSE1NeuronSynapseJax Instance
            layout = DynapSE1Layout(Cnmda=self.Capacitor.value, Io=self.Io.value)
            nmda = DPIParameters(
                Itau=self.Itau.value, Ith=self.Ith.value, Iw=self.Iw.value
            )
            params = DynapSE1Parameters(ahp=nmda, nmda=nmda)

            self.se1 = TimedModuleWrapper(
                DynapSE1NeuronSynapseJax(
                    shape=self.shape,
                    out_rate=0.0002,
                    dt=self.dt.value,
                    params=params,
                    layout=layout,
                    spiking_input=False,
                )
            )

            self.se1.reset_state()

            # - Evolve with the spiking input
            tsOutput, new_state, record_dict = self.se1(
                self.vin.start_at(self.se1.t), record=True
            )

            self.plot_synaptic_response(record_dict)

    def _on_toggle_changed(self, change):

        if change["new"] == "random":
            self.base.disabled = True
            self.kernel.disabled = True
            self.hit.disabled = False

            self.interact = widgets.interactive(
                self.plot_random_spike_train,
                duration=self.duration,
                pulse_width=self.pulse_width,
                amplitude=self.amplitude,
                rate=self.rate,
                dt=self.dt,
                name=self.name,
            )

        if change["new"] == "custom":
            self.base.disabled = False
            self.kernel.disabled = False
            self.hit.disabled = True

            self.interact = widgets.interactive(
                self.plot_custom_spike_train,
                base=self.base,
                kernel=self.kernel,
                duration=self.duration,
                pulse_width=self.pulse_width,
                amplitude=self.amplitude,
                dt=self.dt,
                name=self.name,
            )

        self.gui.close()
        self._create_gui()
        self.display()

    def _create_gui(self):

        # INPUT
        select = widgets.HBox([self.select_type, self.hit])
        tab_input = widgets.VBox([select, self.interact, self.input_out])

        # RESPONSE
        tab_response = widgets.VBox(
            [
                self.Capacitor,
                self.Itau,
                self.Ith,
                self.Iw,
                self.Io,
                self.run,
                self.response_out,
            ]
        )

        # GUI
        tab = widgets.Tab(children=[tab_input, tab_response])
        tab.set_title(0, self.input_tab)
        tab.set_title(1, self.response_tab)
        self.gui = widgets.VBox(children=[tab])

    def display(self):
        display(self.gui)
