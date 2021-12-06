"""
Graphic User Interface to help finding the best parameter set.
One can create a custom or a random input pattern
Then the effect on DPI circuitry can be observed.  

Note : The package needs an notebook environment to run.

Example:
    ```
    from dpi_gui import ResponseGUI
    res = ResponseGUI((2,))
    res.display_widgets()
    ```

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
02/08/2021
"""
import time
import numpy as np
import ipywidgets as widgets
import matplotlib.pyplot as plt

from dataclasses import dataclass
from IPython.display import display
from rockpool import TSEvent, TSContinuous
from rockpool.nn.modules import TimedModuleWrapper

from rockpool.devices.dynapse.adexpif_jax import (
    DynapSEAdExpIFJax,
)

from rockpool.devices.dynapse.dynapse1_simconfig import (
    DynapSE1Layout,
    DynapSE1SimCore,
    SynapseParameters,
)

from rockpool.devices.dynapse.utils import (
    spike_to_pulse,
    custom_spike_train,
    random_spike_train,
    get_tau,
    Isyn_inf,
)

from typing import Union, Tuple, Optional

plt.rcParams["figure.figsize"] = [12, 4]
plt.rcParams["figure.dpi"] = 300


@dataclass
class SpikeParams:
    times: str = "2e-3,3e-2,5e-2,7e-2"
    channels: Optional[str] = None
    duration: float = 1e-1
    dt: float = 1e-6
    pulse_width: float = 1e-5
    rate: float = 40
    amplitude: float = 1.0


@dataclass
class DPIParams:
    kappa_n: float = 0.75
    kappa_p: float = 0.66
    Ut: float = 25e-3
    Io: float = 5e-13
    Capacitor: float = 28e-12
    Itau: float = 10e-12
    Ith: float = 40e-12
    Iw: float = 1e-7


class ResponseGUI:
    def __init__(
        self,
        shape: tuple,
        spike_params: Optional[SpikeParams] = None,
        dpi_params: Optional[DPIParams] = None,
        input_tab: Optional[str] = "Input Spikes",
        response_tab: Optional[str] = "DPI Response",
        spike_in: Union[TSEvent, TSContinuous] = None,
    ):
        self.shape = shape
        self.input_tab = input_tab
        self.response_tab = response_tab
        self.spike_in = spike_in

        if spike_params is None:
            spike_params = SpikeParams()

        if dpi_params is None:
            dpi_params = DPIParams()

        self.init_spike_params = spike_params
        self.init_dpi_params = dpi_params

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

        times, channels = self._check_times_channels(
            params.times, params.channels, self.shape[0]
        )
        self.times = widgets.Text(
            value=times,
            description="Times :",
            disabled=False,
        )

        self.channels = widgets.Text(
            value=channels,
            description="Channels:",
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
            description="$t_{pulse}$ (s):",
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
            description="$V_{DD}$ (V):",
            disabled=False,
        )

        self.name = widgets.Text(value="Input", description="Name:", disabled=False)

        self.hit = widgets.Button(description="Hit Me!", disabled=True)
        self.hit.on_click(self._on_hit_clicked)

        self.interact = widgets.interactive(
            self.plot_custom_spike_train,
            times=self.times,
            channels=self.channels,
            duration=self.duration,
            pulse_width=self.pulse_width,
            amplitude=self.amplitude,
            dt=self.dt,
            name=self.name,
        )

        self.input_out = widgets.Output()

    def _check_times_channels(
        self,
        times: str,
        channels: Optional[str],
        num_channels: int,
    ) -> Tuple[str, str]:

        if num_channels > 1:
            channels_arr = (
                list(map(int, channels.split(",")))
                if channels is not None
                else np.random.randint(
                    low=0,
                    high=num_channels,
                    size=len(times.split(",")),
                    dtype=int,
                )
            )
            channels = ",".join([str(item) for item in channels_arr])
        return times, channels

    def _create_response_widgets(self, params):

        self.Ut = widgets.FloatText(
            step=1e-3,
            value=params.Ut,
            description="$U_{T}$ (V):",
            disabled=False,
            readout_format=".1e",
            layout=widgets.Layout(width="20%"),
        )

        self.kappa_n = widgets.FloatText(
            step=1e-2,
            value=params.kappa_n,
            description="$\\kappa_{n}$:",
            disabled=False,
            readout_format=".1e",
            layout=widgets.Layout(width="20%"),
        )

        self.kappa_p = widgets.FloatText(
            step=1e-2,
            value=params.kappa_p,
            description="$\\kappa_{p}$:",
            disabled=False,
            readout_format=".1e",
            layout=widgets.Layout(width="20%"),
        )

        self.Io = widgets.FloatText(
            step=1e-13,
            value=params.Io,
            description="$I_{o}$ (A):",
            disabled=False,
            readout_format=".1e",
            layout=widgets.Layout(width="20%"),
        )

        self.Capacitor = widgets.FloatLogSlider(
            min=-12,
            max=-3,
            value=params.Capacitor,
            description="$C$ (F):",
            disabled=False,
            readout_format=".1e",
        )

        self.Itau = widgets.FloatLogSlider(
            min=-12,
            max=-3,
            value=params.Itau,
            description="$I_{\\tau}$ (A):",
            disabled=False,
            readout_format=".1e",
        )

        self.Ith = widgets.FloatLogSlider(
            min=-12,
            max=-3,
            value=params.Ith,
            description="$I_{th}$ (A):",
            disabled=False,
            readout_format=".1e",
        )

        self.Iw = widgets.FloatLogSlider(
            min=-12,
            max=-3,
            value=params.Iw,
            description="$I_{w}$ (A):",
            disabled=False,
            readout_format=".1e",
        )

        self.run = widgets.Button(description="Run")
        self.run.on_click(self._on_run_clicked)
        self.default = widgets.Button(description="Set to Default")
        self.default.on_click(self._on_default_clicked)

        self.response_out = widgets.Output()

        self.response_interactive = widgets.interactive(
            self.plot_response_dependends,
            Ut=self.Ut,
            kappa_n=self.kappa_n,
            kappa_p=self.kappa_p,
            C=self.Capacitor,
            Itau=self.Itau,
            Ith=self.Ith,
            Iw=self.Iw,
        )

    def plot_response_dependends(
        self,
        Ut: float,
        kappa_n: float,
        kappa_p: float,
        C: float,
        Itau: float,
        Ith: float,
        Iw: float,
    ):
        layout = DynapSE1Layout()
        tau = get_tau(
            C,
            Itau,
            Ut,
            kappa_n,
            kappa_p,
        )
        Itau_inf = Isyn_inf(Ith, Itau, Iw)

        tau_str = "$\\tau = \\dfrac{C U_{T}}{\\kappa I_{\\tau}} =$ %.1e s\n\n" % tau
        infinity_str = (
            "$I_{syn_{\infty}} = \\dfrac{I_{th}}{I_{\\tau}} \cdot I_{w} =$ %.1e A\n\n"
            % Itau_inf
        )

        ax = plt.axes(
            [0, 0, 0, 0], xmargin=-0.04, ymargin=-0.04
        )  # left,bottom,width,height
        ax.axis("off")

        plt.text(
            0,
            0,
            tau_str + infinity_str,
            size=3.5,
        )

    def plot_input(self, st, dt, pulse_width, amplitude, name):

        plt.figure()
        st.plot()

        if dt < pulse_width:
            self.vin = spike_to_pulse(
                input_spike=st,
                dt=dt,
                pulse_width=pulse_width,
                amplitude=amplitude,
                name=name,
            )

            plt.figure()
            self.vin.plot(stagger=self.vin.max * 1.2)
        else:
            self.vin = st
        plt.show()

    def plot_custom_spike_train(
        self, times, channels, duration, dt, pulse_width, amplitude, name
    ):
        try:
            times = list(map(float, times.split(",")))
            if channels != "":
                channels = list(map(int, channels.split(",")))
            else:
                channels = None

            if len(channels) != len(times):
                print(
                    "TSEvent `Input`: `channels` must have the same number of elements as `times`, be an integer or None."
                )
                channels = None

        except:
            None

        if isinstance(times, list) and (isinstance(channels, list) or channels is None):
            self.spike_train = custom_spike_train(
                times=times,
                channels=channels,
                duration=duration,
                name=name,
            )
            if channels is not None:
                if max(channels) >= self.shape[0]:
                    print("Number of channels does not match the shape provided!!!")
                if min(channels) < 0:
                    print("Channel ID should not be negative!")

            self.plot_input(self.spike_train, dt, pulse_width, amplitude, name)

        else:
            print("Base and Kernel is not given properly <value>,<value>,<value>!")

    def plot_random_spike_train(self, duration, dt, pulse_width, amplitude, rate, name):

        try:
            self.spike_train = random_spike_train(
                duration=duration, n_channels=self.shape[0], rate=rate, dt=dt, name=name
            )
            self.plot_input(self.spike_train, dt, pulse_width, amplitude, name)

        except ValueError as e:
            print(e)

        except:
            print("Fatal error occured! Comment out the try except block!")

    def plot_synaptic_response(self, rd):
        plt.figure()
        if self.dt.value > self.pulse_width.value:
            input_data = TSEvent.from_raster(
                rd["input_data"], dt=self.dt.value, name="Input Spikes", periodic=True
            )
            input_data.plot()

        else:
            input_data = TSContinuous.from_clocked(
                rd["input_data"], dt=self.dt.value, name="$V_{in}$"
            )
            input_data.plot(stagger=input_data.max * 1.2)

        I_syn = TSContinuous.from_clocked(
            rd["Inmda"], dt=self.dt.value, name="$I_{syn}$"
        )

        I_mem = TSContinuous.from_clocked(
            rd["Imem"], dt=self.dt.value, name="$I_{mem}$"
        )

        Ispkthr = np.ones_like(rd["Iahp"]) * self.se1._module.Ispkthr
        I_spkthr = TSContinuous.from_clocked(
            Ispkthr, dt=self.dt.value, name="$I_{spkthr}$"
        )

        spikes = TSEvent.from_raster(
            rd["spikes"], dt=self.dt.value, name="Output Spikes"
        )

        plt.figure()
        I_syn.plot(stagger=I_syn.max * 1.2)

        plt.figure()
        I_mem.plot(stagger=I_mem.max * 1.2)
        I_spkthr.plot(stagger=I_mem.max * 1.2, linestyle="dashed")

        plt.figure()
        spikes.plot()

        plt.show()

    def _on_hit_clicked(self, button):
        self.interact.update()

    def _on_run_clicked(self, button):

        self.response_out.clear_output()
        tic = time.perf_counter()
        # Create a new DynapSEAdExpIFJax Instance
        layout = DynapSE1Layout(Io=self.Io.value)
        nmda = SynapseParameters(
            Itau=self.Itau.value,
            C=self.Capacitor.value,
            Iw=self.Iw.value,
        )
        sim_config = DynapSE1SimCore(layout=layout, ahp=nmda, nmda=nmda)
        spiking_input = True if self.dt.value > self.pulse_width.value else False

        self.se1 = TimedModuleWrapper(
            DynapSEAdExpIFJax(
                shape=self.shape,
                dt=self.dt.value,
                sim_config=sim_config,
                spiking_input=spiking_input,
            )
        )

        self.se1.reset_state()
        # - Evolve with the spiking input
        tsOutput, new_state, self.record_dict = self.se1(
            self.vin.start_at(self.se1.t), record=True
        )

        toc = time.perf_counter()

        with self.response_out:
            print(f"RUNTIME : {toc-tic:.2f} s")
            self.plot_synaptic_response(self.record_dict)

    def _on_default_clicked(self, button):
        self._create_response_widgets(self.init_dpi_params)
        self.gui.close()
        self._create_gui()
        self.gui.children[0].selected_index = 1
        self.display()

    def _on_toggle_changed(self, change):

        if change["new"] == "random":
            self.times.disabled = True
            self.channels.disabled = True
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
            self.times.disabled = False
            self.channels.disabled = False
            self.hit.disabled = True

            self.interact = widgets.interactive(
                self.plot_custom_spike_train,
                times=self.times,
                channels=self.channels,
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
        # tau = widgets.HBox([self.Itau, self.tau])
        lay = list(self.response_interactive.children[:3])
        lay.append(self.Io)
        layout_widgets = widgets.HBox(
            lay,
            layout=widgets.Layout(flex_flow="row wrap"),
        )
        controls = widgets.VBox(
            self.response_interactive.children[3:-1],
            layout=widgets.Layout(flex_flow="column wrap"),
        )
        output = self.response_interactive.children[-1]
        response = widgets.HBox([controls, output])
        DPIResponse = widgets.VBox([layout_widgets, response])
        buttons = widgets.HBox([self.run, self.default])
        tab_response = widgets.VBox(
            [
                DPIResponse,
                buttons,
                self.response_out,
            ]
        )
        self.response_interactive.update()

        # GUI
        tab = widgets.Tab(children=[tab_input, tab_response])
        tab.set_title(0, self.input_tab)
        tab.set_title(1, self.response_tab)
        self.gui = widgets.VBox(children=[tab])

        if self.spike_in is not None:
            self.times.value = ",".join([str(item) for item in self.spike_in.times])
            self.channels.value = ",".join(
                [str(item) for item in self.spike_in.channels]
            )
            self.duration.value = self.spike_in.duration
            self.gui.children[0].selected_index = 1
            self._on_run_clicked(None)

    def display(self):
        display(self.gui)
