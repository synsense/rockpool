from samna.pollen.configuration import PollenConfiguration
from cimulator.pollen import Synapse, PollenLayer
import numpy as np
from rockpool.nn.modules.timed_module import TimedModule
from rockpool.timeseries import TSEvent
from rockpool.parameters import Parameter, State, SimulationParameter

class Cimulator(TimedModule):
    def __init__(self,
                 config: PollenConfiguration,
                 dt: float = 0.001):
        """
        Cimulator based layer to simulate the Pollen hardware.

        Parameters:
        dt: float
            Timestep for simulation in range 0.001 to 0.01 seconds
        config: PollenConfiguration
            ``samna.pollen.Configuration`` object to specify all parameters. See samna documentation for details.

        """
        super().__init__(dt=dt)

        self.config = config
 
        self.synapses_in = []
        for pre, w_pre in enumerate(config.input_expansion.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))

            if config.synapse2_enable:
                w2_pre = config.input_expansion.syn2_weights[pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(Synapse(post, 1, w2_pre[post]))

            self.synapses_in.append(tmp)

        self.synapses_rec = []
        for pre, w_pre in enumerate(config.reservoir.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))
            
            if config.synapse2_enable:
                w2_pre = config.reservoir.syn2_weights[pre]
                for post in np.where(w2_pre)[0]:
                    tmp.append(Synapse(post, 1, w2_pre[post]))

            self.synapses_rec.append(tmp)

        self.synapses_out = []
        for pre, w_pre in enumerate(config.readout.weights):
            tmp = []
            for post in np.where(w_pre)[0]:
                tmp.append(Synapse(post, 0, w_pre[post]))
            self.synapses_out.append(tmp)

        self.threshold = []
        self.dash_syn = []
        self.dash_mem = []
        self.aliases = []

        for neuron in config.reservoir.neurons:
            if neuron.alias_target:
                self.aliases.append([neuron.alias_target])
            else:
                self.aliases.append([])
            self.threshold.append(neuron.threshold)
            self.dash_mem.append(neuron.v_mem_decay)
            self.dash_syn.append([neuron.i_syn_decay, neuron.i_syn2_decay])

        self.threshold_out = []
        self.dash_syn_out = []
        self.dash_mem_out = []

        for neuron in config.readout.neurons:
            self.threshold_out.append(neuron.threshold)
            self.dash_mem_out.append(neuron.v_mem_decay)
            self.dash_syn_out.append([neuron.i_syn_decay])

        self.weight_shift_inp = config.input_expansion.weight_bit_shift
        self.weight_shift_rec = config.reservoir.weight_bit_shift
        self.weight_shift_out = config.readout.weight_bit_shift
        
        self.pollen_layer = PollenLayer(synapses_in=self.synapses_in,
                                        synapses_rec=self.synapses_rec,
                                        synapses_out=self.synapses_out,
                                        aliases=self.aliases,
                                        threshold=self.threshold,
                                        threshold_out=self.threshold_out,
                                        weight_shift_inp=self.weight_shift_inp,
                                        weight_shift_rec=self.weight_shift_rec,
                                        weight_shift_out=self.weight_shift_out,
                                        dash_mem=self.dash_mem,
                                        dash_mem_out=self.dash_mem_out,
                                        dash_syns=self.dash_syn,
                                        dash_syns_out=self.dash_syn_out,
                                        name="my cim layer")


        self.recordings = {"vmem": [], 
                           "isyn": [], 
                           "isyn2": [], 
                           "spikes": [], 
                           "vmem_out": [],
                           "isyn_out": [], 
                           }

    def evolve(self,
               ts_input=None,
               duration=None,
               num_timesteps=None,
               kwargs_timeseries=None,
               record: bool = True,
               *args,
               **kwargs,):

        inp = ts_input.raster(dt=self.dt, add_events=True)
        out = self.pollen_layer.evolve(inp)
        ts_out = TSEvent.from_raster(out, dt=self.dt, t_start=ts_input.t_start)

        self.recordings['vmem'] = self.pollen_layer.rec_v_mem 
        self.recordings['isyn'] = self.pollen_layer.rec_i_syn
        self.recordings['isyn2'] = self.pollen_layer.rec_i_syn2
        self.recordings['spikes'] = self.pollen_layer.rec_recurrent_spikes
        self.recordings['vmem_out'] = self.pollen_layer.rec_v_mem_out
        self.recordings['isyn_out'] = self.pollen_layer.rec_i_syn_out


        return ts_out, {}, self.recordings


    def reset_all(self):
        self.pollen_layer.reset_all()




