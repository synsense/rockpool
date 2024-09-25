def test_imports():
    import pytest

    pytest.importorskip("samna")
    from rockpool.devices.xylo.syns65302 import (
        save_config,
        load_config,
        XyloSamna,
        config_from_specification,
    )
    import rockpool.devices.xylo.syns65302.xa3_devkit_utils as putils

def test_acceleratedtime_samna():
    import pytest
    samna = pytest.importorskip("samna")

    # Open the device and connect the source and sink nodes so we can communicate with Xylo.
    board = samna.device.open_device("XyloAudio3TestBoard")
    xylo = board.get_model()
    source = samna.graph.source_to(xylo.get_sink_node())

    power_monitor = board.get_power_monitor()
    sink_pm = samna.graph.sink_from(power_monitor.get_source_node())
    stopwatch = board.get_stop_watch()

    # Start the stopwatch to enable time-stamped power sampling.
    stopwatch.start()

    # We are only interested in Readout events, so we make a filter graph to filter only these events.
    # Important note: `graph` needs to be kept alive for the filter to work.
    graph = samna.graph.EventFilterGraph() 
    _, etf, readout_sink = graph.sequential([xylo.get_source_node(), "XyloAudio3OutputEventTypeFilter", samna.graph.JitSink()])
    etf.set_desired_type('xyloAudio3::event::Readout')

    # Create a basic network with a simple one-to-one mapping of input to output neurons.
    xylo_config = samna.xyloAudio3.configuration.XyloConfiguration()
    input_count = 3
    hidden_count = input_count
    output_count = input_count
    xylo_config.input.weights = [[127, 0, 0], [0, 127, 0], [0, 0, 127]] # shape(input_count, hidden_count)
    xylo_config.hidden.weights = [[0, 0, 0], [0, 0, 0], [0, 0, 0]] # shape(hidden_count, hidden_count)
    xylo_config.hidden.neurons = [samna.xyloAudio3.configuration.HiddenNeuron(threshold=127, v_mem_decay=1, i_syn_decay=1)] * hidden_count
    xylo_config.readout.weights = [[127, 0, 0], [0, 127, 0], [0, 0, 127]] # shape(hidden_count, output_count)
    xylo_config.readout.neurons = [samna.xyloAudio3.configuration.OutputNeuron(threshold=127, v_mem_decay=1, i_syn_decay=1)] * output_count

    # Configure Xylo in Accelerated-Time mode and to work with input spike events.
    xylo_config.operation_mode = samna.xyloAudio3.OperationMode.AcceleratedTime
    xylo_config.input_source = samna.xyloAudio3.InputSource.SpikeEvents


    # Send the configuration to Xylo
    xylo.apply_configuration(xylo_config)

    def read_register(addr):
        events = sink.get_n_events(1, 3000)
        assert(len(events) == 1)
        return events[0].data

    def get_current_timestep():
        """Utility function to obtain the current timestep.

        The `Readout` event always contains the current timestep so we could simply save it.
        But in case you forgot what timestep was last processed, you can obtain it as follows.
        """
        source.write([samna.xyloAudio3.event.TriggerReadout()])
        evts = readout_sink.get_n_events(1, timeout=3000)
        assert(len(evts) == 1)
        return evts[0].timestep


    def evolve(input):
        """Continue to evolve the model with the given input.
        
        Args:
            input (list[list[int]]): Per timestep a list of integers to specify the number of spikes to send to that `neuron_id`.
                                    Max number of spikes per timestep for a neuron is 15. All lists must have the same length.

        Returns:
            readouts (list[Readout]): The `Readout` events for the given timesteps.
        """
        timestep_count = len(input)
        if not timestep_count:
            return []

        power_monitor.start_auto_power_measurement(100) # 100Hz

        start_timestep = get_current_timestep() + 1
        final_timestep = start_timestep + timestep_count - 1
        print("** To be processed timesteps: ", [timestep for timestep in range(start_timestep, final_timestep + 1)])
        
        input_events_list = []
        for i, spike_counts in enumerate(input):
            timestep = start_timestep + i
            for neuron_id, count in enumerate(spike_counts):
                spikes = [samna.xyloAudio3.event.Spike(neuron_id=neuron_id, timestep=timestep)] * count
                input_events_list.extend(spikes)

        input_events_list.append(samna.xyloAudio3.event.TriggerProcessing(target_timestep = final_timestep + 1))
        source.write(input_events_list)

        events = readout_sink.get_n_events(timestep_count, timeout=3000)
        assert(len(events) == timestep_count)

        print("waiting for power events")
        power_events = sink_pm.get_n_events(900, timeout=4000)
        power_monitor.stop_auto_power_measurement()

        counts = [0,0,0]
        sums = [0,0,0]
        avgs = [0,0,0]
        
        for e in power_events:
            sums[e.channel] += e.value
            counts[e.channel] += 1
        
        idx = 0
        for sum, count in zip(sums, counts):
            avgs[idx] = sum/count * 1000
            idx += 1
        
        print(avgs, " in mW")


        return events


    # Now we are ready to send spikes to our network and see how the SNN core reacts.
    # Start the filter graph, otherwise the events aren't propagated.
    graph.start()

    import numpy as np
    import matplotlib.pyplot as plt

    input_data = np.load("/home/vleite/Downloads/rare_sound_3_second_samples_afe.npy", allow_pickle=True).item()['data'][7]

    # Send spikes to trigger output neuron 0
    readouts = evolve(input_data)

    print(readouts)

    # We have finished processing, so we can stop the graph.
    graph.stop()

