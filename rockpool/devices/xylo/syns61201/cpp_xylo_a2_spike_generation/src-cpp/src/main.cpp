// other necessary modules
#include<iostream>
#include<cassert>
#include<vector>
#include<future>
#include <thread>
#include <functional>
#include<cmath>
#include <cstdlib>

using namespace std;



// modules needed for pybind11
#include<pybind11/pybind11.h>
#include<Python.h>
#include <pybind11/stl.h>

namespace py = pybind11;








////////////////////////////////////////////////////////////////////
/* LIF spike generation module based on the parameters of Xylo-A2 */
////////////////////////////////////////////////////////////////////




/////////////////////////////////////////////////////////////////////
/*                Ordinary Version as in AFESim                    */
/////////////////////////////////////////////////////////////////////

std::pair<std::vector<int>, float> _encode_spikes_single_channel(
    float initial_state, float dt, std::vector<float> data, float v2i_gain, float c_iaf, float leakage, float thr_up, float vcc
)
{
    /* parameters:
    initial_state (float): initial value of the voltage of the capacitor in the integrator circuit.
    dt (float): clock or sampling period of the input signal.
    data (vector<float>): input data coming from rectifier.
    v2i_gain (float): voltage to current conversion gain at the input of the integrator circuit.
    c_iaf (float): capacitor value of spike generation integrator circuit.
    leakage (float): the leakage conductance of spike generation integrator circuit.
    thr_up (float): spike generation voltage threshold.
    vcc (float): supply voltage value, i.e., the maximum value the integrator output can reach before saturation.

    Returns:
        tuple(vector<int>, float): vector containing the generated spikes + the final value of the capacitor voltage (final state).
    */

    // check the value of voltage supply
    if (vcc <= thr_up){
        std::cout<<"firing threshold is less then voltage supply: the neuron does not fire at all!" << std::endl;
        exit(1);
    }
   
    // generated spikes
    std::vector<int> spikes;

    // initialize the state
    float cdc = initial_state;


    for (int i=0; i<int(data.size()); i++){
        // leakage current when the capacitor has a voltage of cdc
        float lk = leakage * cdc;

        // how much charge is depleted from the capacitor during `dt`
        float dq_lk = lk * dt;
        //**std::cout << "charge due to leakage is " << dq_lk << std::endl;
        

        // how much charge is added to the capacitor because of input data in `dt`
        float dq_data = dt * (data.at(i) * v2i_gain);
        //**std::cout << "charge due to data is " << dq_data << std::endl;


        // variation in capacitor voltage dur to data + leakage
        float dv = (dq_data - dq_lk) / (c_iaf);
        //**std::cout << "voltage change is " << dv << std::endl;


        // Accumulate membrane voltage, clip to the range [0, VCC]
        cdc += dv;
        //**std::cout << "capacitor voltage before reset is " << cdc << std::endl;

        // check if capacitor voltage is within the valid range of [0, Vcc]
        if (cdc < 0.0){
            cdc = 0.0;
        }
        
        if (cdc > vcc){
            cdc = vcc;
        }

        // check if any spikes are going to be produced
        // NOTE: 
        // in software it is possible to work with integer {0,1,2,...} number of spikes
        // and do soft-reset by dropping the membrane voltage by the same integer multiple of threshold.
        // This is not the case in chip: only {0,1} spikes are possible and hard-reset happen when spike is produced.
        bool there_is_spike = (cdc >= thr_up);

        if (there_is_spike){
            spikes.push_back(1);
            cdc = 0;
        }else{
            spikes.push_back(0);
        }

    }

    // return rastered spikes and final voltage of the capacitor
    std::pair<std::vector<int>, float> result(spikes, cdc);

    return result;

}


// std::pair<std::vector<std::vector<int>>, std::vector<float>> _encode_spikes(
//     std::vector<float> initial_state, 
//     float dt,
//     std::vector<std::vector<float>> data,
//     float v2i_gain,
//     float c_iaf,
//     float leakage, 
//     float thr_up,
//     float vcc)
// {
//     /* parameters:
//     initial_state : a vector containing the initial values of the voltages of the capacitors in integrator circuits used for spike generation.
//     dt (float): clock or sampling period of the input signal.
//     data : vector of input data coming from rectifier along channels corresponding to filters.
//     v2i_gain (float): voltage to current conversion gain at the input of the integrator circuit.
//     c_iaf (float): capacitor value of spike generation integrator circuit.
//     leakage (float): the leakage conductance of spike generation integrator circuit.
//     threshold (float): spike generation voltage threshold.
//     vcc (float): supply voltage value, i.e., the maximum value the integrator output can reach before saturation.
    
//     Returns:
//         tuple(vector<vector<int>>, vector<float>): vector containing the spikes at different channels + 
//                                                     the final values of the capacitor voltages (final states) at different channels.
//     */

//     std::vector<std::vector<int>> spikes;
//     std::vector<float> final_state;

//     int num_channels = int(data.size());

//     for (int i=0; i<num_channels; i++){
//         std::pair<std::vector<int>, float> channel_output = _encode_spikes_single_channel(
//             initial_state.at(i),
//             dt,
//             data.at(i),
//             v2i_gain,
//             c_iaf,
//             leakage,
//             thr_up,
//             vcc
//         );

//         spikes.push_back(channel_output.first);
//         final_state.push_back(channel_output.second);
//     }

//     // return results
//     auto result = std::pair(spikes, final_state);

//     return result;
// }

std::pair<std::vector<std::vector<int>>, std::vector<float>> _encode_spikes(
    std::vector<float> initial_state, 
    float dt,
    std::vector<std::vector<float>> data,
    float v2i_gain,
    float c_iaf,
    float leakage, 
    float thr_up,
    float vcc)
{
    /* parameters:
    initial_state : a vector containing the initial values of the voltages of the capacitors in integrator circuits used for spike generation.
    dt (float): clock or sampling period of the input signal.
    data : vector of input data coming from rectifier along channels corresponding to filters.
    v2i_gain (float): voltage to current conversion gain at the input of the integrator circuit.
    c_iaf (float): capacitor value of spike generation integrator circuit.
    leakage (float): the leakage conductance of spike generation integrator circuit.
    threshold (float): spike generation voltage threshold.
    vcc (float): supply voltage value, i.e., the maximum value the integrator output can reach before saturation.
    
    Returns:
        tuple(vector<vector<int>>, vector<float>): vector containing the spikes at different channels + 
                                                    the final values of the capacitor voltages (final states) at different channels.
    */

    // container for saving the results
    std::vector<std::future<std::pair<std::vector<int>, float>>> futures;
    
    int num_channels = int(data.size());

    // call the filter function and register the output
    for (auto i=0; i<num_channels; i++){
        auto task = std::async(std::launch::async, _encode_spikes_single_channel, initial_state.at(i), dt, data.at(i), v2i_gain, c_iaf, leakage, thr_up, vcc);
        futures.emplace_back(std::move(task));
    }

    // wait for it
    // create the output list
    std::vector<std::vector<int>> spikes;
    std::vector<float> final_state;

    for (auto& e : futures){
        auto result = e.get();
        spikes.emplace_back(result.first);
        final_state.push_back(result.second);
    } 

    auto output = std::pair<std::vector<std::vector<int>>, std::vector<float>>(spikes, final_state);

    return output;
}








/////////////////////////////////////////////////////////////////////
/*              Rastered Version : More efficient                  */
/*                                                                 */
/* This is saved for future optimization in case needed.           */
/////////////////////////////////////////////////////////////////////


// std::pair<std::vector<int>, float> lif_spike_gen_single(float initial_state, std::vector<float> data, float v2i_gain, float c_iaf, float leakage, 
//                                       float thr_up, float vcc, float dt, float raster_dt, int max_num_spikes){
//     /* parameters:
//     initial_state (float): initial value of the voltage of the capacitor in integrator circuit.
//     data (vector<float>): input data coming from rectifier.
//     v2i_gain (float): voltage to current conversion gain at the input of the integrator circuit.
//     c_iaf (float): capacitor value of spike generation integrator circuit.
//     leakage (float): the leakage conductance of spike generation integrator circuit.
//     thr_up (float): spike generation voltage threshold.
//     vcc (float): supply voltage value, i.e., the maximum value the integrator output can reach before saturation.
//     dt (float): clock or sampling period of the input signal.
//     raster_dt (float): clock or sampling period of the accumulated spikes after being rastered.
//     max_num_spikes (int): maximum number of spikes accepeted within a rastering period. In Xylo chip this is limited to 15.

//     Returns:
//         tuple(vector<int>, float): vector containing the accumulated/rastered spikes + the final value of the capacitor voltage (final state).
//     */

//     // check the value of voltage supply
//     if (vcc <= thr_up){
//         std::cout<<"firing threshold is less then voltage supply: the neuron does not fire at all!" << std::endl;
//         exit(1);
//     }
   
//     // whole duration of the signal
//     float signal_duration = data.size() * dt;

//     // number of rastered spikes period to expect
//     int num_rastered_period = ceil(signal_duration/raster_dt);

//     // initialize the rastered spikes
//     std::vector<int> rastered_spikes;

//     // initialize the state
//     float cdc = initial_state;
//     int clock_cycle = -1;
//     float next_raster_time = raster_dt;
//     int num_raster_spikes = 0;


//     for (int i=0; i<int(data.size()); i++){
//         // update the clock
//         //std::cout << " time is " << clock<< std::endl;
//         clock_cycle += 1;

//         if (clock_cycle > next_raster_time / dt ){
//             /* it is time to spit-out rastered spike accumulated during the past raster period */

//             // put limit on the number of spikes
//             if (num_raster_spikes > max_num_spikes){
//                 num_raster_spikes = max_num_spikes;
//             }

//             //**std::cout << num_raster_spikes << "====> spike were added to the list at time " << clock << std::endl;
//             rastered_spikes.push_back(num_raster_spikes);

//             // reset spike counts to 0
//             num_raster_spikes = 0;

//             // set the time for the next raster
//             next_raster_time += raster_dt;
//         }

//         /* continue processing the input data */

//         // leakage current when the cpacitor has a voltage of cdc
//         float lk = leakage * cdc;

//         // how much charge is depleted from the capacitor during `dt`
//         float dq_lk = lk * dt;
//         //**std::cout << "charge due to leakage is " << dq_lk << std::endl;
        

//         // how much charge is added to the capacitor because of input data in `dt`
//         float dq_data = dt * (data.at(i) * v2i_gain);
//         //**std::cout << "charge due to data is " << dq_data << std::endl;


//         // variation in capacitor voltage dur to data + leakage
//         float dv = (dq_data - dq_lk) / (c_iaf);
//         //**std::cout << "voltage change is " << dv << std::endl;


//         // Accumulate membrane voltage, clip to the range [0, VCC]
//         cdc += dv;
//         //**std::cout << "capacitor voltage before reset is " << cdc << std::endl;

//         // check if capacitor voltage is within the valid range of [0, Vcc]
//         if (cdc < 0.0){
//             cdc = 0.0;
//         }
        
//         if (cdc > vcc){
//             cdc = vcc;
//         }

//         // check if any spikes are going to be produced
//         // NOTE: 
//         // in software it is possible to work with integer {0,1,2,...} number of spikes
//         // and do soft-reset by dropping the membrane voltage by the same integer multiple of threshold.
//         // This is not the case in chip: only {0,1} spikes are possible and hard-reset happen when spike is produced.
//         bool there_is_spike = (cdc > thr_up);

//         if (there_is_spike){
//             num_raster_spikes++;
        
//             //**std::cout << "number of spikes accumulated during this clock " << num_spikes << std::endl;
//             //**std::cout << "total number of rastered spikes within this period so far: " << num_raster_spikes << std::endl;

//             cdc = 0;
//             //**std::cout << "capacitor voltage after possible reset is " << cdc << std::endl<< std::endl << std::endl;
//         }

//     }

//     // this is the last raster period: append all the remaining spikes
//     if (num_raster_spikes > max_num_spikes){
//         num_raster_spikes = max_num_spikes;
//     }
//     rastered_spikes.push_back(num_raster_spikes);

//     // return rastered spikes and final voltage of the capacitor
//     std::pair<std::vector<int>, float> result(rastered_spikes, cdc);

//     return result;

// }


// std::pair<std::vector<std::vector<int>>, std::vector<float>> lif_spike_gen(
//     std::vector<float> initial_state, 
//     std::vector<std::vector<float>> data,
//     float v2i_gain,
//     float c_iaf,
//     float leakage, 
//     float thr_up,
//     float vcc,
//     float dt,
//     float raster_dt,
//     int max_num_spikes
// ){
//     /* parameters:
//     initial_state : a vector containing the initial values of the voltages of the capacitors in integrator circuits used for spike generation.
//     data : vector of input data coming from rectifier along channels corresponding to filters.
//     v2i_gain (float): voltage to current conversion gain at the input of the integrator circuit.
//     c_iaf (float): capacitor value of spike generation integrator circuit.
//     leakage (float): the leakage conductance of spike generation integrator circuit.
//     threshold (float): spike generation voltage threshold.
//     vcc (float): supply voltage value, i.e., the maximum value the integrator output can reach before saturation.
//     dt (float): clock or sampling period of the input signal.
//     raster_dt (float): clock or sampling period of the accumulated spikes after being rastered.
//     max_num_spikes (int): maximum number of spikes accepeted within a rastering period. In Xylo chip this is limited to 15.

//     Returns:
//         tuple(vector<vector<int>>, vector<float>): vector containing the accumulated/rastered spikes at different channels + 
//                                                     the final values of the capacitor voltages (final states) at different channels.
//     */

//     std::vector<std::vector<int>> spikes;
//     std::vector<float> final_state;

//     int num_channels = int(data.size());

//     for (int i=0; i<num_channels; i++){
//         std::pair<std::vector<int>, float> channel_output = lif_spike_gen_single(
//             initial_state.at(i),
//             data.at(i),
//             v2i_gain,
//             c_iaf,
//             leakage,
//             thr_up,
//             vcc,
//             dt,
//             raster_dt,
//             max_num_spikes
//         );

//         spikes.push_back(channel_output.first);
//         final_state.push_back(channel_output.second);
//     }

//     // return results
//     auto result = std::pair(spikes, final_state);

//     return result;
// }







//simple binding
PYBIND11_MODULE(xylo_a2_spike_generation, m) {
    m.doc() = "This module implements the LIF spike generation method in Xylo-A2 in C++."; // optional module docstring

    m.def(
        "_encode_spikes",
        &_encode_spikes,
        "This function implements the LIF spike generation method in Xylo-A2 to convert the output of the rectifier into spikes.",
        py::arg("initial_state"),
        py::arg("dt")=1.0/48000,
        py::arg("data"),
        py::arg("v2i_gain")=3.333e-7,
        py::arg("c_iaf")=5.0e-12,
        py::arg("leakage")=1.0e-9,
        py::arg("thr_up")=0.5,
        py::arg("vcc")=1.1
    );

}




////////////////////////////
/* some simple test cases */
///////////////////////////
/*
void test_spike_gen_single(){
    float initial_state = 0.0;
    float v2i_gain = 3.333e-6;
    float c_iaf = 5e-12;
    float leakage = 1.0e-9;
    float thr_up = 0.5;
    float vcc = 1.1;
    float dt = 1.0/50000.0;
    float raster_dt = 1.0/10000.0;
    int max_num_spikes = 15;

    std::vector<float> data;

    // push some positive rectified values into data
    int num_data = 100;
    for (int i=0; i<num_data; i++){
        data.push_back( (30e-3 * rand())/RAND_MAX);
    }

    std::cout<<"input data is :" << std::endl;
    for (int i=0; i<num_data; i++){
        std::cout<<data.at(i) << std::endl;
    }

    auto output = lif_spike_gen_single(
        initial_state,
        data,
        v2i_gain,
        c_iaf,
        leakage,
        thr_up,
        vcc,
        dt,
        raster_dt,
        max_num_spikes
    );

    auto spikes = output.first;

    std::cout <<"number of rasters produced: " << spikes.size() << std::endl;

    // print spikes
    for (int i=0; i<int(spikes.size()); i++){
        std::cout<<spikes.at(i) << std::endl;
    }

}
*/




int main(){
    // test spike generation
    //test_spike_gen_single();
    return 0;
}
 
