"""
Network class for implementing networks that can learn arbitrary dynamical systems (see https://arxiv.org/pdf/1705.08026.pdf for more information)
Author: Julian Buechel
Note that ADS stands for Arbitrary Dynamical System
"""
import numpy as np
from ..network import Network
from ...layers import PassThrough, FFExpSyn, RecFSSpikeADS
from ...timeseries import TSContinuous
import json

from tqdm import tqdm

from typing import Union, Callable, Tuple, List

import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt

import time

__all__ = ["NetworkADS"]

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0, axis=0), axis=0) 
    return (cumsum[N:,:] - cumsum[:-N,:]) / float(N)

def pISI_variance(sim_result):
    """
    Compute the variance of the population inter-spike intervals
    Parameters:
        sim_result : Object of type evolution Object that was returned after having called evolve
    Returns:
        variance of difference array (variance of pISI) in milliseconds
    """
    times_c = sim_result['lyrRes'].times[sim_result['lyrRes'].channels > -1]
    np.sort(times_c) # Sorts in ascending order
    diff = np.diff(times_c)
    return np.sqrt(np.var(diff * 1000))

class NetworkADS(Network):

    def __init__(self):
        super().__init__()

    @staticmethod
    def load(filename: str) -> "Network":
        """
        Load a network from a JSON file

        :param str filename:    filename of a JSON filr that contains a saved network
        :return Network:        A network object with all the layers loaded from `filename`
        """
        # - Load dict holding the parameters
        with open(filename, "r") as f:
            loaddict: dict = json.load(f)

        return NetworkADS.load_from_dict(loaddict)

    @staticmethod
    def load_from_dict(config: dict, **kwargs):

        # - Overwrite parameters with kwargs
        config = dict(config, **kwargs)
        conf = config["layers"][1]
        conf.pop("class_name", None)
        conf.pop("name", None)
        conf["tau_syn_r_out"] = config["layers"][2]["tau_syn"]
        conf.pop("pre_layer_name", None)
        conf.pop("external_input", None)
        return NetworkADS.SpecifyNetwork(**conf)

    @staticmethod
    def SpecifyNetwork(N : int,
                        Nc : int,
                        Nb : int,
                        weights_in : np.ndarray,
                        weights_out : np.ndarray,
                        weights_fast : np.ndarray,
                        weights_slow : np.ndarray,
                        M : np.ndarray,
                        theta : np.ndarray,
                        eta : float,
                        k : float,
                        noise_std : float,
                        dt : float,
                        bias : np.ndarray = 0.0,
                        v_thresh: Union[np.ndarray, float] = -0.055,
                        v_reset: Union[np.ndarray, float] = -0.065,
                        v_rest: Union[np.ndarray, float] = -0.065,
                        tau_mem: float = 0.02,
                        tau_syn_r_fast: float = 0.001,
                        tau_syn_r_slow: float = 0.1,
                        tau_syn_r_out: float = 0.1,
                        refractory: float = -np.finfo(float).eps,
                        record : bool = False,
                        phi : str = "tanh"):


        """
        @brief
            Creates a network for learning 

        @params
            Nc : Dimension of input signal
            N : Number of neurons in the recurrent layer
            Nb : Number of basis functions used

            The above are passed explicitly to ensure that the desired network size matches the matrix shapes

            weights_in : [Nc,N] Weights that connect the input to the recurrent population
            weights_out : [N,Nc] Weights that reconstruct desired output x from r of the netwok (x_hat = Dr), typically set to weights_in.T
            weights_fast : [N,N] Recurrent weights for tight E/I-balance
            weights_slow : [Nb,N] Weight matrix relating psi(r) to the neurons
            theta : [Mb,1] Bias in the non-linear activation function
            eta : Learning rate for adjusting weights_slow
            k : Gain of error feedback during learning. Typically decreased while learning for generalization
            noise_std : Standard deviation of noise added to the neurons membrane potentials
            phi : Non-linear function of type ndarray -> ndarray. Implements tanh per default
        """

        # Assertions for checking the dimensions
        assert (np.asarray(weights_in).shape == (Nc,N)), ("Input matrix has shape %s but should have shape (%d,%d)" % (str(np.asarray(weights_in).shape),N,Nc))
        assert (np.asarray(weights_out).shape == (N,Nc)), ("Output matrix has shape %s but should have shape (%d,%d)" % (str(np.asarray(weights_out).shape),Nc,N))
        assert (np.asarray(weights_fast).shape == (N,N)), ("Fast recurrent matrix has shape %s but should have shape (%d,%d)" % (str(np.asarray(weights_fast).shape),N,N))
        assert (np.asarray(weights_slow).shape == (Nb,N)), ("Slow recurrent matrix has shape %s but should have shape (%d,%d)" % (str(np.asarray(weights_slow).shape),Nb,N))
        assert (np.asarray(theta).shape == (Nb,1) or np.asarray(theta).shape == (Nb,)), ("Theta has shape %s but should have shape (%d,1)" % (str(np.asarray(theta).shape),Nb))
        assert (phi == "tanh" or phi == "relu"), ("Please specify phi to be either tanh or relu")

        ads_layer = RecFSSpikeADS(weights_fast=np.asarray(weights_fast).astype("float"), weights_slow=np.asarray(weights_slow).astype("float"), weights_out = np.asarray(weights_out).astype("float"), weights_in=np.asarray(weights_in).astype("float"),
                                    M=np.asarray(M).astype("float"),theta=np.asarray(theta).astype("float"),eta=eta,k=k,bias=np.asarray(bias).astype("float"),noise_std=noise_std,
                                    dt=dt,v_thresh=np.asarray(v_thresh).astype("float"),v_reset=np.asarray(v_reset).astype("float"),v_rest=np.asarray(v_rest).astype("float"),
                                    tau_mem=tau_mem,tau_syn_r_fast=tau_syn_r_fast,tau_syn_r_slow=tau_syn_r_slow, tau_syn_r_out=tau_syn_r_out,
                                    refractory=refractory,phi=phi,record=record,name="lyrRes")

        # Input layer
        input_layer = PassThrough(np.asarray(weights_in).astype("float"), dt=dt, noise_std=noise_std, name="input_layer")

        # Output layer
        output_layer = FFExpSyn(
            np.asarray(weights_out).astype("float"),
            dt=dt,
            noise_std=noise_std,
            tau_syn=tau_syn_r_out,
            name="output_layer"
        )

        net_ads = NetworkADS()
        net_ads.input_layer = net_ads.add_layer(input_layer, external_input=True) # External -> Input
        net_ads.lyrRes = net_ads.add_layer(ads_layer, input_layer) # Input -> ADS
        net_ads.output_layer = net_ads.add_layer(output_layer, ads_layer) # ADS -> Output

        return net_ads

    def train_network(self,
                        data_train : List[Tuple[np.ndarray,np.ndarray]] = None,
                        data_val : List[Tuple[np.ndarray,np.ndarray]] = None,
                        get_data : Callable[[],np.ndarray] = None,
                        num_validate : int = -1,
                        num_iter : int = -1,
                        validation_step : int = 0,
                        time_base : np.ndarray = None,
                        verbose : int = 0,
                        N_filter : int = 31,
                        ):
        """
        Training function that trains the ADS network using either:
            1) A function to generate input,target pairs. In this case the network will be trained for num_iter many iterations
            and validated on a validation set of size num_validate generated by the function prior to training.
            2) Two data-sets, data_train and data_val, which contain tuples of input,target pairs. The network will be trained on the whole training data-set
            for num_iter many iterations. If function should step through the data-set once, simply set num_iter to 1. After validation_step many
            steps, the network will be evaluated on the validation data.
        Parameters:
            data_train : List[Tuple[np.ndarray,np.ndarray]] = None A list of training examples. Assert that data_val is not None if this field contains data
            data_val : List[Tuple[np.ndarray,np.ndarray]] = None A list of validation examples.
            get_data : Callable[[],np.ndarray] = None Should take no argument and simply return one training/validation example. This should be used for option 1)
            num_validate : int = -1 Used for option 1). If get_data is not None assert that this is larger than -1. Used to set the number of validation examples.
            num_iter : int = -1 Used for all options. If option is 1) num_iter = The number of training signals presented over training period. If the option is
                                2) then num_iter will be the number of times data_train will be iterated over.
            validation_step : int = 0 Equal meaning for all options: After how many signal iterations will the validation be performed. Assert that this is > 0
            time_base : np.ndarray = None Time base used for creating TSContinuous inputs
            verbose : int = 0 Verbose output. 0 for no verbosity, 1 for validation only and 2 for verbose output during training and validation
        """

        if(data_train is not None):
            assert(data_val is not None), "Training data set supplied, but no validation data set"
            assert(num_validate == -1), "Training data set supplied. Please omit num_validate"
            assert(num_iter > -1), "Please specify a number of iterations"
            assert(get_data is None), "Training data set supplied, but also a data-generating function"

        if(data_val is not None):
            assert(data_train is not None), "Validation set provided, but no training set"

        if(get_data is not None):
            assert(num_validate > -1), "Get_data function specified, but no number of validation signals (num_validate)"

        assert(validation_step > 0), "Please specify a validation step (Number of signal iterations after which network is validated)"

        assert(time_base is not None), "Please specify time_base for creating TSContinuous object"

        num_signal_iterations = 0        

        def perform_validation_set():
            assert(self.lyrRes.is_training == False), "Validating, but is_training flag is set"
            assert(self.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"

            errors = []
            variances = []
            for (input_val, target_val) in data_val:
                ts_input_val = TSContinuous(time_base, input_val.T)
                ts_target_val = TSContinuous(time_base, target_val.T)
                self.lyrRes.ts_target = ts_target_val
                val_sim = self.evolve(ts_input=ts_input_val, verbose=(verbose > 0))
                variances.append(pISI_variance(val_sim))
                out_val = val_sim["output_layer"].samples.T
                self.reset_all()

                if(verbose > 0):
                    fig = plt.figure(figsize=(20,5))
                    plt.plot(time_base, out_val[0:2,:].T, label=r"Reconstructed")
                    plt.plot(time_base, target_val[0:2,:].T, label=r"Target")
                    plt.title(r"Target vs reconstruction")
                    plt.legend()
                    plt.draw()
                    plt.waitforbuttonpress(0) # this will wait for indefinite time
                    plt.close(fig)

                if(target_val.ndim == 1):
                    target_val = np.reshape(target_val, (out_val.shape))
                    target_val = target_val.T
                    out_val = out_val.T

                network_response_smoothed_padded = np.zeros(target_val.shape)
                network_response_smoothed = running_mean(out_val.T, N_filter)
                network_response_smoothed_padded[:,int((N_filter-1)/2):network_response_smoothed.shape[0]+int((N_filter-1)/2)] = network_response_smoothed.T

                err = np.sum(np.var(target_val-out_val, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
                err = np.sum(np.var(target_val-network_response_smoothed_padded, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
                errors.append(err)
                self.lyrRes.ts_target = None
            return np.mean(np.asarray(errors)), np.mean(np.asarray(variances))
        # End perform_validation_set

        ########## Training on data set and validating on validation set ##########
        if(data_train is not None and data_val is not None):
            print("Start training on data set...Total number of training signals presented: %d" % (num_iter*len(data_train)))
            assert(self.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"

            with tqdm(total=num_iter) as pbar_outer:
                pbar_outer.set_description("Batch")
            
                with tqdm(total=len(data_train), bar_format="{postfix[0]} # of signal iterations: {postfix[1][num_signal_iterations]} Validation error: {postfix[1][validation_error]} Mean pISI: {postfix[1][mean_pISI]} eta: {postfix[1][eta]} k: {postfix[1][k]} {l_bar}{bar}",
                    postfix=["Training:", dict(num_signal_iterations=0,validation_error=0,mean_pISI=0,eta=0,k=0)]) as t:

                    for iteration in range(num_iter):
                        for (input_train, target_train) in data_train:
                            assert(self.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"
                            if(num_signal_iterations % validation_step == 0):
                                validation_error, pISI = perform_validation_set()

                            self.lyrRes.is_training = True
                            ts_input_train = TSContinuous(time_base, input_train.T)
                            ts_target_train = TSContinuous(time_base, target_train.T)

                            self.lyrRes.ts_target = ts_target_train
                            self.evolve(ts_input=ts_input_train, verbose=(verbose==2))
                            self.reset_all()
                            self.lyrRes.is_training = False
                            self.lyrRes.ts_target = None
                            num_signal_iterations += 1

                            t.postfix[1]["num_signal_iterations"] = num_signal_iterations
                            t.postfix[1]["validation_error"] = validation_error
                            t.postfix[1]["mean_pISI"] = pISI
                            t.postfix[1]["eta"] = self.lyrRes.eta
                            t.postfix[1]["k"] = self.lyrRes.k
                            t.update(1)
                        t.reset()
                        pbar_outer.update(1)


        ########## Training using newly generated data from get_data ##########
        elif(get_data is not None):
            print("Start training using get_data...")
            # Create val_data
            data_val = [get_data() for _ in range(num_validate)]
            assert(self.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"

            with tqdm(total=num_iter, bar_format="{postfix[0]} # of signal iterations: {postfix[1][num_signal_iterations]} Validation error: {postfix[1][validation_error]} Mean pISI: {postfix[1][mean_pISI]} eta: {postfix[1][eta]} k: {postfix[1][k]} {l_bar}{bar}",
                    postfix=["Training:", dict(num_signal_iterations=0,validation_error=0,mean_pISI=0,eta=0,k=0)]) as t:
                for iteration in range(1,num_iter+1):
                    if(num_signal_iterations % validation_step == 0):
                            validation_error, pISI = perform_validation_set()
                    (input_train, target_train) = get_data()
                    self.lyrRes.is_training = True
                    ts_input_train = TSContinuous(time_base, input_train.T)
                    ts_target_train = TSContinuous(time_base, target_train.T)

                    if(iteration % 100 == 0 and self.lyrRes.eta > 0.0001):
                        self.lyrRes.eta = 1/np.sqrt(iteration) * self.lyrRes.eta_initial
                    if(iteration % 100 == 0 and self.lyrRes.k > 0.001):
                        self.lyrRes.k = 1/np.sqrt(iteration) * self.lyrRes.k_initial

                    self.lyrRes.ts_target = ts_target_train
                    self.evolve(ts_input=ts_input_train, verbose=(verbose==2))
                    self.reset_all()
                    self.lyrRes.is_training = False
                    self.lyrRes.ts_target = None
                    num_signal_iterations += 1
                    
                    t.postfix[1]["num_signal_iterations"] = num_signal_iterations
                    t.postfix[1]["validation_error"] = validation_error
                    t.postfix[1]["mean_pISI"] = pISI
                    t.postfix[1]["eta"] = self.lyrRes.eta
                    t.postfix[1]["k"] = self.lyrRes.k
                    t.update(1)



def _expand_to_size(inp, size: int, var_name: str = "input") -> np.ndarray:
    """
    _expand_to_size: Replicate out a scalar to a desired size

    :param inp:          scalar or array-like (N)
    :param size:           int Desired size to return (=N)
    :param var_name:   str Name of the variable to include in error messages
    :return:                np.ndarray (N) vector
    """
    if np.size(inp) == 1:
        # - Expand input to vector
        inp = np.repeat(inp, size)

    assert np.size(inp) == size, "`{}` must be a scalar or have {} elements".format(
        var_name, size
    )

    # - Return object of correct shape
    return np.reshape(inp, size)