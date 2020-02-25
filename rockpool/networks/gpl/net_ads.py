"""
Network class for implementing networks that can learn arbitrary dynamical systems (see https://arxiv.org/pdf/1705.08026.pdf for more information)
Author: Julian Buechel
Note that ADS stands for Arbitrary Dynamical System
"""
import numpy as np
from ..network import Network
from ...layers import PassThrough, FFExpSyn, RecFSSpikeADS
from ...timeseries import TSContinuous

from progress.bar import ChargingBar

from typing import Union, Callable, Tuple, List

import matplotlib
matplotlib.rc('font', family='Times New Roman')
matplotlib.rc('text')
matplotlib.rcParams['lines.linewidth'] = 0.5
matplotlib.rcParams['lines.markersize'] = 0.5
import matplotlib.pyplot as plt # For quick plottings

import time

#! Need to import RecFSSpikeADS

__all__ = ["NetworkADS"]

class NetworkADS(Network):

    def __init__(self):
        super().__init__()

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
                        tau_syn_out: float = 0.1,
                        refractory: float = -np.finfo(float).eps,
                        record : bool = False,
                        phi : Callable[[np.ndarray],np.ndarray] = lambda x : np.tanh(x)):


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
        assert (weights_in.shape == (Nc,N)), ("Input matrix has shape %s but should have shape (%d,%d)" % (str(weights_in.shape),N,Nc))
        assert (weights_out.shape == (N,Nc)), ("Output matrix has shape %s but should have shape (%d,%d)" % (str(weights_out.shape),Nc,N))
        assert (weights_fast.shape == (N,N)), ("Fast recurrent matrix has shape %s but should have shape (%d,%d)" % (str(weights_fast.shape),N,N))
        assert (weights_slow.shape == (Nb,N)), ("Slow recurrent matrix has shape %s but should have shape (%d,%d)" % (str(weights_slow.shape),Nb,N))
        assert (theta.shape == (Nb,1) or theta.shape == (Nb,)), ("Theta has shape %s but should have shape (%d,1)" % (str(theta.shape),Nb))

        ads_layer = RecFSSpikeADS(weights_fast=weights_fast, weights_slow=weights_slow, weights_out = weights_out, weights_in=weights_in,
                                    M=M,theta=theta,eta=eta,k=k,bias=bias,noise_std=noise_std,
                                    dt=dt,v_thresh=v_thresh,v_reset=v_reset,v_rest=v_rest,
                                    tau_mem=tau_mem,tau_syn_r_fast=tau_syn_r_fast,tau_syn_r_slow=tau_syn_r_slow,
                                    refractory=refractory,phi=phi,learning_callback=None,record=record,name="ADS-Layer")

        # Input layer
        input_layer = PassThrough(weights_in, dt=dt, noise_std=noise_std, name="Input")

        # Output layer
        output_layer = FFExpSyn(
            weights_out,
            dt=dt,
            noise_std=noise_std,
            tau_syn=tau_syn_out,
            name="Output"
        )

        net_ads = NetworkADS()
        net_ads.input_layer = net_ads.add_layer(input_layer, external_input=True) # External -> Input
        net_ads.lyrRes = net_ads.add_layer(ads_layer, input_layer) # Input -> ADS
        net_ads.output_layer = net_ads.add_layer(output_layer, ads_layer) # ADS -> Output

        return net_ads

    def pISI_variance(self, sim_result):
        """
        Compute the variance of the population inter-spike intervals
        Parameters:
            sim_result : Object of type evolution Object that was returned after having called evolve
        Returns:
            variance of difference array (variance of pISI)
        """
        times_c = sim_result['ADS-Layer'].times[sim_result['ADS-Layer'].channels > -1]
        np.sort(times_c) # Sorts in ascending order
        diff = np.diff(times_c)
        return np.var(1000*diff)


    def train_network(self,
                        single_data : Tuple[np.ndarray,np.ndarray] = None,
                        data_train : List[Tuple[np.ndarray,np.ndarray]] = None,
                        data_val : List[Tuple[np.ndarray,np.ndarray]] = None,
                        get_data : Callable[[],np.ndarray] = None,
                        num_validate : int = -1,
                        num_iter : int = -1,
                        validation_step : int = 0,
                        time_base : np.ndarray = None,
                        verbose = 0
                        ):
        """
        Training function that trains the ADS network using either:
            1) A function to generate input,target pairs. In this case the network will be trained for num_iter many iterations
            and validated on a validation set of size num_validate generated by the function prior to training.
            2) A single input,target pair. In this case the network will be trained and validated on the same input for num_iter many iterations.
            3) Two data-sets, data_train and data_val, which contain tuples of input,target pairs. The network will be trained on the whole training data-set
            for num_iter many iterations. If function should step through the data-set once, simply set num_iter to 1. After validation_step many
            steps, the network will be evaluated on the validation data.
        Parameters:
            single_data : Tuple[np.ndarray,np.ndarray] = None Pass a single tuple for option 2)
            data_train : List[Tuple[np.ndarray,np.ndarray]] = None A list of training examples. Assert that data_val is not None if this field contains data
            data_val : List[Tuple[np.ndarray,np.ndarray]] = None A list of validation examples.
            get_data : Callable[[],np.ndarray] = None Should take no argument and simply return one training/validation example. This should be used for option 1)
            num_validate : int = -1 Used for option 1). If get_data is not None assert that this is larger than -1. Used to set the number of validation examples.
            num_iter : int = -1 Used for all options. If option is 1) or 2) num_iter = The number of training signals presented over training period. If the option is
                                3) then num_iter will be the number of times data_train will be iterated over.
            validation_step : int = 0 Equal meaning for all options: After how many signal iterations will the validation be performed. Assert that this is > 0
            time_base : np.ndarray = None Time base used for creating TSContinuous inputs
            verbose : int = 0 Verbose output. 0 for no verbosity, 1 for validation only and 2 for verbose output during training and validation
        """
        if(single_data is not None):
            assert(data_train is None), "You supplied a single training/validation example and also a training data set"
            assert(data_val is None), "You supplied a single training/validation example and also a validation data set"
            assert(num_validate == -1), ("You supplied a single training/validation example, but num_validate was set to %d instead of -1" % num_validate)
            assert(num_iter > -1), "Please specify a number of iterations"
            assert(get_data is None), "Single training/validation example provided, but also a data-generating function"

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

        # Learning callback
        def learning_callback(weights_slow, phi_r, weights_in, e, dt):
            """
            Learning callback implementing learning rule W_slow_dot = eta*phi(r)(D.T @ e).T
            """
            return np.outer(phi_r,(weights_in.T @ e).T)

        # Set the learning callback
        self.lyrRes.learning_callback = learning_callback

        num_signal_iterations = 0

        def perform_validation_set():
            assert(self.lyrRes.is_training == False), "Validating, but is_training flag is set"
            assert(self.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"

            errors = []
            errors_last_third = []
            variances = []
            for (input_val, target_val) in data_val:
                ts_input_val = TSContinuous(time_base, input_val.T)
                ts_target_val = TSContinuous(time_base, target_val.T)
                self.lyrRes.ts_target = ts_target_val
                val_sim = self.evolve(ts_input=ts_input_val, verbose=(verbose > 0))
                variances.append(self.pISI_variance(val_sim))
                out_val = val_sim["Output"].samples.T
                self.reset_all()

                if(verbose > 0):
                    fig = plt.figure(figsize=(20,5))
                    plt.plot(time_base, out_val.T, label=r"Reconstructed")
                    plt.plot(time_base, target_val.T, label=r"Target")
                    plt.title(r"Target vs reconstruction")
                    plt.draw()
                    plt.waitforbuttonpress(0) # this will wait for indefinite time
                    plt.close(fig)

                if(target_val.ndim == 1):
                    target_val = np.reshape(target_val, (out_val.shape))
                    target_val = target_val.T
                    out_val = out_val.T
                last_third_start = int(out_val.shape[0] * 2/3)
                err_last_third = np.sum(np.var(target_val[last_third_start:,:]-out_val[last_third_start:,:], axis=0, ddof=1)) / (np.sum(np.var(target_val[last_third_start:,:], axis=0, ddof=1)))
                err = np.sum(np.var(target_val-out_val, axis=0, ddof=1)) / (np.sum(np.var(target_val, axis=0, ddof=1)))
                errors.append(err)
                errors_last_third.append(err_last_third)
                self.lyrRes.ts_target = None
            
            print("Number of signal iterations: %d Validation error: %.6f XOR error: %.6f Mean pISI variance: %.6f" % (num_signal_iterations, np.mean(np.asarray(errors)),np.mean(np.asarray(errors_last_third)), np.mean(np.asarray(variances))))
        # End perform_validation_set

        ########## Perform training/validation on single data point ##########
        if(single_data is not None):
            print("Start training on single data example...")

            ts_input = TSContinuous(time_base, single_data[0].T)
            ts_target = TSContinuous(time_base, single_data[1].T)

            self.lyrRes.ts_target = ts_target

            def perform_validation():
                assert(self.lyrRes.is_training == False), "Validating, but is_training flag is set"
                val_sim = self.evolve(ts_input=ts_input, verbose=(verbose > 1))
        
                out_val = val_sim["Output"].samples.T
                self.reset_all()

                if(verbose > 0):
                    fig = plt.figure(figsize=(20,5))
                    plt.plot(time_base, out_val.T, label=r"Reconstructed")
                    plt.plot(time_base, single_data[1].T, label=r"Target")
                    plt.title(r"Target vs reconstruction")
                    plt.draw()
                    plt.waitforbuttonpress(0) # this will wait for indefinite time
                    plt.close(fig)
                
                err = np.sum(np.var(single_data[1]-out_val, axis=0, ddof=1)) / (np.sum(np.var(single_data[1], axis=0, ddof=1)))
                print("Number of steps: %d Validation error: %.6f pISI variance: %.6f" % (iteration, err, self.pISI_variance(val_sim)))

            for iteration in range(num_iter):
                if(iteration % validation_step == 0):
                    perform_validation()

                self.lyrRes.is_training = True
                # Call evolve on self to perform one iteration of the network
                self.evolve(ts_input=ts_input, verbose=(verbose==2))
                # Reset state and time
                self.reset_all()
                # Reset to non-training state
                self.lyrRes.is_training = False

        ########## Training on data set and validating on validation set ##########
        elif(data_train is not None and data_val is not None):
            print("Start training on data set...Total number of training signals presented: %d" % (num_iter*len(data_train)))
            assert(self.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"

            for iteration in range(num_iter):
                print("Iteration %d through the training set" % iteration)
                bar = ChargingBar(message=("Training iteration %d" % iteration), max=len(data_train)+1)
                for (input_train, target_train) in data_train:
                    assert(self.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"
                    if(num_signal_iterations % validation_step == 0):
                        perform_validation_set()

                    self.lyrRes.is_training = True
                    ts_input_train = TSContinuous(time_base, input_train.T)
                    ts_target_train = TSContinuous(time_base, target_train.T)

                    self.lyrRes.ts_target = ts_target_train
                    self.evolve(ts_input=ts_input_train, verbose=(verbose==2))
                    self.reset_all()
                    self.lyrRes.is_training = False
                    self.lyrRes.ts_target = None
                    num_signal_iterations += 1
                    bar.next()
                bar.next(); bar.finish()


        ########## Training using newly generated data from get_data ##########
        elif(get_data is not None):
            print("Start training using get_data...")
            # Create val_data
            data_val = [get_data() for _ in range(num_validate)]
            assert(self.lyrRes.ts_target is None), "ts_target not set to None in spike_ads layer"

            bar = ChargingBar(message="Training", max=num_iter+1)
            for iteration in range(1,num_iter+1):
                if(num_signal_iterations % validation_step == 0):
                        perform_validation_set()
                (input_train, target_train) = get_data()
                self.lyrRes.is_training = True
                ts_input_train = TSContinuous(time_base, input_train.T)
                ts_target_train = TSContinuous(time_base, target_train.T)

                if(iteration % 10 == 0 and self.lyrRes.eta > 0.00005):
                    self.lyrRes.eta = 1/np.sqrt(iteration) * self.lyrRes.eta_initial
                    print("Reduced learning rate to %.7f" % self.lyrRes.eta)
                if(iteration % 100 == 0 and self.lyrRes.k > 0.001):
                    self.lyrRes.k = 1/np.sqrt(iteration) * self.lyrRes.k_initial
                    print("Reduced k to %.7f" % self.lyrRes.k)

                self.lyrRes.ts_target = ts_target_train
                self.evolve(ts_input=ts_input_train, verbose=(verbose==2))
                self.reset_all()
                self.lyrRes.is_training = False
                self.lyrRes.ts_target = None
                num_signal_iterations += 1
                bar.next()
            bar.next(); bar.finish()



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