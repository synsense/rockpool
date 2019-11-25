"""
rr_trained_layer.py - Define a super class that layers can inherit from if they
                      should be trained with ridge regression
"""

### --- Import statements

# - Built-ins
from abc import ABC, abstractmethod
from typing import Union, Dict
from warnings import warn

# - Third party packages
import numpy as np

# - Local imports
from ....timeseries import TSEvent, TSContinuous, TimeSeries
from ...layer import Layer
from .train_rr import RidgeRegrTrainer


class RRTrainedLayer(Layer, ABC):
    """
    RRTrainedLayer - Base class that defines methods for training with ridge
                     regression. `Layer` subclasses can inherit from this class
                     (by means of multiple inheritance) to provide ridge
                     regression functionality.

    """

    def train_rr(
        self,
        ts_target: TSContinuous,
        ts_input: Union[TSEvent, TSContinuous] = None,
        regularize: float = 0,
        is_first: bool = True,
        is_last: bool = False,
        train_biases: bool = True,
        calc_intermediate_results: bool = False,
        return_training_progress: bool = True,
        return_trained_output: bool = False,
        fisher_relabelling: bool = False,
        standardize: bool = False,
    ) -> Union[Dict, None]:
        """
        train_rr - Train self with ridge regression over one of possibly
                   many batches. Use Kahan summation to reduce rounding
                   errors when adding data to existing matrices from
                   previous batches.
        :param ts_target:        TimeSeries - target for current batch
        :param ts_input:         TimeSeries - input to self for current batch
        :param regularize:       float - regularization for ridge regression
        :param is_first:         bool - True if current batch is the first in training
        :param is_last:          bool - True if current batch is the last in training
                                       traning. This has the same effect as if data from both trainings
                                       were presented at once.
        :param train_biases:     bool - If True, train biases as if they were weights
                                        Otherwise present biases will be ignored in
                                        training and not be changed.
        :param calc_intermediate_results: bool - If True, calculates the intermediate weights not in the final batch
        :param return_training_progress:  bool - If True, return dict of current training
                                                 variables for each batch.
        :param standardize:      bool  -  Train with z-score standardized data, based on
                                          means and standard deviations from first batch
        :return:
            If `return_training_progress`, return dict with current trainig variables
            (xtx, xty, kahan_comp_xtx, kahan_comp_xty).
            Weights and biases are returned if `is_last` or if `calc_intermediate_results`.
            If `return_trained_output`, the dict contains the output of evolveing with
            the newly trained weights.
        """
        inp, target, time_base = self._prepare_training_data(
            ts_target=ts_target, ts_input=ts_input, is_first=is_first, is_last=is_last
        )

        if is_first:
            # - Generate trainer object
            self.trainer = RidgeRegrTrainer(
                num_features=self.size_in,
                num_outputs=self.size_out,
                regularization=regularize,
                fisher_relabelling=fisher_relabelling,
                standardize=standardize,
                train_biases=train_biases,
            )
        else:
            # - Make sure that trainig parameters are consistent
            for new_val, name in zip(
                (regularize, fisher_relabelling, standardize, train_biases),
                ("regularize", "fisher_relabelling", "standardize", "train_biases"),
            ):
                old_val = getattr(self.trainer, name)
                if old_val != new_val:
                    warn(
                        self.start_print
                        + f"Parameter `{name}` ({new_val}) differs from first "
                        + f"training batch. Will keep old value ({old_val})."
                    )

        update_weights = (
            is_last or return_training_progress or calc_intermediate_results
        )
        tr_data = self._batch_update(
            inp=inp,
            target=target,
            reset=is_last,
            train_biases=train_biases,
            standardize=standardize,
            update_weights=update_weights,
            return_training_progress=return_training_progress,
        )

        if return_trained_output:
            output_samples = inp @ self.trainer.weights + self.trainer.bias
            tr_data["output"] = TSContinuous(time_base, output_samples)

        if return_trained_output or return_training_progress:
            return tr_data

    def _batch_update(
        self,
        inp: np.ndarray,
        target: np.ndarray,
        reset: bool,
        train_biases: bool,
        standardize: bool,
        update_weights: bool,
        return_training_progress: bool,
    ) -> Dict:
        """
        _batch_update - Train with the already processed input and target data
                        of the current batch. Update layer wegihts and biases
                        if requested. Provide information on training state if
                        requested.
        :param np.ndarray inp:                  2D-array (num_samples x num_features) of input data.
        :param np.ndarray target:               2D-array (num_samples x 'self.size') of target data.
        :param bool reset:                      If `True', internal variables will be reset at the end.
        :param bool train_bises:                Should biases be trained or only weights?
        :param bool standardize:                Has input data been z-score standardized?
        :param bool update_weights:             Set 'True' to update layer weights and biases.
        :param bool return_training_progress:   Return intermediate training data (e.g. xtx, xty,...)
        :return dict:                           Dict with information on training progress, depending on values of other function arguments.
        """
        self.trainer.train_batch(inp, target)

        training_data = dict()

        if return_training_progress:
            training_data["trainig_progress"] = dict(
                xtx=self.trainer.xtx,
                xty=self.trainer.xty,
                kahan_comp_xtx=self.trainer.kahan_comp_xtx,
                kahan_comp_xty=self.trainer.kahan_comp_xty,
            )
            if standardize:
                training_data["trainig_progress"]["inp_mean"] = self.trainer.inp_mean
                training_data["trainig_progress"]["inp_std"] = self.trainer.inp_std

        if update_weights:
            self.trainer.update_model()
            self.weights = self.trainer.weights
            if train_biases:
                self.bias = self.trainer.bias
            if return_training_progress:
                training_data["trainig_progress"]["weights"] = self.trainer.weights
                if train_biases:
                    training_data["trainig_progress"]["biases"] = self.trainer.bias

        if reset:
            self.trainer.reset()

        return training_data

    @abstractmethod
    def _prepare_training_data(
        self, ts_target: TSContinuous, ts_input: TimeSeries, is_last: bool
    ) -> (Union[None, np.ndarray], np.ndarray, np.ndarray):
        """
        _prepare_training_data - Template for preparation of training data.
                                 Length of data is determined, dimensions are
                                 verified and target data is extracted from
                                 TSContinuous. Can be used in child classes
                                 through `super()._prepare_training_data`.
                                 Extraction of input data needs to be
                                 implemented in child classes.
        :param TSContinuous ts_target:  Target time series
        :param TimeSeries ts_input:     Input time series
        :param bool is_last:            Set `True` if batch is last of trianing.

        :return Union[None, np.ndarray]:  Extracted input data or 'None'.
        :return np.ndarray:               Extracted target data.
        :return np.ndarray:               Time points corresponding to target data.
        """
        # - Discrete time steps for evaluating input and target time series
        num_timesteps = int(np.round(ts_target.duration / self.dt))
        time_base = self._gen_time_trace(ts_target.t_start, num_timesteps)

        if not is_last:
            # - Discard last sample to avoid counting time points twice
            time_base = time_base[:-1]

        # - Make sure time_base does not exceed ts_target
        time_base = time_base[time_base <= ts_target.t_stop]

        # - Prepare target data
        target = ts_target(time_base)

        # - Make sure no nan is in target, as this causes learning to fail
        assert not np.isnan(
            target
        ).any(), "Layer `{}`: nan values have been found in target (where: {})".format(
            self.name, np.where(np.isnan(target))
        )

        # - Check target dimensions
        if target.ndim == 1 and self.size == 1:
            target = target.reshape(-1, 1)

        assert (
            target.shape[-1] == self.size
        ), "Layer `{}`: Target dimensions ({}) does not match layer size ({})".format(
            self.name, target.shape[-1], self.size
        )

        # Warn if input time range does not cover whole target time range
        if (
            not ts_target.contains(time_base)
            and not ts_input.periodic
            and not ts_target.periodic
        ):
            warn(
                "WARNING: ts_input (t = {} to {}) does not cover ".format(
                    ts_input.t_start, ts_input.t_stop
                )
                + "full time range of ts_target (t = {} to {})\n".format(
                    ts_target.t_start, ts_target.t_stop
                )
                + "Assuming input to be 0 outside of defined range.\n"
                + "If you are training by batches, check that the target signal is also split by batch.\n"
            )

        return None, target, time_base
