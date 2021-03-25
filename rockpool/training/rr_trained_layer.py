"""
rr_trained_layer.py - Define a super class that layers can inherit from if they
                      should be trained with ridge regression
"""

### --- Import statements

raise ImportError("This module needs to be ported to the v2 API")

# - Built-ins
from abc import ABC, abstractmethod
from typing import Union, Dict, Optional
from warnings import warn

# - Third party packages
import numpy as np

# - Local imports
from rockpool.timeseries import TSEvent, TSContinuous, TimeSeries
from rockpool.nn.layers.layer import Layer
from rockpool.training.train_rr import RidgeRegrTrainer


class RRTrainedLayer(Layer, ABC):
    """
    Base class that defines methods for training with ridge regression. Subclasses can inherit from this class to provide ridge regression functionality.

    :Usage:

    When writing a new layer class, simply inherit from `.RRTrainedLayer` instead of from `.Layer`. Subclasses must provide a concrete implementation of the the `._prepare_training_data` abstract method. See the documentation for that method below, to understand how this can be implemented. `.RRTrainedLayer` provides an implementation that can be called with :py:func:`super`.

    This class provides the `.train_rr` method, which performs ridge regression training over multiple batches, called independently for each batch.

    `.RRTrainedLayer` also provides the `._batch_update` private method


    """

    def train_rr(
        self,
        ts_target: TSContinuous,
        ts_input: Optional[Union[TSEvent, TSContinuous]] = None,
        regularize: float = 0,
        is_first: bool = True,
        is_last: bool = False,
        train_biases: bool = True,
        calc_intermediate_results: bool = False,
        return_training_progress: bool = True,
        return_trained_output: bool = False,
        fisher_relabelling: bool = False,
        standardize: bool = False,
        n_prune: int = 0,
    ) -> Union[Dict, None]:
        """
        Train this layer with ridge regression over one of possibly many batches. Use Kahan summation to reduce rounding errors when adding data to existing matrices from previous batches.

        :param TSContinuous ts_target:                      Target signal for current batch
        :param Optional[TimeSeries] ts_input:               Input to layer for current batch. Default: ``None``, no input for this batch
        :param float regularize:                  Regularization parameter for ridge regression. Default: ``0``, no regularization
        :param bool is_first:                     Set to ``True`` if current batch is the first in training. Default: ``True``, initialise training with this batch as the first batch
        :param bool is_last:                      Set to ``True`` if current batch is the last in training. This has the same effect as if data from both trainings were presented at once.
        :param bool train_biases:                 If ``True``, train biases as if they were weights. Otherwise present biases will be ignored in training and not be changed. Default: ``True``, train biases as well as weights
        :param bool calc_intermediate_results:    If ``True``, calculates the intermediate weights not in the final batch. Default: ``False``, do not compute intermediate weights
        :param bool return_training_progress:     If ``True``, return dict of current training variables for each batch. Default: ``True``, return training progress
        :param bool return_trained_output:        If ``True``, return the result of evolving the layer with the trained weights in the output dict. Default: ``False``, do not return the trained output
        :param bool fisher_relabelling:           If ``True``, relabel target data such that the training algorithm is equivalent to Fisher discriminant analysis. Default: ``False``, use standard ridge / linear regression
        :param bool standardize:                  Train with z-score standardized data, based on means and standard deviations from first batch. Default: ``False``, do not standardize data
        :param n_prune: int                       Number of coefficients to prune.

        :return:
            If ``return_training_progress`` is ``True``, return a dict with current training variables (xtx, xty, kahan_comp_xtx, kahan_comp_xty).
            Weights and biases are returned if ``is_last`` is ``True`` or if ``calc_intermediate_results`` is ``True``.
            If ``return_trained_output`` is ``True``, the dict contains the output of evolving the layer with the newly trained weights.
        """
        inp, target, time_base = self._prepare_training_data(
            ts_target=ts_target, ts_input=ts_input, is_first=is_first, is_last=is_last
        )

        if is_first:
            # - Generate trainer object
            self.trainer = RidgeRegrTrainer(
                num_features=self.size_in,
                num_outputs=self.size_out,
                regularize=regularize,
                fisher_relabelling=fisher_relabelling,
                standardize=standardize,
                train_biases=train_biases,
            )
        else:
            # - Make sure that training parameters are consistent
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

        # - Update the training for this batch
        tr_data = self._batch_update(
            inp=inp,
            target=target,
            reset=is_last,
            train_biases=train_biases,
            standardize=standardize,
            update_weights=update_weights,
            return_training_progress=return_training_progress,
            n_prune=n_prune,
        )

        if return_trained_output:

            bias = self.trainer.bias if train_biases else self.bias

            output_samples = inp @ self.trainer.weights + bias
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
        n_prune: int = 0,
    ) -> Dict:
        """
        Train with the already processed input and target data of the current batch. Update layer weights and biases if requested. Provide information on training state if requested.

        :param np.ndarray inp:                  2D-array (``num_samples`` x ``num_features``) of input data.
        :param np.ndarray target:               2D-array (``num_samples`` x ``self.size``) of target data.
        :param bool reset:                      If ``True``, internal variables will be reset at the end.
        :param bool train_bises:                Should biases be trained or only weights?
        :param bool standardize:                Has input data been z-score standardized?
        :param bool update_weights:             Set ``True`` to update layer weights and biases.
        :param bool return_training_progress:   Return intermediate training data (e.g. ``xtx``, ``xty``,...)
        :param n_prune: int                     Number of coefficients to prune in this step.

        :return dict:                           Dict with information on training progress, depending on values of other function arguments.
        """
        self.trainer.train_batch(inp, target)

        training_data = dict()

        if return_training_progress:
            training_data["training_progress"] = dict(
                xtx=self.trainer.xtx,
                xty=self.trainer.xty,
                kahan_comp_xtx=self.trainer.kahan_comp_xtx,
                kahan_comp_xty=self.trainer.kahan_comp_xty,
            )
            if standardize:
                training_data["training_progress"]["inp_mean"] = self.trainer.inp_mean
                training_data["training_progress"]["inp_std"] = self.trainer.inp_std

        if update_weights:
            self.trainer.update_model(n_prune=n_prune)
            self.weights = self.trainer.weights

            if train_biases:
                self.bias = self.trainer.bias

            if return_training_progress:
                training_data["training_progress"]["weights"] = self.trainer.weights
                if train_biases:
                    training_data["training_progress"]["biases"] = self.trainer.bias

        if reset:
            self.trainer.reset()

        return training_data

    @abstractmethod
    def _prepare_training_data(
        self,
        ts_target: TSContinuous,
        ts_input: TimeSeries,
        is_first: bool,
        is_last: bool,
    ) -> (Union[None, np.ndarray], np.ndarray, np.ndarray):
        """
        Template for preparation of training data. Length of data is determined, dimensions are verified and target data is extracted from `ts_target` argument. Can be used in child classes through ``super()._prepare_training_data``. Extraction of input data needs to be implemented in child classes.

        :param TSContinuous ts_target:  Target time series
        :param TimeSeries ts_input:     Input time series
        :param bool is_first:           Set ``True`` if batch is first of training.
        :param bool is_last:            Set ``True`` if batch is last of training.

        :return (input, target, time_base):
            input np.ndarray:           Should be extracted input data. This abstract method returns ``None``
            target np.ndarray:          Extracted target data
            time_base np.ndarray:       Time base for the input and target data

        :Usage:

        Child classes must implement this method to be instantiated. However, the abstract method provided here performs several checks and useful functions::

            # - In the child class, call the superclass method
            __, target, time_base = super()._prepare_training_data(ts_target, ts_input, is_first, is_last)

            # ... perform input extraction from ``ts_input`` here in child class
        """
        # - Discrete time steps for evaluating input and target time series
        #   If `is_last`, include final sample
        num_timesteps = int(np.round(ts_target.duration / self.dt)) + int(is_last)
        time_base = self._gen_time_trace(ts_target.t_start, num_timesteps)

        # - Make sure time_base does not exceed ts_target
        time_base = time_base[time_base <= ts_target.t_stop]

        # - Prepare target data
        target = ts_target(time_base)

        # - Make sure no nan is in target, as this causes learning to fail
        if np.isnan(target).any():
            raise ValueError(
                self.start_print
                + "'nan' values have been found in target "
                + f"(where: {np.where(np.isnan(target))})"
            )

        # - Check target dimensions
        if target.ndim == 1 and self.size == 1:
            target = target.reshape(-1, 1)

        if target.shape[-1] != self.size:
            raise ValueError(
                self.start_print
                + f"Target dimensions ({target.shape[-1]}) does not match "
                + f"layer size ({self.size})"
            )

        # - Warn if input time range does not cover whole target time range
        if (
            not ts_input.contains(time_base)
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
