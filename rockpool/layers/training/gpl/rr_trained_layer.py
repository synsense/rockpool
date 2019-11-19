"""
rr_trained_layer.py - Define a super class that layers can inherit from if they
                      should be trained with ridge regression
"""

### --- Import statements

# - Built-ins
from abc import ABC, abstractmethod
from typing import Union, Dict
from warnings import warn

# - Local imports
from ....timeseries import TSEvent, TSContinuous
from .train_rr import RidgeRegrTrainer


class RRTrainedLayer(ABC):
    @abstractmethod
    def __init__(self, size_in: int):
        self.size_in = size_in

    def train_rr(
        self,
        ts_target: TSContinuous,
        ts_input: TSEvent = None,
        regularize: float = 0,
        is_first: bool = True,
        is_last: bool = False,
        store_states: bool = True,
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
        :regularize:             float - regularization for ridge regression
        :is_first:               bool - True if current batch is the first in training
        :is_last:                bool - True if current batch is the last in training
        :store_states:           bool - Include last state from previous training and store state from this
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
            ts_target=ts_target,
            ts_input=ts_input,
            is_first=is_first,
            is_last=is_last,
            store_states=store_states,
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

        tr_data = self._batch_update(
            inp=inp,
            target=target,
            is_last=is_last,
            train_biases=train_biases,
            standardize=standardize,
            calc_intermediate_results=calc_intermediate_results,
            return_trained_output=return_trained_output,
            return_training_progress=return_training_progress,
        )

        if return_trained_output:
            output_samples = inp @ self.trainer.weights + self.trainer.bias
            tr_data["output"] = TSContinuous(time_base, output_samples)

        if store_states and return_training_progress:
            tr_data["trainig_progress"]["training_state"] = self._training_state

        if return_trained_output or return_training_progress:
            return tr_data

    def _batch_update(
        self,
        inp,
        target,
        is_last,
        train_biases,
        standardize,
        calc_intermediate_results,
        return_trained_output,
        return_training_progress,
    ):

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

        if calc_intermediate_results or return_trained_output or is_last:
            self.trainer.update_model()
            self.weights = self.trainer.weights
            if train_biases:
                self.bias = self.trainer.bias
            if return_training_progress:
                training_data["trainig_progress"]["weights"] = self.trainer.weights
                if train_biases:
                    training_data["trainig_progress"]["biases"] = self.trainer.bias

        if is_last:
            self.trainer.reset()

        return training_data

    @abstractmethod
    def _prepare_training_data(self):
        pass
