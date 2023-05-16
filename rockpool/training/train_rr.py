"""
train_rr.py - Define class for training ridge regression. Can be used by various layers.
"""

raise ImportError("This module needs to be ported to the v2 API")

from typing import Tuple
import numpy as np


class RidgeRegrTrainer:
    """
    RidgeRegrTrainer - Class to perform ridge regression.
    """

    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        regularize: float,
        fisher_relabelling: bool,
        standardize: bool,
        train_biases: bool,
    ):
        """
        RidgeRegrTrainer - Class to perform ridge regression.
        :param int num_features:        Number of input features.
        :param int num_outputs:         Number of output units to be trained.
        :param float regularize:        Regularization parameter.
        :param bool fisher_relabelling: Relabel target data such that algorithm is equivalent to Fisher discriminant analysis.
                                        Warning: When training over multiple batches, fisher_relabelling should only be True if the
                                                 distribution of targets is approximately the same across all batches. Otherwise
                                                 some data points of the same class can have a stronger effect on the output
                                                 than others.
        :param bool standardize:        Perform z-score standardization based on mean and variance of first input batch.
        :param bool train_biases:       Train constant biases along with weights.
        """
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.regularize = regularize
        self.fisher_relabelling = fisher_relabelling
        self.standardize = standardize
        self.train_biases = train_biases
        self.init_matrices()

    def init_matrices(self):
        """
        init_matrices - Initialize matrices for storing intermediate training data.
        """

        # - Effective number of features, including one for the biases if they are trained
        num_ftrs_effective = self.num_features + int(self.train_biases)
        if self.fisher_relabelling:
            xtx_shape = (self.num_outputs, num_ftrs_effective, num_ftrs_effective)
        else:
            xtx_shape = (num_ftrs_effective, num_ftrs_effective)

        # - Array for holding features and target
        self.xty = np.zeros((num_ftrs_effective, self.num_outputs))

        # - Array for holding features
        self.xtx = np.zeros(xtx_shape)

        # - Arrays to be used for Kahan summation (against rounding errors)
        self.kahan_comp_xty = np.zeros_like(self.xty)
        self.kahan_comp_xtx = np.zeros_like(self.xtx)

        # - Mask which is applied to the input data to zero out previously pruned coefficients
        self.prune_mask = np.ones((self.num_features, self.num_outputs))

    def determine_z_score_params(self, inp: np.ndarray):
        """
        determine_z_score_params - For each feature, find its mean and standard
                                   deviation and store them internally
        :param np.ndarray inp:     Input data in 2D-array (num_samples x num_features)
        """
        self.inp_mean = np.mean(inp, axis=0).reshape(1, -1)
        self.inp_std = np.std(inp, axis=0).reshape(1, -1)
        # - Set standard deviation to 1 wherever it is zero, to avoid division by zero
        self.inp_std[self.inp_std == 0] = 1

    def z_score_standardization(self, inp: np.ndarray) -> np.ndarray:
        """
        z_score_standardization - For each feature subtract `self.inp_mean` and scale it
                                  with `self.inp_std`. If these parameters have not yet
                                  been defined, use 'self.determine_z_score_params to
                                  find them based on 'inp'.
        :param np.ndarray inp:    Input data in 2D-array (num_samples x num_features)
        :return np.ndarray:       Standardized input data.
        """
        try:
            return (inp - self.inp_mean) / self.inp_std
        except AttributeError:
            self.determine_z_score_params(inp)
            return (inp - self.inp_mean) / self.inp_std

    def _relabel_fisher(
        self, inp: np.ndarray, target: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        _relabel_fisher - Relabel target and input such that training is equivalent to
                          Fisher discriminant analysis. Input will have additional axis
                          to have individual relabelled input for each output dimension.
        :param np.ndarray inp:     2D-array (num_samples x num_features) of input data
                                   Convention for target values:
                                       1: Class is present (True)
                                       0: Class is not present (False)
                                      -1: Ignore this datapoint. This is particularly
                                          useful for all-vs-all classifiers, where
                                          the target class is not within the corresponding
                                          pair of classes.
        :param np.ndarray target:  2D-array (num_samples x num_outputs) of target data
        :return np.ndarray:        Relabeled input data (num_outputs x num_samples x num_samples).
        :return np.ndarray:        Relabeled target data (num_samples x num_outputs).
        """

        # - Weight samples by relative number of occurences for each class
        int_tgt = np.round(target).astype(int)
        nums_true = np.sum(int_tgt == 1, axis=0)
        nums_false = np.sum(int_tgt == 0, axis=0)
        nums_total = np.shape(int_tgt)[0]
        weights_true = nums_total / nums_true
        weights_false = nums_total / nums_false

        # - New target vector
        target = target.astype(float)
        # - Extended input vector with weighted samples for each output dimension
        inp = np.expand_dims(inp, axis=0)
        inp_extd = np.repeat(inp, repeats=self.num_outputs, axis=0)

        # - Apply weights
        for i_tgt, (tgt_vec, w_t, w_f) in enumerate(
            zip(int_tgt.T, weights_true, weights_false)
        ):
            target[tgt_vec == 1, i_tgt] = w_t
            target[tgt_vec == 0, i_tgt] = -w_f
            inp_extd[i_tgt, tgt_vec == 1] *= w_t
            inp_extd[i_tgt, tgt_vec == 0] *= w_f
            # Points with target -1 get weight 0, to be ignored during training
            target[tgt_vec == -1, i_tgt] = 0
            inp_extd[i_tgt, tgt_vec == -1] = 0

        return inp_extd, target

    def _prepare_data(
        self, inp: np.ndarray, target: np.ndarray
    ) -> (np.ndarray, np.ndarray):
        """
        _prepare_data - Prepare input and target data by verifying shapes and,
                        if required, standardization and Fisher relabelling.
        :param np.ndarray inp:     Input data in 2D-array (num_samples x num_features)
        :param np.ndarray target:  2D-array (num_samples x num_outputs) of target data

        :return np.ndarray:        Processed input data.
        :return np.ndarray:        Processed target data.
        """
        if inp.shape[1] != self.num_features:
            raise ValueError(
                f"RidgeRegrTrainer: Number of columns in `inp` must be {self.num_features}."
            )
        if target.shape[1] != self.num_outputs:
            raise ValueError(
                f"RidgeRegrTrainer: Number of columns in `target` must be {self.num_outputs}."
            )
        if inp.shape[0] != target.shape[0]:
            raise ValueError(
                "RidgeRegrTrainer: `inp` and `target` must have same number of data points"
                + f" (`inp` has {inp.shape[0]}, `target` has {target.shape[0]})."
            )

        # apply pruning mask to input
        # prune away only those neurons which have zero weight to all target neurons
        inp = inp * self.prune_mask.max(axis=1)

        if self.standardize:
            inp = self.z_score_standardization(inp)

        if self.train_biases:
            # - Include constant values for training biases
            inp = np.hstack((inp, np.ones((len(inp), 1))))

        # - Fisher relabelling
        if self.fisher_relabelling:
            inp_extd, target = self._relabel_fisher(inp, target)
            return (inp, inp_extd), target
        else:
            return inp, target

    def train_batch(self, inp: np.ndarray, target: np.ndarray, update_model=False):
        """
        train_batch - Update internal variables for one training batch
        :param np.ndarray inp:     Prepared input data in 2D-array (num_samples x num_features)
        :param np.ndarray target:  2D-array (num_samples x num_outputs) of prepared target data
        """
        if self.fisher_relabelling:
            (inp, inp_extd), target = self._prepare_data(inp, target)
            upd_xtx = inp.T @ inp_extd - self.kahan_comp_xtx
        else:
            inp, target = self._prepare_data(inp, target)
            upd_xtx = inp.T @ inp - self.kahan_comp_xtx
        upd_xty = inp.T @ target - self.kahan_comp_xty
        xty_new = self.xty + upd_xty
        xtx_new = self.xtx + upd_xtx

        self.kahan_comp_xty = (xty_new - self.xty) - upd_xty
        self.kahan_comp_xtx = (xtx_new - self.xtx) - upd_xtx
        self.xty += upd_xty
        self.xtx += upd_xtx

        if update_model:
            self.update_model()

    def fit(self, inp: np.ndarray, target: np.ndarray):
        """
        fit - Train with one single batch
        :param np.ndarray inp:     Prepared input data in 2D-array (num_samples x num_features)
        :param np.ndarray target:  2D-array (num_samples x num_outputs) of prepared target data
        """
        self.init_matrices()
        self.train_batch(inp, target, update_model=True)
        self.reset()

    def solve(self, xtx, xty, prune=False):
        """
        solve - Choses approximate solver among np.linalg.solve and np.linalg.inv for finding coefficients.
        """
        if not prune:
            # use np.linalg.solve as first choice solution
            return np.linalg.solve(xtx, xty)
        else:
            # use np.linalg.inv to explicitly calculate the inverse of XTX
            self.xtx_inv = np.linalg.inv(
                xtx
            )  # TODO does this really work in case of fisher relabeling?
            return np.matmul(self.xtx_inv, xty)

    def update_model(self, n_prune: int = 0):
        """
        update_model - Update model weights and biases based on current collected training data.
        """

        if self.fisher_relabelling and n_prune > 0:
            raise NotImplementedError(
                "Pruning is not implemented if fisher relabelling is enabled"
            )

        # - Matrix for regularization
        reg = self.regularize * np.eye(self.num_features + int(self.train_biases))

        if self.fisher_relabelling:
            solution = np.array(
                [
                    self.solve(xtx_this, xty_this, n_prune > 0)
                    for xtx_this, xty_this in zip(self.xtx + reg, self.xty.T)
                ]
            ).T
        else:
            solution = self.solve(self.xtx + reg, self.xty, n_prune > 0)

        if n_prune > 0:
            std_err_approx = np.sqrt(
                np.diagonal(self.xtx_inv)
            )  # should be actually sqrt(diag(XTX_inv) * residual)
            t_stat = (solution.T / std_err_approx).T
            if self.train_biases:
                t_stat = t_stat[:-1]

            tol = np.sort(np.abs(t_stat), axis=0)[n_prune]

            coefs_to_prune = np.where(t_stat < tol)
            self.prune_mask[coefs_to_prune] = 0

        if self.train_biases:
            # apply prune mask to weights
            self.weights = solution[:-1] * self.prune_mask
            self.bias = solution[-1]
        else:
            # apply prune mask to weights
            self.weights = solution * self.prune_mask

        if self.standardize:
            self.weights /= self.inp_std.T
            if self.train_biases:
                self.bias -= (self.inp_mean @ self.weights).ravel()

    def reset(self):
        """reset - Reset internal training data."""
        self.init_matrices()
        if self.standardize:
            del self.inp_mean
            del self.inp_std
