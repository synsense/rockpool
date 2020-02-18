"""
train_rr.py - Define class for training ridge regression. Can be used by various layers.
"""

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
        self.xty = np.zeros(
            (self.num_features + int(self.train_biases), self.num_outputs)
        )
        self.xtx = np.zeros(
            (
                self.num_features + int(self.train_biases),
                self.num_features + int(self.train_biases),
            )
        )
        self.kahan_comp_xty = np.zeros_like(self.xty)
        self.kahan_comp_xtx = np.zeros_like(self.xtx)

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

    def _relabel_fisher(self, target: np.ndarray) -> np.ndarray:
        """
        _relabel_fisher - Relabel target such that training is equivalent to
                          Fisher discriminant analysis.
        :param np.ndarray target:  2D-array (num_samples x num_outputs) of target data
        :return np.ndarray:        Relabeled target data.
        """

        num_timesteps = len(target)

        # - Relabel target based on number of occurences of corresponding data points
        bool_tgt = target.astype(bool)
        nums_true = np.sum(bool_tgt, axis=0)
        nums_false = num_timesteps - nums_true
        labels_true = num_timesteps / nums_true
        labels_false = -num_timesteps / nums_false
        target = target.astype(float)
        for i_tgt, (tgt_vec_bool, lbl_t, lbl_f) in enumerate(
            zip(bool_tgt.T, labels_true, labels_false)
        ):
            target[tgt_vec_bool, i_tgt] = lbl_t
            target[tgt_vec_bool == False, i_tgt] = lbl_f

        return target

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

        if self.standardize:
            inp = self.z_score_standardization(inp)

        if self.train_biases:
            # - Include constant values for training biases
            inp = np.hstack((inp, np.ones((len(inp), 1))))

        # - Fisher relabelling
        if self.fisher_relabelling:
            target = self._relabel_fisher(target)

        return inp, target

    def train_batch(self, inp: np.ndarray, target: np.ndarray, update_model=False):
        """
        train_batch - Update internal variables for one training batch
        :param np.ndarray inp:     Prepared input data in 2D-array (num_samples x num_features)
        :param np.ndarray target:  2D-array (num_samples x num_outputs) of prepared target data
        """
        inp, target = self._prepare_data(inp, target)
        upd_xty = inp.T @ target - self.kahan_comp_xty
        upd_xtx = inp.T @ inp - self.kahan_comp_xtx
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

    def update_model(self):
        """
        update_model - Update model weights and biases based on current collected training data.
        """
        solution = np.linalg.solve(
            self.xtx
            + self.regularize * np.eye(self.num_features + int(self.train_biases)),
            self.xty,
        )
        if self.train_biases:
            self.weights = solution[:-1]
            self.bias = solution[-1]
        else:
            self.weights = solution

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
