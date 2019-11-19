"""
train_rr.py - Define class for training ridge regression. Can be used by various layers.
"""

import numpy as np


class RidgeRegrTrainer:
    def __init__(
        self,
        num_features: int,
        num_outputs: int,
        regularization: float,
        fisher_relabelling: bool,
        standardize: bool,
        train_biases: bool,
    ):
        self.num_features = num_features
        self.num_outputs = num_outputs
        self.regularization = regularization
        self.fisher_relabelling = fisher_relabelling
        self.standardize = standardize
        self.train_biases = train_biases
        self.init_matrices()

    def init_matrices(self):
        self.xty = np.zeros((self.num_features + 1, self.num_outputs))
        self.xtx = np.zeros(
            (self.num_features + 1, self.num_features + int(self.train_biases))
        )
        self.kahan_comp_xty = np.zeros_like(self.xty)
        self.kahan_comp_xtx = np.zeros_like(self.xtx)

    def determine_z_score_params(self, inp):
        """
        determine_z_score_params - For each feature, find its mean and standard
                                   deviation and store them internally
        """
        self.inp_mean = np.mean(axis=0).reshape(-1, 1)
        self.inp_std = np.std(axis=0).reshape(-1, 1)

    def z_score_standardization(self, inp):
        """
        z_score_standardization - For each feature subtract `self.inp_mean` and
                                  scale it with `self.inp_std`
        """
        return (inp - self.mean) / self.inp_std

    def _relabel_fisher(self, target):
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

    def _prepare_data(self, inp, target):
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

    def train_batch(self, inp, target, update_model=False):

        inp, target = self._prepare_data(inp, target)
        print(inp)
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

    def fit(self, inp, target):
        self.init_matrices()
        self.train_batch(inp, target, update_model=True)
        self.reset()

    def update_model(self):
        solution = np.linalg.solve(
            self.xtx
            + self.regularization * np.eye(self.num_features + int(self.train_biases)),
            self.xty,
        )
        if self.train_biases:
            self.weights = solution[:-1]
            self.bias = solution[-1]
        else:
            self.weights = solution

        if self.standardize:
            self.weights /= self.inp_std
            if self.train_biases:
                self.bias -= self.inp_mean @ self.weights

    def reset(self):
        self.init_matrices()
        if self.standardize:
            del self.inp_mean
            del self.inp_std
