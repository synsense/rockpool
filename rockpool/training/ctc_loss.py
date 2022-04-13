"""
Implementations of the CTC loss, in numpy, Jax and Torch
"""
import warnings

import numpy as np

from typing import Sequence


def ctc_loss_numpy(
    label: Sequence, log_prob: np.ndarray, seq_length: int, big_num: float = 1e10
) -> float:
    """
    Numpy implementation of the CTC loss

    Args:
        label (Sequence[int]): A sequence of actual label ids ``(Tlabels,)``
        log_prob (np.ndarray): A matrix ``(T, Nlabels)`` of log probabilities of observing each possible label out of ``Nlabels``, for each network output time-bin ``T``.
        seq_length (int): The length of the original sequence
        big_num (float): A large number used to represent a very low log-probability

    Returns:
        float: The calculated CTC sequence loss
    """
    label_ = [0, 0]
    l = len(label)
    for i in range(l):
        label_.append(int(label[i]))
        label_.append(0)

    l_ = 2 * l + 1
    aa = np.full((seq_length, l_ + 1), -big_num)
    aa[0][1] = log_prob[0][0]
    aa[0][2] = log_prob[0][label_[2]]

    for i in range(1, seq_length):
        aa[i][1] = aa[i - 1][1] + log_prob[i][0]
        aa[i][2] = np.logaddexp(aa[i - 1][2], aa[i - 1][1]) + log_prob[i][label_[2]]

        for j in range(3, l_ + 1):
            aa[i][j] = np.logaddexp(aa[i - 1][j], aa[i - 1][j - 1])

            if label_[j] != 0 and label_[j] != label_[j - 2]:
                aa[i][j] = np.logaddexp(aa[i][j], aa[i - 1][j - 2])

            aa[i][j] += log_prob[i][label_[j]]

    return -np.logaddexp(aa[seq_length - 1][l_], aa[seq_length - 1][l_ - 1])


try:
    import jax.numpy as jnp
    from jax.lax import scan

    def ctc_loss_jax(
        label: Sequence, log_prob: jnp.ndarray, seq_length: int, big_num: float = 1e10
    ) -> float:
        """
        Jax-compatible implementation of the CTC loss

        Args:
            label (Sequence[int]): A sequence of actual label ids ``(Tlabels,)``
            log_prob (np.ndarray): A matrix ``(T, Nlabels)`` of log probabilities of observing each possible label out of ``Nlabels``, for each network output time-bin ``T``.
            seq_length (int): The length of the original sequence
            big_num (float): A large number used to represent a very low log-probability

        Returns:
            float: The calculated CTC sequence loss
        """
        label_ = jnp.full(2 * len(label) + 2, 0)
        label_ = label_.at[2::2].set(label)

        # - Initialise `aa` matrix
        l_ = 2 * len(label) + 1

        # append a row zeros to log_prob
        log_prob_ = jnp.full([log_prob.shape[0] + 1, log_prob.shape[1]], -big_num)
        log_prob_ = log_prob_.at[:-1, :].set(log_prob)

        # aa = np.full((seq_length, l_ + 1), -big_num)
        aa_0 = jnp.full((l_ + 1,), -big_num)

        #     aa[0][1] = log_prob[0][0]
        #     aa[0][2] = log_prob[0][label_[2]]
        aa_0 = aa_0.at[1].set(log_prob_[0, 0])
        aa_0 = aa_0.at[2].set(log_prob_[0, label_[2]])

        # - Pre-index log prob
        log_prob_label = log_prob_[:, label_]

        #### Outer loop

        # for i in range(1, seq_length):
        def outer(carry, inputs) -> (jnp.ndarray, jnp.ndarray):
            (log_prob_i, log_prob_label_i) = inputs
            aa_isub1 = carry

            # - Initialise aa_ret_i
            aa_ret_i = jnp.full((l_ + 1,), -big_num)

            # aa[i][1] = aa[i - 1][1] + log_prob[i][0]
            # aa[i][2] = np.logaddexp(aa[i - 1][2], aa[i - 1][1]) + log_prob[i][label_[2]]
            aa_ret_i = aa_ret_i.at[1].set(aa_isub1[1] + log_prob_i[0])
            aa_ret_i = aa_ret_i.at[2].set(
                jnp.logaddexp(aa_isub1[2], aa_isub1[1]) + log_prob_label_i[2]
            )

            #### - Set up for inner loop
            # aa_isub1 = aa[i-1, :]
            # aa_isub1 = aa_isub1

            # log_prob_label_i = log_prob_label_i

            # for j in range(3, l_ + 1):
            def inner(_, inputs) -> (None, float):
                (
                    aa_isub1_j,
                    aa_isub1_jsub1,
                    aa_isub1_jsub2,
                    label__j,
                    label__jsub2,
                ) = inputs

                # aa[i][j] = np.logaddexp(aa[i - 1][j], aa[i - 1][j - 1])
                aa_ret_j = jnp.logaddexp(aa_isub1_j, aa_isub1_jsub1)

                # if label_[j] != 0 and label_[j] != label_[j - 2]:
                # aa[i][j] = np.logaddexp(aa[i][j], aa[i - 1][j - 2])

                cond = (label__j != 0) * (label__j != label__jsub2)
                aa_ret_j = jnp.logaddexp(aa_ret_j, aa_isub1_jsub2) * cond + aa_ret_j * (
                    1 - cond
                )

                # aa[i][j] += log_prob[i][label_[j]]
                # Performed outside loop

                return None, aa_ret_j

            # - Perform inner loop
            j = np.arange(3, l_ + 1)
            inputs = (
                aa_isub1[j],
                aa_isub1[j - 1],
                aa_isub1[j - 2],
                label_[j],
                label_[j - 2],
            )
            _, aa_ret_inner = scan(
                f=inner,
                xs=inputs,
                init=None,
            )

            aa_ret_inner += log_prob_label_i[3 : l_ + 1]

            #### - end inner loop

            # - Assign inner loop results to aa_ret_i
            aa_ret_i = aa_ret_i.at[3 : l_ + 1].set(aa_ret_inner)

            # return carry, outputs
            return aa_ret_i, aa_ret_i

        inputs = (log_prob_[1 : seq_length + 1], log_prob_label[1 : seq_length + 1])
        _, aa_final = scan(
            f=outer,
            xs=inputs,
            init=aa_0,
        )

        return -jnp.logaddexp(
            aa_final[seq_length - 2][l_], aa_final[seq_length - 2][l_ - 1]
        )

except (ModuleNotFoundError, ImportError) as err:
    warnings.warn(f"Could not import dependencies for ctc_loss_jax: {err}.")

    def ctc_loss_jax(
        label: Sequence, log_prob, seq_length: int, big_num: float = 1e10
    ) -> float:
        raise ImportError("Jax backend not available.")


try:
    import torch

    def ctc_loss_torch(
        label: Sequence, log_prob: torch.Tensor, seq_length: int, big_num: float = 1e10
    ) -> float:
        """
        Torch implementation of the CTC loss

        Args:
            label (Sequence[int]): A sequence of actual label ids ``(Tlabels,)``
            log_prob (np.ndarray): A matrix ``(T, Nlabels)`` of log probabilities of observing each possible label out of ``Nlabels``, for each network output time-bin ``T``.
            seq_length (int): The length of the original sequence
            big_num (float): A large number used to represent a very low log-probability

        Returns:
            float: The calculated CTC sequence loss
        """

        def my_log1p(x):
            if torch.isinf(x) and x > 0:
                return x
            else:
                u = 1.0 + x
                d = u - 1.0
                if d == 0:
                    return x
                else:
                    return torch.log(u) * x / d

        def my_log_add_exp(x, y):
            if x == y:
                return x + my_log1p(torch.Tensor([2]))
            tmp = x - y
            if tmp > 0:
                return x + my_log1p(torch.exp(-tmp))

            if tmp <= 0:
                return y + my_log1p(torch.exp(tmp))

            return tmp

        label_ = [0, 0]

        l = len(label)
        for i in range(l):
            label_.append(int(label[i]))
            label_.append(0)

        l_ = 2 * l + 1
        aa = torch.full((seq_length, l_ + 1), -big_num)
        aa[0][1] = log_prob[0][0]
        aa[0][2] = log_prob[0][label_[2]]
        for i in range(1, seq_length):
            aa[i][1] = aa[i - 1][1] + log_prob[i][0]
            aa[i][2] = (
                my_log_add_exp(aa[i - 1][2], aa[i - 1][1]) + log_prob[i][label_[2]]
            )
            for j in range(3, l_ + 1):
                aa[i][j] = my_log_add_exp(aa[i - 1][j], aa[i - 1][j - 1])
                if label_[j] != 0 and label_[j] != label_[j - 2]:
                    aa[i][j] = my_log_add_exp(aa[i][j], aa[i - 1][j - 2])
                aa[i][j] += log_prob[i][label_[j]]

        return -my_log_add_exp(aa[seq_length - 1][l_], aa[seq_length - 1][l_ - 1])

except (ModuleNotFoundError, ImportError) as err:
    warnings.warn(f"Could not import dependencies for ctc_loss_torch: {err}.")

    def ctc_loss_torch(
        label: Sequence, log_prob, seq_length: int, big_num: float = 1e10
    ) -> float:
        raise ImportError("Torch backend not available.")
