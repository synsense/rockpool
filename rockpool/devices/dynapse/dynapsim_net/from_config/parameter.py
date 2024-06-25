"""
Dynap-SE2 from_config parameter composition helper, serves as a developer tool

* Non User Facing *
"""

from __future__ import annotations

from typing import Dict, List, Tuple
from dataclasses import dataclass

import numpy as np

from rockpool.typehints import FloatVector

from rockpool.devices.dynapse.samna_alias import Dendrite, Dynapse2Configuration
from rockpool.devices.dynapse.quantization import WeightHandler
from rockpool.devices.dynapse.parameters import DynapSimCurrents
from rockpool.devices.dynapse.lookup import CHIP_POS

from .memory import MemorySE2


@dataclass
class ParameterHandler:
    """
    ParameterHandler leads the simulated network configuration by holding and processing the connectivity scheme
    """

    weights_in: np.ndarray
    """input (virtual -> hardware) weight matrix (integer masks)"""

    dendrites_in: np.ndarray
    """input (virtual -> hardware) connection gates (AMPA, GABA, NMDA, SHUNT)"""

    weights_rec: np.ndarray
    """recurrent (hardware -> hardware) weight matrix"""

    dendrites_rec: np.ndarray
    """recurrent (hardware -> hardware) connection gates"""

    core_map: FloatVector
    """the mapping between neuron index to respective core ID (chip, core)"""

    def __post_init__(self) -> None:
        """
        __post_init__ applies after initialization controls

        :raises ValueError: Input and recurrent weight shapes incompatible!
        :raises ValueError: Input weight shape does not match with dendrite shape!
        :raises ValueError: Recurrent weight shape does not match with the dendrite shape!
        """

        # Make sure that all matrices are numpy arrays
        self.weights_in = np.array(self.weights_in)
        self.dendrites_in = np.array(self.dendrites_in)
        self.weights_rec = np.array(self.weights_rec)
        self.dendrites_rec = np.array(self.dendrites_rec)

        # Check if the number of neurons are the same
        if self.weights_in.any() and self.weights_rec.any():
            if self.weights_in.shape[1] != self.weights_rec.shape[1]:
                raise ValueError("Input and recurrent weight shapes incompatible!")

        # Get the number of physical neurons indicated by the weight matrices
        if self.weights_in.any():
            n_rec = self.weights_in.shape[1]
        elif self.weights_rec.any():
            n_rec = self.weights_rec.shape[1]
        else:
            n_rec = 0

        # Check if the weight matrix shapes are compatible with the dendrites
        if self.weights_in.any():
            if self.weights_in.shape != self.dendrites_in.shape:
                raise ValueError(
                    "Input weight shape does not match with dendrite shape!"
                )

        if self.weights_rec.any():
            if self.weights_rec.shape != self.dendrites_rec.shape:
                raise ValueError(
                    "Recurrent weight shape does not match with the dendrite shape!"
                )

        # Compute each column's dendritic score (see ``class.dendrite_score``)
        self.score_list = [self.dendrite_score(n) for n in range(n_rec)]

    @classmethod
    def from_config(
        cls, config: Dynapse2Configuration, chip_pos: Dict[int, Tuple[int]] = CHIP_POS
    ) -> ParameterHandler:
        """
        from_config is a class factory method which construct the parameter handler object by processing a samna configuration object

        :param config: a samna configuration object used to configure all the system level properties
        :type config: Dynapse2Configuration
        :param chip_pos: global chip position dictionary (chip_id : (xpos,ypos)), defaults to CHIP_POS
        :type chip_pos: Dict[int, Tuple[int]], optional
        :return: a parameter handler object which leads a simulation network construction
        :rtype: ParameterHandler
        """
        mem = MemorySE2(chip_pos)
        spec = mem.spec_from_config(config)
        handler = cls(
            spec["weights_in"],
            spec["dendrites_in"],
            spec["weights_rec"],
            spec["dendrites_rec"],
            spec["core_map"],
        )
        return handler

    def dendrite_score(self, post_neuron_id: int) -> Dict[Dendrite, int]:
        """
        dendrite_score computes the sum of weights of dendrites on post-synaptic neuron
        It generates a dictionary having the total weight on each dendrite
            i.e {"AMPA":3, "GABA":4}
        This dictionary is used for computing a reasonable dendrite specific currents by computing a weighted avarage.
        For example if {"AMPA":3, "GABA":4}, then
            Itau_syn = (Itau_ampa * 3 + Itau_gaba * 4) / 7

        :param post_neuron_id: the column id that the score is computed.
        :type post_neuron_id: int
        :return: the scores (sum of weights) of the all represented dendrites on a specific column
        :rtype: Dict[Dendrite, int]
        """

        def stack(w1: np.ndarray, w2: np.ndarray) -> np.ndarray:
            """stack concatanes two columns obtained from the two matrices considering that the matrices are allowed to be empty"""
            in_col = w1[:, post_neuron_id] if w1.any() else np.array([])
            rec_col = w2[:, post_neuron_id] if w2.any() else np.array([])
            return np.hstack((in_col, rec_col))

        # select and merge the same post-synaptic columns
        weight_column = stack(self.weights_in, self.weights_rec)
        dendrite_column = stack(self.dendrites_in, self.dendrites_rec)

        # get the unique set of represented dendrites except for none
        dendrite_set = set(dendrite_column) - set([0])
        scores = {}

        # get the prominance scores of each dendrites
        for dendrite in dendrite_set:
            score = np.sum(weight_column[dendrite_column == dendrite])
            scores[dendrite] = score
        return scores

    def __compose_syn_currents(
        self,
        dscore: Dict[Dendrite, int],
        Iampa: float,
        Igaba: float,
        Inmda: float,
        Ishunt: float,
    ) -> float:
        """
        __compose_syn_currents applies weighted avarage to the dendrite specific synaptic currents

        :param dscore: the dendritic score dictionary for the neuron (see ``class.dendrite_score()``)
        :type dscore: Dict[Dendrite, int]
        :param Iampa: AMPA dendrite related current
        :type Iampa: float
        :param Igaba: GABA dendrite related current
        :type Igaba: float
        :param Inmda: NMDA dendrite related current
        :type Inmda: float
        :param Ishunt: SHUNT dendrite related current
        :type Ishunt: float
        :raises ValueError: Dendrite is not recognized!
        :return: a weighted average of dendrite specific currents
        :rtype: float
        """

        gross_sum = 0
        gross_score = 0

        for dendrite, score in dscore.items():
            if dendrite == Dendrite.ampa:
                gross_sum += score * Iampa
            elif dendrite == Dendrite.gaba:
                gross_sum += score * Igaba
            elif dendrite == Dendrite.nmda:
                gross_sum += score * Inmda
            elif dendrite == Dendrite.shunt:
                gross_sum += score * Ishunt
            else:
                raise ValueError("Dendrite is not recognized!")

            gross_score += score

        gross_score = 1 if (gross_score == 0) else gross_score
        return gross_sum / gross_score

    def compose_Igain_syn(
        self, currents: DynapSimCurrents, post_neuron_id: int
    ) -> float:
        """
        compose_Igain_syn composes the gain currents

        :param currents: an object encapsulating the common simulation currents
        :type currents: DynapSimCurrents
        :param post_neuron_id: the post-synaptic neuron id (column idx)
        :type post_neuron_id: int
        :return: the weighted average of dendrite specific Igain currents
        :rtype: float
        """
        return self.__compose_syn_currents(
            dscore=self.score_list[post_neuron_id],
            Iampa=currents.Igain_ampa,
            Igaba=currents.Igain_gaba,
            Inmda=currents.Igain_nmda,
            Ishunt=currents.Igain_shunt,
        )

    def compose_Itau_syn(
        self, currents: DynapSimCurrents, post_neuron_id: int
    ) -> float:
        """
        compose_Itau_syn composes the synaptic tau currents

        :param currents: an object encapsulating the common simulation currents
        :type currents: DynapSimCurrents
        :param post_neuron_id: the post-synaptic neuron id (column idx)
        :type post_neuron_id: int
        :return: the weighted average of dendrite specific Itau currents
        :rtype: float
        """
        return self.__compose_syn_currents(
            dscore=self.score_list[post_neuron_id],
            Iampa=currents.Itau_ampa,
            Igaba=currents.Itau_gaba,
            Inmda=currents.Itau_nmda,
            Ishunt=currents.Itau_shunt,
        )

    def __sign_mask(self, dendrites: np.ndarray) -> np.ndarray:
        """
        __sign_mask processes the dendrites matrix and generates a sign mask placing -1 for inhibitory dendrites, and 1 for excitatory dendrites

        :param dendrites: a matrix of dendrites (AMPA, GABA, NMDA, SHUNT, NONE)
        :type dendrites: np.ndarray
        :return: the +- signs of the weight values, + means excitatory; - means inhibitory
        :rtype: np.ndarray
        """
        if not dendrites.any():
            return np.empty_like(dendrites)

        sign = np.zeros_like(dendrites)
        sign[dendrites == Dendrite.ampa] = 1
        sign[dendrites == Dendrite.gaba] = -1
        sign[dendrites == Dendrite.nmda] = 1
        sign[dendrites == Dendrite.shunt] = -1
        return sign

    def __scaled_weights(
        self,
        weights: np.ndarray,
        signs: np.ndarray,
        Iw_trace: List[np.ndarray],
        Iscale: float,
        n_bits: int = 4,
    ) -> np.ndarray:
        """
        __scaled_weights uses the Iw currents of the respective neurons to restore the whole 4-bit integer weight matrix
        as a matrix of weight currents. Since each simulated neuron can belong to a different core, each column has to
        be computed seperately using the respective neuron's core parameters.
        The Iscale parameter scales the current weight matrix to a reasonable SNN matrix

        :param weights: the 4-bit integer weights (stored in CAMs)
        :type weights: np.ndarray
        :param signs: the +- signs of the weight values, + means excitatory; - means inhibitory
        :type signs: np.ndarray
        :param Iw_trace: the weight bit currents of the each neuron
        :type Iw_trace: List[np.ndarray]
        :param Iscale: the scaling current
        :type Iscale: float
        :param n_bits: number of bits allocated per weight, defaults to 4
        :type n_bits: int, optional
        :return: a weight matrix storing the current value of each connection
        :rtype: np.ndarray
        """
        w_cols = []

        # Compute each column separately! Each neuron can have a different Iw_0, Iw_1, Iw_2, Iw_3 settting.s
        for i, Iw in enumerate(Iw_trace):
            w_cols.append(
                WeightHandler.restore_weight_matrix(
                    n_bits=n_bits,
                    code=Iw,
                    int_mask=weights[:, i],
                    sign_mask=signs[:, i],
                )
            )

        # Restore the shape and scale of the weight matrix
        w_shaped = np.array(w_cols).T
        w_scaled = w_shaped / Iscale
        return w_scaled

    def get_scaled_weights_in(
        self, Iw_trace: List[np.ndarray], Iscale: float, n_bits: int = 4
    ) -> np.ndarray:
        """
        get_scaled_weights_in returns the scaled and restored input weights
        (Equivalent to a weight matrix reconstructed after quantization)

        :param Iw_trace: the weight bit currents of the each neuron
        :type Iw_trace: List[np.ndarray]
        :param Iscale: the scaling current
        :type Iscale: float
        :param n_bits: number of bits allocated per weight, defaults to 4
        :type n_bits: int, optional
        :return: _description_
        :rtype: np.ndarray
        """
        return self.__scaled_weights(
            self.weights_in, self.get_sign_in(), Iw_trace, Iscale, n_bits
        )

    def get_scaled_weights_rec(
        self, Iw_trace: List[np.ndarray], Iscale: float, n_bits: int = 4
    ) -> np.ndarray:
        """
        get_scaled_weights_rec returns the scaled and restored recurrent weights
        (Equivalent to a weight matrix reconstructed after quantization)

        :param Iw_trace: the weight bit currents of the each neuron
        :type Iw_trace: List[np.ndarray]
        :param Iscale: the scaling current
        :type Iscale: float
        :param n_bits: number of bits allocated per weight, defaults to 4
        :type n_bits: int, optional
        :return: _description_
        :rtype: np.ndarray
        """
        return self.__scaled_weights(
            self.weights_rec, self.get_sign_rec(), Iw_trace, Iscale, n_bits
        )

    def get_sign_in(self) -> np.ndarray:
        """get_sign_in returns the input sign mask (see ``class.__sign_mask``)
        the +- signs of the weight values, + means excitatory; - means inhibitory"""
        return self.__sign_mask(self.dendrites_in)

    def get_sign_rec(self) -> np.ndarray:
        """get_sign_rec returns the recurrent sign mask (see ``class.__sign_mask``)
        the +- signs of the weight values, + means excitatory; - means inhibitory"""
        return self.__sign_mask(self.dendrites_rec)

    def __decode_core_list(self, hash: int) -> int:
        """__decode_core_list is a wrapper for ``MemorySE2.decode_hash`` using tuple length as 2 for core address extraction"""
        return MemorySE2.decode_hash(hash, tuple_length=2)

    @property
    def core_list(self) -> List[Tuple[int]]:
        """core_list returns list of unique core addresses from the core_map"""
        core_list = set(map(MemorySE2.address_hash, self.core_map))
        core_list = list(map(self.__decode_core_list, core_list))
        return core_list

    @property
    def n_in(self) -> int:
        """n_in returns the number of input (virtual) neurons"""
        if self.weights_in.any():
            return self.weights_in.shape[0]
        elif self.weights_rec.any():
            return self.weights_rec.shape[0]
        else:
            return 0

    @property
    def n_rec(self) -> int:
        """n_rec returns the number of recurrent (hardware) neurons"""
        return len(self.core_map)
