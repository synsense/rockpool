"""
Dynap-SE graph transformer package state machine implementation

* Non User Facing *
"""

from __future__ import annotations

from typing import Optional
from dataclasses import dataclass

__all__ = ["DFA_Placement"]


@dataclass
class DFA_Placement:
    """
    DFA_Placement defines an algorithmic state machine and keeps track of weight installation process
    """

    lif: bool = False
    """the bit that identifies lif layer can be processed at that step, defaults to False (linear state)"""

    rec: bool = False
    """the bit that identifies recurrent layer can be processed at that step, defaults to False (linear state)"""

    linear: bool = True
    """the bit that identifies linear layer can be processed at that step, defaults to True (linear state)"""

    def __eq__(self, __o: DFA_Placement) -> bool:
        """
        __eq__ overrides the equality operator

        :param __o: the object to be compared
        :type __o: DFA_Placement
        :return: True if all data fields are equal
        :rtype: bool
        """
        return self.lif == __o.lif and self.rec == __o.rec and self.linear == __o.linear

    def assign(self, __o: DFA_Placement) -> None:
        """
        assign equates an existing object and the class instance of interest

        :param __o: the external object
        :type __o: DFA_Placement
        """
        self.lif = __o.lif
        self.rec = __o.rec
        self.linear = __o.linear

    def next(self, flag_rec: Optional[bool] = None):
        """
        next handles the state transition depending on the current step and the inputs

        :param flag_rec: the recurrent layer flag, it branches the post-lift state, defaults to None
        :type flag_rec: Optional[bool], optional
        :raises ValueError: post-lif state requires recurrent flag input!
        :raises ValueError: Illegal State!
        """

        if self == self.state_linear():
            self.__pre_lif()

        elif self == self.state_pre_lif():
            self.__post_lif()

        elif self == self.state_post_lif():
            if flag_rec is None:
                raise ValueError("post-lif state requires recurrent flag input!")
            if flag_rec:
                self.__linear()
            else:
                self.__pre_lif()
        else:
            raise ValueError("Illegal State!")

    ### --- Hidden assignment methods --- ###
    def __pre_lif(self) -> None:
        self.assign(self.state_pre_lif())

    def __post_lif(self) -> None:
        self.assign(self.state_post_lif())

    def __linear(self) -> None:
        self.assign(self.state_linear())

    #### --- Define all the possible states --- ###
    @classmethod
    def state_pre_lif(cls) -> DFA_Placement:
        return DFA_Placement(lif=True, rec=False, linear=False)

    @classmethod
    def state_post_lif(cls) -> DFA_Placement:
        return DFA_Placement(lif=False, rec=True, linear=True)

    @classmethod
    def state_linear(cls) -> DFA_Placement:
        return DFA_Placement(lif=False, rec=False, linear=True)
