"""
Dynap-SE1 samna alias definitions. Ensures consistency over samna and rockpool

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
08/04/2022
"""
from enum import Enum


class Dynapse1SynType(Enum, int):
    NMDA = 2
    AMPA = 3
    GABA_B = 0
    GABA_A = 1
