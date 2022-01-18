"""
DynapSE simulator parameter names and their reciprocals in SE1 and SE2 device configurations
0th column is for simulator, 1st column for SE1 2nd column is for SE2
The 0th column is actually a dummy column and makes it easier to reach the bias parameter name with the version number
table[Itau_ahp][1] -> SE1 bias parameter name
table[Itau_ahp][2] -> SE2 bias parameter name

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
18/01/2022
"""


table = {
    "Itau_ahp": ["Itau_ahp", "IF_AHTAU_N", None],
    "Ith_ahp": ["Ith_ahp", "IF_AHTHR_N", None],
    "Iw_ahp": ["Iw_ahp", "IF_AHW_P", None],
    "Idc": ["Idc", "IF_DC_P", None],
    "If_nmda": ["If_nmda", "IF_NMDA_N", None],
    "Iref": ["Iref", "IF_RFR_N", None],
    "Itau_mem": ["Itau_mem", "IF_TAU1_N", None],
    "Itau2_mem": ["Itau2_mem", "IF_TAU2_N", None],
    "Ith_mem": ["Ith_mem", "IF_THR_N", None],
    "Itau_ampa": ["Itau_ampa", "NPDPIE_TAU_F_P", None],
    "Itau_nmda": ["Itau_nmda", "NPDPIE_TAU_S_P", None],
    "Ith_ampa": ["Ith_ampa", "NPDPIE_THR_F_P", None],
    "Ith_nmda": ["Ith_nmda", "NPDPIE_THR_S_P", None],
    "Itau_gaba_a": ["Itau_gaba_a", "NPDPII_TAU_F_P", None],
    "Itau_gaba_b": ["Itau_gaba_b", "NPDPII_TAU_S_P", None],
    "Ith_gaba_a": ["Ith_gaba_a", "NPDPII_THR_F_P", None],
    "Ith_gaba_b": ["Ith_gaba_b", "NPDPII_THR_S_P", None],
    "Iw_3": ["Iw_3", "PS_WEIGHT_EXC_F_N", "SYAM_W3_P"],  # AMPA SE1
    "Iw_2": ["Iw_2", "PS_WEIGHT_EXC_S_N", "SYAM_W2_P"],  # NMDA SE1
    "Iw_1": ["Iw_1", "PS_WEIGHT_INH_F_N", "SYAM_W1_P"],  # GABA_A SE1
    "Iw_0": ["Iw_0", "PS_WEIGHT_INH_S_N", "SYAM_W0_P"],  # GABA_B SE1
    "Ipulse": ["Ipulse", "PULSE_PWLK_P", None],
}
