"""
Dynap-SE2 simulator parameter names and their reciprocals in SE2 device configurations

* Non User Facing *
"""

__all__ = ["sim2device_se2", "device2sim_se2"]


sim2device_se2 = {
    "Idc": "SOIF_DC_P",
    "If_nmda": "DENM_NMREV_N",
    "Igain_ahp": "SOAD_GAIN_P",
    "Igain_ampa": "DEAM_EGAIN_P",
    "Igain_gaba": "DEGA_IGAIN_P",
    "Igain_nmda": "DENM_EGAIN_P",
    "Igain_shunt": "DESC_IGAIN_P",
    "Igain_mem": "SOIF_GAIN_N",
    "Ipulse_ahp": "SOAD_PWTAU_N",
    "Ipulse": "SYPD_EXT_N",
    "Iref": "SOIF_REFR_N",
    "Ispkthr": "SOIF_SPKTHR_P",
    "Itau_ahp": "SOAD_TAU_P",
    "Itau_ampa": "DEAM_ETAU_P",
    "Itau_gaba": "DEGA_ITAU_P",
    "Itau_nmda": "DENM_ETAU_P",
    "Itau_shunt": "DESC_ITAU_P",
    "Itau_mem": "SOIF_LEAK_N",
    "Iw_0": "SYAM_W0_P",
    "Iw_1": "SYAM_W1_P",
    "Iw_2": "SYAM_W2_P",
    "Iw_3": "SYAM_W3_P",
    "Iw_ahp": "SOAD_W_N",
}

device2sim_se2 = {v: k for k, v in sim2device_se2.items()}
