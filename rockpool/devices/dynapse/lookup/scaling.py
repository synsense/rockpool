"""
Both Dynap-SE1 and Dynap-SE2 scalig factor tables

Dynap-SE1 circuit bias generator transistor current response scaling factors
Neither experimental nor theoretical. Waiting for tests to be verified!

Dynap-SE2 circuit bias generator transistor current response scaling factors
Multiply the theoretical value obtained with the table values considering the parameter name
Obtained from 
https://hardware.basket.office.synsense.ai/documentation/dynapse2docs/Sections/input_interface_config.html
for more accurate simulation of DynapSE-2 bias generator

merged : scaling_factor_se1.py -> scaling_factor.py @ 220529

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
14/01/2022
"""

__all__ = ["scale_factor_se1", "scale_factor_se2"]

scale_factor_se1 = {
    "IF_AHTAU_N": 1.0,
    "IF_AHTHR_N": 1.0,
    "IF_AHW_P": 1.0,
    "IF_BUF_P": 1.0,
    "IF_CASC_N": 1.0,
    "IF_DC_P": 1.0,
    "IF_NMDA_N": 1.0,
    "IF_RFR_N": 1.0,
    "IF_TAU1_N": 1e-4,
    "IF_TAU2_N": 1.0,
    "IF_THR_N": 1.0,
    "NPDPIE_TAU_F_P": 1.0,
    "NPDPIE_TAU_S_P": 1.0,
    "NPDPIE_THR_F_P": 1.0,
    "NPDPIE_THR_S_P": 1.0,
    "NPDPII_TAU_F_P": 1.0,
    "NPDPII_TAU_S_P": 1.0,
    "NPDPII_THR_F_P": 1.0,
    "NPDPII_THR_S_P": 1.0,
    "PS_WEIGHT_EXC_F_N": 1.0,
    "PS_WEIGHT_EXC_S_N": 1.0,
    "PS_WEIGHT_INH_F_N": 1.0,
    "PS_WEIGHT_INH_S_N": 1.0,
    "PULSE_PWLK_P": 1.0,
    "R2R_P": 1.0,
}

scale_factor_se2 = {
    "LBWR_VB_P": 0.57,
    "SOIF_GAIN_N": 1.05,
    "SOIF_LEAK_N": 0.61,
    "SOIF_REFR_N": 1.05,
    "SOIF_DC_P": 0.38,
    "SOIF_SPKTHR_P": 0.38,
    "SOIF_CC_N": 0.66,
    "SOAD_PWTAU_N": 0.92,
    "SOAD_GAIN_P": 0.68,
    "SOAD_TAU_P": 0.2,
    "SOAD_W_N": 0.92,
    "SOAD_CASC_P": 0.385,
    "SOCA_W_N": 0.92,
    "SOCA_GAIN_P": 0.68,
    "SOCA_TAU_P": 0.2,
    "SOHO_VB_P": 1.47,
    "SOHO_VH_P": 1.47,
    "SOHO_VREF_P": 0.68,
    "SOHO_VREF_L_P": 1.47,
    "SOHO_VREF_H_P": 1.47,
    "SOHO_VREF_M_P": 1.47,
    "DEAM_ETAU_P": 0.2,
    "DEAM_EGAIN_P": 0.68,
    "DEAM_ITAU_P": 0.2,
    "DEAM_IGAIN_P": 0.68,
    "DENM_ETAU_P": 0.2,
    "DENM_EGAIN_P": 0.68,
    "DENM_ITAU_P": 0.2,
    "DENM_IGAIN_P": 0.68,
    "DEGA_IGAIN_P": 0.68,
    "DEGA_ITAU_P": 0.2,
    "DESC_IGAIN_P": 0.68,
    "DESC_ITAU_P": 0.2,
    "DENM_NMREV_N": 1.05,
    "DEAM_VRES_P": 0.385,
    "DEAM_HRES_P": 0.385,
    "DEAM_NRES_P": 0.385,
    "SYSA_VRES_N": 1.9,
    "SYSA_VB_P": 1.9,
    "SYPD_EXT_N": 0.92,
    "SYPD_DYL0_P": 0.22,
    "SYPD_DYL1_P": 0.22,
    "SYPD_DYL2_P": 1.6,
    "SYAM_W0_P": 0.22,
    "SYAM_W1_P": 0.22,
    "SYAM_W2_P": 0.22,
    "SYAM_W3_P": 0.22,
    "SYAW_STDSTR_N": 0.2,
    "SYAM_STDW_N": 0.67,
    "NCCF_CAL_REFBIAS_V": 1.0,
    "NCCF_PWLK_P": 0.89,
    "NCCF_HYS_P": 1.47,
    "NCCF_BIAS_P": 1.47,
    "NCCF_CAL_OFFBIAS_P": 1.0,
    "SYAM_STDWCCB": 1.7,
    "R2R_BUFFER_CCB": 1.7,
}
