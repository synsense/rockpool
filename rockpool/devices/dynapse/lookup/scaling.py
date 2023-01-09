"""
Both Dynap-SE2 scalig factor tables

Dynap-SE2 circuit bias generator transistor current response scaling factors.

Multiply the theoretical value obtained with the table values considering the parameter name.

:math:`I_{scaled} = I * \\text{scale_factor}`

Obtained from 
https://hardware.basket.office.synsense.ai/documentation/dynapse2docs/Sections/input_interface_config.html
for more accurate simulation of DynapSE-2 bias generator

* Non User Facing *
"""

__all__ = ["scale_factor_se2"]

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
