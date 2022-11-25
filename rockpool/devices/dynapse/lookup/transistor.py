"""
Dynap-SE2 circuit parameter transistor locations

Obtained from 
https://hardware.basket.office.synsense.ai/documentation/dynapse2docs/Sections/input_interface_config.html
for more accurate simulation of DynapSE-2 bias generator

Project Owner : Dylan Muir, SynSense AG
Author : Ugurcan Cakal
E-mail : ugurcan.cakal@gmail.com
14/01/2022
"""

__all__ = ["transistor_se2"]

transistor_se2 = {
    "LBWR_VB_P": "(s.5)",
    "SOIF_GAIN_N": "(s3)",
    "SOIF_LEAK_N": "(5.6x6)",
    "SOIF_REFR_N": "(s3)",
    "SOIF_DC_P": "(s2)",
    "SOIF_SPKTHR_P": "(s3)",
    "SOIF_CC_N": "(2x1)",
    "SOAD_PWTAU_N": "(s2)",
    "SOAD_GAIN_P": "(1x2)",
    "SOAD_TAU_P": "(3x1)",
    "SOAD_W_N": "(s2)",
    "SOAD_CASC_P": "(s2)",
    "SOCA_W_N": "(s2)",
    "SOCA_GAIN_P": "(1x2)",
    "SOCA_TAU_P": "(3x1)",
    "SOHO_VB_P": "(.5x2)",
    "SOHO_VH_P": "(.5x2)",
    "SOHO_VREF_P": "()",
    "SOHO_VREF_L_P": "(.5x2)",
    "SOHO_VREF_H_P": "(.5x2)",
    "SOHO_VREF_M_P": "(.5x2)",
    "DEAM_ETAU_P": "(3x1)",
    "DEAM_EGAIN_P": "(1x2)",
    "DEAM_ITAU_P": "(3x1)",
    "DEAM_IGAIN_P": "(1x2)",
    "DENM_ETAU_P": "(3x1)",
    "DENM_EGAIN_P": "(1x2)",
    "DENM_ITAU_P": "(3x1)",
    "DENM_IGAIN_P": "(1x2)",
    "DEGA_IGAIN_P": "(1x2)",
    "DEGA_ITAU_P": "(3x1)",
    "DESC_IGAIN_P": "(1x2)",
    "DESC_ITAU_P": "(3x1)",
    "DENM_NMREV_N": "(s3)",
    "DEAM_VRES_P": "(s2)",
    "DEAM_HRES_P": "(s2)",
    "DEAM_NRES_P": "(s2)",
    "SYSA_VRES_N": "(s.3)",
    "SYSA_VB_P": "(s.3)",
    "SYPD_EXT_N": "(s2)",
    "SYPD_DYL0_P": "(2x1)",
    "SYPD_DYL1_P": "(2x1)",
    "SYPD_DYL2_P": "(s.25)",
    "SYAM_W0_P": "(1.92x1)",
    "SYAM_W1_P": "(1.92x1)",
    "SYAM_W2_P": "(1.92x1)",
    "SYAM_W3_P": "(1.92x1)",
    "SYAW_STDSTR_N": "(2x.3)",
    "SYAM_STDW_N": "(1.92x1)",
    "NCCF_CAL_REFBIAS_V": "(N s2.5 P 2.5x7.5)",
    "NCCF_PWLK_P": "(1x3)",
    "NCCF_HYS_P": "(.5x2)",
    "NCCF_BIAS_P": "(.5x2)",
    "NCCF_CAL_OFFBIAS_P": "(N s2.5: P 2.5x7.5)",
    "SYAM_STDWAMPB": "(.25x20)",
    "R2R_BUFFER_CCB": "(.25x1)",
}
