"""
Defines register names for Xylo™Audio 3 SYNS65302
"""

# ===========================================================================
# Description     : test file
# Author          : Bai, Xin
# Created On      : 2023-12-21 11:54:09
# Last Modified By: Bai, Xin
# Last Modified On: 2024-01-16 15:54:53
# ===========================================================================
# Copyright       : 2023 Shanghai SynSense Technologies Co., Ltd.
# ===========================================================================

# ===========================================================================
# register address
# ===========================================================================

version = 0x0000
ctrl1 = 0x0001
ctrl2 = 0x0002
ctrl3 = 0x0003
tr_wrap = 0x0004
hm_tr_wrap = 0x0005
clk_ctrl = 0x0006
clk_div = 0x0007
pwr_ctrl1 = 0x0008
pwr_ctrl2 = 0x0009
pwr_ctrl3 = 0x000A
pwr_ctrl4 = 0x000B
pad_ctrl = 0x000C
otp_ctrl = 0x000D
ie1 = 0x000E
ie2 = 0x000F
wo = 0x0010
out_ctrl = 0x0011
out_wrap = 0x0012
baddr = 0x0013
blen = 0x0014
ivgen_ctrl1 = 0x0015
ivgen_ctrl2 = 0x0016
ldo_ctrl1 = 0x0017
ldo_ctrl2 = 0x0018
afe_ctrl = 0x0019
adc_ctrl = 0x001A
dfe_ctrl = 0x001B
agc_ctrl1 = 0x001C
agc_ctrl2 = 0x001D
agc_ctrl3 = 0x001E
agc_aaf_a1 = 0x001F
agc_aaf_a2 = 0x0020
agc_aaf_a3 = 0x0021
agc_aaf_a4 = 0x0022
agc_aaf_b_reg0 = 0x0023
agc_aaf_b_reg1 = 0x0024
agc_wt0 = 0x0025
agc_wt1 = 0x0026
agc_wt2 = 0x0027
agc_wt3 = 0x0028
agc_wt4 = 0x0029
agc_wt5 = 0x002A
agc_wt6 = 0x002B
agc_wt7 = 0x002C
agc_wt8 = 0x002D
agc_wt9 = 0x002E
agc_wt10 = 0x002F
agc_wt11 = 0x0030
agc_wt12 = 0x0031
agc_wt13 = 0x0032
agc_wt14 = 0x0033
agc_wt15 = 0x0034
agc_at_reg0 = 0x0035
agc_at_reg1 = 0x0036
agc_at_reg2 = 0x0037
agc_at_reg3 = 0x0038
agc_at_reg4 = 0x0039
agc_at_reg5 = 0x003A
agc_at_reg6 = 0x003B
agc_at_reg7 = 0x003C
agc_pgiv_reg0 = 0x003D
agc_pgiv_reg1 = 0x003E
agc_ugrsa0_l = 0x003F
agc_ugrsa1_l = 0x0040
agc_ugrsa2_l = 0x0041
agc_ugrsa3_l = 0x0042
agc_ugrsa4_l = 0x0043
agc_ugrsa5_l = 0x0044
agc_ugrsa6_l = 0x0045
agc_ugrsa7_l = 0x0046
agc_ugrsa8_l = 0x0047
agc_ugrsa9_l = 0x0048
agc_ugrsa10_l = 0x0049
agc_ugrsa11_l = 0x004A
agc_ugrsa12_l = 0x004B
agc_ugrsa13_l = 0x004C
agc_ugrsa14_l = 0x004D
agc_ugrsa_h = 0x004E
agc_dgrsa0_l = 0x004F
agc_dgrsa1_l = 0x0050
agc_dgrsa2_l = 0x0051
agc_dgrsa3_l = 0x0052
agc_dgrsa4_l = 0x0053
agc_dgrsa5_l = 0x0054
agc_dgrsa6_l = 0x0055
agc_dgrsa7_l = 0x0056
agc_dgrsa8_l = 0x0057
agc_dgrsa9_l = 0x0058
agc_dgrsa10_l = 0x0059
agc_dgrsa11_l = 0x005A
agc_dgrsa12_l = 0x005B
agc_dgrsa13_l = 0x005C
agc_dgrsa14_l = 0x005D
agc_dgrsa_h = 0x005E
agc_ghri_l = 0x005F
vad_tw = 0x0060
hm_thr = 0x0061
hm_tw = 0x0062
bpf_bb_reg0 = 0x0063
bpf_bb_reg1 = 0x0064
bpf_bwf_reg0 = 0x0065
bpf_bwf_reg1 = 0x0066
bpf_baf_reg0 = 0x0067
bpf_baf_reg1 = 0x0068
bpf_a1_reg0 = 0x0069
bpf_a1_reg1 = 0x006A
bpf_a1_reg2 = 0x006B
bpf_a1_reg3 = 0x006C
bpf_a1_reg4 = 0x006D
bpf_a1_reg5 = 0x006E
bpf_a1_reg6 = 0x006F
bpf_a1_reg7 = 0x0070
bpf_a2_reg0 = 0x0071
bpf_a2_reg1 = 0x0072
bpf_a2_reg2 = 0x0073
bpf_a2_reg3 = 0x0074
bpf_a2_reg4 = 0x0075
bpf_a2_reg5 = 0x0076
bpf_a2_reg6 = 0x0077
bpf_a2_reg7 = 0x0078
iaf_thr0_l = 0x0079
iaf_thr0_h = 0x007A
iaf_thr1_l = 0x007B
iaf_thr1_h = 0x007C
iaf_thr2_l = 0x007D
iaf_thr2_h = 0x007E
iaf_thr3_l = 0x007F
iaf_thr3_h = 0x0080
iaf_thr4_l = 0x0081
iaf_thr4_h = 0x0082
iaf_thr5_l = 0x0083
iaf_thr5_h = 0x0084
iaf_thr6_l = 0x0085
iaf_thr6_h = 0x0086
iaf_thr7_l = 0x0087
iaf_thr7_h = 0x0088
iaf_thr8_l = 0x0089
iaf_thr8_h = 0x008A
iaf_thr9_l = 0x008B
iaf_thr9_h = 0x008C
iaf_thr10_l = 0x008D
iaf_thr10_h = 0x008E
iaf_thr11_l = 0x008F
iaf_thr11_h = 0x0090
iaf_thr12_l = 0x0091
iaf_thr12_h = 0x0092
iaf_thr13_l = 0x0093
iaf_thr13_h = 0x0094
iaf_thr14_l = 0x0095
iaf_thr14_h = 0x0096
iaf_thr15_l = 0x0097
iaf_thr15_h = 0x0098
dn_eps_reg0 = 0x0099
dn_eps_reg1 = 0x009A
dn_eps_reg2 = 0x009B
dn_eps_reg3 = 0x009C
dn_eps_reg4 = 0x009D
dn_eps_reg5 = 0x009E
dn_eps_reg6 = 0x009F
dn_eps_reg7 = 0x00A0
dn_b_reg0 = 0x00A1
dn_b_reg1 = 0x00A2
dn_b_reg2 = 0x00A3
dn_b_reg3 = 0x00A4
dn_b_reg4 = 0x00A5
dn_b_reg5 = 0x00A6
dn_b_reg6 = 0x00A7
dn_b_reg7 = 0x00A8
dn_k1_reg0 = 0x00A9
dn_k1_reg1 = 0x00AA
dn_k1_reg2 = 0x00AB
dn_k1_reg3 = 0x00AC
dn_k1_reg4 = 0x00AD
dn_k1_reg5 = 0x00AE
dn_k1_reg6 = 0x00AF
dn_k1_reg7 = 0x00B0
dn_k2_reg0 = 0x00B1
dn_k2_reg1 = 0x00B2
dn_k2_reg2 = 0x00B3
dn_k2_reg3 = 0x00B4
dn_k2_reg4 = 0x00B5
dn_k2_reg5 = 0x00B6
dn_k2_reg6 = 0x00B7
dn_k2_reg7 = 0x00B8
pdm_h_reg0 = 0x00B9
pdm_h_reg1 = 0x00BA
pdm_h_reg2 = 0x00BB
pdm_h_reg3 = 0x00BC
pdm_h_reg4 = 0x00BD
pdm_h_reg5 = 0x00BE
pdm_h_reg6 = 0x00BF
pdm_h_reg7 = 0x00C0
pdm_h_reg8 = 0x00C1
pdm_h_reg9 = 0x00C2
pdm_h_reg10 = 0x00C3
pdm_h_reg11 = 0x00C4
pdm_h_reg12 = 0x00C5
pdm_h_reg13 = 0x00C6
pdm_h_reg14 = 0x00C7
pdm_h_reg15 = 0x00C8
pdm_h_reg16 = 0x00C9
pdm_h_reg17 = 0x00CA
pdm_h_reg18 = 0x00CB
pdm_h_reg19 = 0x00CC
pdm_h_reg20 = 0x00CD
pdm_h_reg21 = 0x00CE
pdm_h_reg22 = 0x00CF
pdm_h_reg23 = 0x00D0
pdm_h_reg24 = 0x00D1
pdm_h_reg25 = 0x00D2
pdm_h_reg26 = 0x00D3
pdm_h_reg27 = 0x00D4
pdm_h_reg28 = 0x00D5
pdm_h_reg29 = 0x00D6
pdm_h_reg30 = 0x00D7
pdm_h_reg31 = 0x00D8
pdm_h_reg32 = 0x00D9
pdm_h_reg33 = 0x00DA
pdm_h_reg34 = 0x00DB
pdm_h_reg35 = 0x00DC
pdm_h_reg36 = 0x00DD
pdm_h_reg37 = 0x00DE
pdm_h_reg38 = 0x00DF
pdm_h_reg39 = 0x00E0
pdm_h_reg40 = 0x00E1
pdm_h_reg41 = 0x00E2
pdm_h_reg42 = 0x00E3
pdm_h_reg43 = 0x00E4
pdm_h_reg44 = 0x00E5
pdm_h_reg45 = 0x00E6
pdm_h_reg46 = 0x00E7
pdm_h_reg47 = 0x00E8
pdm_h_reg48 = 0x00E9
pdm_h_reg49 = 0x00EA
pdm_h_reg50 = 0x00EB
pdm_h_reg51 = 0x00EC
pdm_h_reg52 = 0x00ED
pdm_h_reg53 = 0x00EE
pdm_h_reg54 = 0x00EF
pdm_h_reg55 = 0x00F0
pdm_h_reg56 = 0x00F1
pdm_h_reg57 = 0x00F2
pdm_h_reg58 = 0x00F3
pdm_h_reg59 = 0x00F4
pdm_h_reg60 = 0x00F5
pdm_h_reg61 = 0x00F6
pdm_h_reg62 = 0x00F7
pdm_h_reg63 = 0x00F8
pdm_h_reg64 = 0x00F9
pdm_h_reg65 = 0x00FA
pdm_h_reg66 = 0x00FB
pdm_h_reg67 = 0x00FC
pdm_h_reg68 = 0x00FD
pdm_h_reg69 = 0x00FE
pdm_h_reg70 = 0x00FF
pdm_h_reg71 = 0x0100
pdm_h_reg72 = 0x0101
pdm_h_reg73 = 0x0102
pdm_h_reg74 = 0x0103
pdm_h_reg75 = 0x0104
pdm_h_reg76 = 0x0105
pdm_h_reg77 = 0x0106
pdm_h_reg78 = 0x0107
pdm_h_reg79 = 0x0108
pdm_h_reg80 = 0x0109
pdm_h_reg81 = 0x010A
pdm_h_reg82 = 0x010B
pdm_h_reg83 = 0x010C
pdm_h_reg84 = 0x010D
pdm_h_reg85 = 0x010E
pdm_h_reg86 = 0x010F
pdm_h_reg87 = 0x0110
pdm_h_reg88 = 0x0111
pdm_h_reg89 = 0x0112
pdm_h_reg90 = 0x0113
pdm_h_reg91 = 0x0114
pdm_h_reg92 = 0x0115
pdm_h_reg93 = 0x0116
pdm_h_reg94 = 0x0117
pdm_h_reg95 = 0x0118
pdm_h_reg96 = 0x0119
pdm_h_reg97 = 0x011A
pdm_h_reg98 = 0x011B
pdm_h_reg99 = 0x011C
pdm_h_reg100 = 0x011D
pdm_h_reg101 = 0x011E
pdm_h_reg102 = 0x011F
pdm_h_reg103 = 0x0120
pdm_h_reg104 = 0x0121
pdm_h_reg105 = 0x0122
pdm_h_reg106 = 0x0123
pdm_h_reg107 = 0x0124
pdm_h_reg108 = 0x0125
pdm_h_reg109 = 0x0126
pdm_h_reg110 = 0x0127
pdm_h_reg111 = 0x0128
pdm_h_reg112 = 0x0129
pdm_h_reg113 = 0x012A
pdm_h_reg114 = 0x012B
pdm_h_reg115 = 0x012C
pdm_h_reg116 = 0x012D
pdm_h_reg117 = 0x012E
pdm_h_reg118 = 0x012F
pdm_h_reg119 = 0x0130
pdm_h_reg120 = 0x0131
pdm_h_reg121 = 0x0132
pdm_h_reg122 = 0x0133
pdm_h_reg123 = 0x0134
pdm_h_reg124 = 0x0135
pdm_h_reg125 = 0x0136
pdm_h_reg126 = 0x0137
pdm_h_reg127 = 0x0138
pdm_h_reg128 = 0x0139
pdm_h_reg129 = 0x013A
pdm_h_reg130 = 0x013B
pdm_h_reg131 = 0x013C
pdm_h_reg132 = 0x013D
pdm_h_reg133 = 0x013E
pdm_h_reg134 = 0x013F
pdm_h_reg135 = 0x0140
pdm_h_reg136 = 0x0141
pdm_h_reg137 = 0x0142
pdm_h_reg138 = 0x0143
pdm_h_reg139 = 0x0144
pdm_h_reg140 = 0x0145
pdm_h_reg141 = 0x0146
pdm_h_reg142 = 0x0147
pdm_h_reg143 = 0x0148
ispkreg0l = 0x0149
ispkreg0h = 0x014A
ispkreg1l = 0x014B
ispkreg1h = 0x014C
id_reg_1 = 0x014D
id_reg_2 = 0x014E
id_reg_3 = 0x014F
die_loc = 0x0150
id_rsvd_reg = 0x0151
stat1 = 0x0152
stat2 = 0x0153
int1 = 0x0154
int2 = 0x0155
omp_stat0 = 0x0156
omp_stat1 = 0x0157
omp_stat2 = 0x0158
omp_stat3 = 0x0159
omp_stat4 = 0x015A
omp_stat5 = 0x015B
omp_stat6 = 0x015C
omp_stat7 = 0x015D
omp_stat8 = 0x015E
omp_stat9 = 0x015F
omp_stat10 = 0x0160
omp_stat11 = 0x0161
omp_stat12 = 0x0162
omp_stat13 = 0x0163
omp_stat14 = 0x0164
omp_stat15 = 0x0165
monsel = 0x0166
mon_grp_sel = 0x0167
dbg_ctrl1 = 0x0168
ana_testmode_ctrl1 = 0x0169
ana_testmode_ctrl2 = 0x016A
ana_testmode_ctrl3 = 0x016B
ana_testmode_ctrl4 = 0x016C
ana_testmode_ctrl5 = 0x016D
adc_test_ctrl = 0x016E
tram_ctrl = 0x016F
hram_ctrl = 0x0170
dbg_stat1 = 0x0171
cntr_stat = 0x0172

# ===========================================================================
# register field position
# ===========================================================================

# ctrl1
ctrl1__man__pos = 0
ctrl1__isyn2_en__pos = 1
ctrl1__alias_en__pos = 2
ctrl1__bias_en__pos = 3
ctrl1__iwbs__pos = 4
ctrl1__hwbs__pos = 8
ctrl1__owbs__pos = 12
ctrl1__mem_clk_on__pos = 16
ctrl1__ram_active__pos = 17
ctrl1__keep_int__pos = 20
ctrl1__always_update_omp_stat__pos = 22
ctrl1__hm_en__pos = 23

# ctrl2
ctrl2__inc__pos = 0
ctrl2__ienc__pos = 8
ctrl2__hnc__pos = 16

# ctrl3
ctrl3__onc__pos = 0
ctrl3__oenc__pos = 8

# clk_ctrl
clk_ctrl__i2c__pos = 0
clk_ctrl__spi__pos = 1
clk_ctrl__adc__pos = 2
clk_ctrl__otp__pos = 3
clk_ctrl__sdm__pos = 4
clk_ctrl__spk2saer__pos = 5
clk_ctrl__ana_testmode_ctrl__pos = 16
clk_ctrl__sadc_if__pos = 17

# clk_div

clk_div__i2c__pos_msb = 4
clk_div__i2c__pos_lsb = 0
clk_div__adc__pos_msb = 10
clk_div__adc__pos_lsb = 8
clk_div__sdm__pos_msb = 21
clk_div__sdm__pos_lsb = 16
clk_div__spk2saer__pos_msb = 27
clk_div__spk2saer__pos_lsb = 24

# stat2
stat2__pd__pos = 0

# pad_ctrl
pad_ctrl__ctrl0__pos_msb = 3
pad_ctrl__ctrl0__pos_lsb = 0
pad_ctrl__ctrl1__pos_msb = 7
pad_ctrl__ctrl1__pos_lsb = 4
pad_ctrl__ctrl2__pos_msb = 11
pad_ctrl__ctrl2__pos_lsb = 8
pad_ctrl__ctrl3__pos_msb = 15
pad_ctrl__ctrl3__pos_lsb = 12
pad_ctrl__ctrl4__pos_msb = 19
pad_ctrl__ctrl4__pos_lsb = 16
pad_ctrl__ctrl5__pos_msb = 23
pad_ctrl__ctrl5__pos_lsb = 20

# out_ctrl
out_ctrl__mode__pos = 0
out_ctrl__out0_en__pos = 4
out_ctrl__out1_en__pos = 5
out_ctrl__out2_en__pos = 6

# wo
wo__srst__pos = 0
wo__tr_trig__pos = 4

# dfe_ctrl
dfe_ctrl__bpf_en__pos = 0
dfe_ctrl__iaf_en__pos = 1
dfe_ctrl__dn_en__pos = 2
dfe_ctrl__hm_en__pos = 3
dfe_ctrl__mic_if_sel_en__pos = 4
dfe_ctrl__global_thr__pos = 5
dfe_ctrl__pdm_clk_dir_en__pos = 8
dfe_ctrl__pdm_clk_edge_en__pos = 9
dfe_ctrl__bfi_en__pos = 10
dfe_ctrl__adc_bit_en__pos_msb = 29
dfe_ctrl__adc_bit_en__pos_lsb = 16

# monsel
monsel__monsel0__pos = 0
monsel__monsel1__pos = 8
monsel__monsel2__pos = 16
monsel__monsel3__pos = 24

# mon_grp_sel
# -----------

mon_grp_sel__pe_sel__pos_msb = 1
mon_grp_sel__pe_sel__pos_lsb = 0
mon_grp_sel__pe_fsm_sel__pos_msb = 6
mon_grp_sel__pe_fsm_sel__pos_lsb = 4

# dbg_ctrl1
dbg_ctrl1__monsig_reg__pos = 8
dbg_ctrl1__spk_src_sel__pos = 18
dbg_ctrl1__stif_sel__pos = 19
dbg_ctrl1__dbg_sta_upd_en__pos = 29
dbg_ctrl1__mon_en__pos = 30
