#!/usr/bin/env python

"""
    Description:
        Convert IMPUTE/hapmap2 format to hapmap3 format.

    Input:
        A prefix for the three files
            - binary haplotype matrix (pref.hap)
            - individuals (pref.hap.indv)
            - SNP legend (pref.legend)
"""

import sys
import os
import numpy as np

hapInfo = []  # "NA18597_A", ...
snpInfo = []  # ["rs4690249", "2906", "A" (0), "T" (1)], ...
INPUT_FORMAT = "HapMap2"


###############################################################################
def go(pref, out_fname, format):

    global hapInfo, snpInfo, INPUT_FORMAT
    INPUT_FORMAT = format

    # ############ read input files #############

    if INPUT_FORMAT == "HapMap2":
        # HapMap2 file names
        phased = pref + "_phased"
        sample = pref + "_sample.txt"
        legend = pref + "_legend.txt"
    elif INPUT_FORMAT == "IMPUTE":
        # IMPUTE file names
        phased = pref + ".hap"
        sample = pref + ".hap.indv"
        legend = pref + ".legend"

    # 1. read sample info
    with open(sample, 'r') as sample_f:
        for i, line in enumerate(sample_f):
            ind_id = line.rstrip().split()[0]
            hapInfo.append(ind_id + "_A")
            hapInfo.append(ind_id + "_B")

    # 2. read legend info
    with open(legend, 'r') as legend_f:
        for line in legend_f:
            if (
                (INPUT_FORMAT == "HapMap2" and line.startswith("rs\tposition"))
                or
                (INPUT_FORMAT == "IMPUTE" and line.startswith("ID pos"))
            ):
                continue

            snpInfo.append(line.rstrip().split())

    # 3. read haplotype info
    n_snps, hap_mat = None, []
    with open(phased, 'r') as phased_f:
        for i, line in enumerate(phased_f):
            hap = np.array(line.rstrip().split())
            hap_mat.append(hap)

            # make sure all haplotypes same legnth
            if n_snps is None:
                n_snps = len(hap)
            else:
                assert n_snps == len(hap), "Error: haplotypes should equal in legnth"

    # ############# create HapMap3 ##############
    outf = open(out_fname, 'w')
    outf.write("rsID position " + " ".join(hapInfo) + "\n")

    # create SNP matrix
    if INPUT_FORMAT == "HapMap2":
        # invert haplotype matrix
        snp_mat = np.column_stack(hap_mat)
    elif INPUT_FORMAT == "IMPUTE":
        # use SNP matrix as-is
        snp_mat = np.array(hap_mat)

    # enumerate rows/SNPs
    for i, snp_arr in enumerate(snp_mat):
        rsid, pos, zero, one = snpInfo[i]
        snp_arr_dna = [zero if snp == "0" else one for snp in snp_arr]  # {0,1} -> {A,C,G,T}
        outf.write(rsid + " " + pos + " " + " ".join(snp_arr_dna) + "\n")  # write snp row

    outf.close()

###############################################################################
#                                  MAIN
###############################################################################
if __name__ == '__main__':

    if(len(sys.argv) != 4):
        print ("\n\tusage: %s <pref> <out> <'HapMap2'/'IMPUTE'>\n"
               % (os.path.basename(sys.argv[0])))
        sys.exit(1)

    go(sys.argv[1], sys.argv[2], sys.argv[3])

    # Known sweeps, HapMap 2 -> 3
    # format = "HapMap2"
    # hm2_pref = ("/home/rronen/Dropbox/UCSD/workspace/SoftSweep/HapMap"
    #             "/hm2_ADH1B/genotypes_chr4_JPT+CHB_r22_nr.b36_fwd")
    # hm2_pref = ("/home/rronen/Dropbox/UCSD/workspace/SoftSweep/HapMap"
    #             "/hm2_EDAR/genotypes_chr2_JPT+CHB_r21_nr_fwd")
    # hm2_pref = ("/home/rronen/Dropbox/UCSD/workspace/SoftSweep/HapMap"
    #             "/hm2_LCT/genotypes_chr2_CEU_r21_nr_fwd")
    # hm2_pref = ("/home/rronen/Dropbox/UCSD/workspace/SoftSweep/HapMap"
    #             "/hm2_PSCA/genotypes_chr8_YRI_r21_nr_fwd")

    # outfname = "hm3_chr4_JPT+CHB_r22_nr.b36_fwd.phased"
    # outfname = "hm2_to_3_chr8_YRI_r21_nr_fwd.phased"

    # nonCMS sweeps, IMPUTE -> HapMap3
    # format = "IMPUTE"
    # hm2_pref = ("/home/rronen/Dropbox/UCSD/workspace/SoftSweep/Andean_HA"
    #             "/anp32d-10kb-pad.impute")
    # hm2_pref = ("/home/rronen/Dropbox/UCSD/workspace/SoftSweep/Andean_HA"
    #             "/senp1-10kb-pad.impute")
    # hm2_pref = ("/home/rronen/Dropbox/UCSD/workspace/SoftSweep/Andean_HA"
    #             "/joint-10kb-pad.impute")

    # outfname = "chr12.anp32d-10kb-pad.andean.b37.phased"
    # outfname = "chr12.senp1-10kb-pad.andean.b37.phased"
    # outfname = "chr12.joint-10kb-pad.andean.b37.phased"

    # go(hm2_pref, outfname, format)
