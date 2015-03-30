#!/usr/bin/env python

import sys
import os
import operator
import bisect
import matplotlib
import brewer2mpl
import numpy as np
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt

from itertools import izip
from matplotlib import rcParams
from matplotlib import gridspec
from scipy import stats
from collections import Counter

''' local imports '''
import cfp_score as cfp
import hfs_utils as hfs
import hap_reader_ms as hap_read

# for regression plots
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/real_sweeps_CMS"
JITTER_FRAC = 0  # 0.01

# general settings
DEBUG = True
DISCARD_NO_CHIMP = True
norm = 1
report_balanced_acc = True
hap_ids = []
chimp_alleles = {}  # chimp alleles
ref_alleles = {}  # reference alleles
radii = np.arange(10000, 2000001, 5000)  # radii to consider

# LCT, CEU
hm3_chr2_ceu = "hm3_LCT/hapmap3_r2_b36_fwd.consensus.qc.poly.chr2_ceu.phased"
hm2_chr2_ceu = "hm2_LCT/hm2_to_3_chr2_CEU_r21_nr_fwd.phased"

# TRPV6, CEU
hm3_chr7_ceu = "hm3_TRPV6/hapmap3_r2_b36_fwd.consensus.qc.poly.chr7_ceu.phased"

# PSCA, YRI, CHB or JPT+CHB
hm3_chr8_yri = "hm2_PSCA/hm2_to_3_chr8_YRI_r21_nr_fwd.phased"
hm3_chr8_chb = "hm3_PSCA/hapmap3_r2_b36_fwd.consensus.qc.poly.chr8_chb.unr.phased"
hm3_chr8_chb_jpt = "hm3_PSCA/hapmap3_r2_b36_fwd.consensus.qc.poly.chr8_jpt+chb.unr.phased"

# EDAR, CHB or JPT+CHB
hm2_chr2_chb = "hm2_EDAR/hm2_to_3_chr2_CHB_r21_nr_fwd.phased"
hm2_chr2_chb_jpt = "hm2_EDAR/hm2_to_3_chr2_JPT+CHB_r21_nr_fwd.phased"

# ADH1B, CHB or JPT+CHB
hm2_chr4_chb = "hm2_ADH1B/hm2_to_3_chr4_CHB_r22_nr.b36_fwd.phased"
hm2_chr4_chb_jpt = "hm2_ADH1B/hm2_to_3_chr4_JPT+CHB_r22_nr.b36_fwd.phased"

# SENP1, CMS & nonCMS
senp1_andean = "sweeps_hm3/chr12.senp1.b37.phased"
# senp1_andean = "sweeps_hm3/chr12.senp1-10kb-pad.b37.phased"
# senp1_andean = "sweeps_hm3/chr12.senp1-region.b37.phased"

# ANP32D, CMS & nonCMS
anp32d_andean = "sweeps_hm3/chr12.anp32d.b37.phased"
# anp32d_andean = "sweeps_hm3/chr12.anp32d-10kb-pad.b37.phased"
# anp32d_andean = "sweeps_hm3/chr12.anp32d-region.b37.phased"

# joint SENP1 and ANP32D in CMS & non-CMS
joint_andean = "sweeps_hm3/chr12.joint-10kb-pad.b37.phased"

# CMS & nonCMS sweeps
senp1_andean = "sweeps_hm3/SENP1.impute.b37.phased"
anp32d_andean = "sweeps_hm3/ANP32D.impute.b37.phased"
arid1b_andean = "sweeps_hm3/ARID1B.impute.b37.phased"
cd3e_andean = "sweeps_hm3/CD3E.impute.b37.phased"
cnnm1_andean = "sweeps_hm3/CNNM1.impute.b37.phased"
duox_andean = "sweeps_hm3/DUOX.impute.b37.phased"
gusbp4_andean = "sweeps_hm3/GUSBP4.impute.b37.phased"
pbx4_andean = "sweeps_hm3/PBX4.impute.b37.phased"
susd5_andean = "sweeps_hm3/SUSD5.impute.b37.phased"

# selective sweeps to plot
sweeps = {
    # known sweeps
    r"LCT":   (hm3_chr2_ceu, "rs4988235"),
    r"PSCA":  (hm3_chr8_yri, "rs2294008"),
    r"TRPV6": (hm3_chr7_ceu, "rs4987682"),
    r"ADH1B": (hm2_chr4_chb_jpt, "rs1229984"),
    r"EDAR":  (hm2_chr2_chb_jpt, "rs3827760"),

    # CMS & nonCMS sweeps
    # r"SENP1": (senp1_andean, None),
    # r"ANP32D": (anp32d_andean, None),
    # r"ARID1B": (arid1b_andean, None),
    # r"CD3E": (cd3e_andean, None),
    # r"CNNM1": (cnnm1_andean, None),
    # r"DUOX": (duox_andean, None),
    # r"GUSBP4": (gusbp4_andean, None),
    # r"PBX4": (pbx4_andean, None),
    # r"SUSD5": (susd5_andean, None),
    # r"SENP1+NP32D": (joint_andean, None),
}

# phenotype data table
# used when no knowledge of adaptive allele
phenotype_table_fpath = "cms_status.txt"
phenotypes = ["Dizz", "Physical weakness 1", "Mental Fatigue 1", "Anorex", "Musc Weak",
              "Joint", "Breathlessness 2", "Palpitations 2", "Sleep disturb",
              "Cyanosis of lips, face or fingers 2", "Injected Conjunctivae 2", "Dilat",
              "Paresth", "Head", "Tinn", "Hct", "Sat", "Hct-num", "Sat%", "TOTAL", "Blood", "Skin"]

phenotypes_to_regress = {
    "TOTAL": r"CMS \,\, score",
    # "Sat%": r"O_2 \,\, Sat",
    # "Hct-num": r"Hematocrit"
}

c1 = brewer2mpl.get_map('Set1',  'Qualitative', 9).mpl_colors


###############################################################################
def go_focal_site_unknown(sweep_name, hapmap_chr_file):
    ''' Run full CFP classification on given sweep '''
    global chimp_polar, no_polar

    print "==============================================="
    print "Working on (unknown focal site) sweep %s..." % sweep_name
    print "===============================================\n"

    # init allele polatization stats
    chimp_polar, no_polar = 0.0, 0.0

    # read haplotype matrix, and chimp-polarize alleles (updates polarization stats)
    hap_mat, col_freqs, ba_col, positions = read_phased_hapmap_chr(hapmap_chr_file, None)

    # clean haplotype matrix, removes all-0 columns (non-segragating)
    # including alleles for sites with no Chimp data, artificially set to all-0
    hap_mat, col_freqs, ba_col, positions = hap_read.remove_zero_cols(hap_mat,
                                                                      col_freqs,
                                                                      ba_col,
                                                                      positions)

    # sanity check
    assert len(hap_ids) == len(hap_mat), "Error: # haplotype IDs != # haplotypes"

    # read phenotypes
    pheno_df = pd.read_csv(phenotype_table_fpath, sep='\t', header=0)
    pheno_df = pheno_df.set_index('Sample ID')

    # create output files
    regress_file = open("regress_stats_%s.txt" % sweep_name, 'w')
    cfp_file = open("cfp_scores_variables_%s.txt" % sweep_name, 'w')

    # decide on 'center' site, and set radii
    center_col = find_center_with_offset(positions, offset=0)
    radii = np.arange(10000, (max(positions) - min(positions))/2.0, 5000)
    radii = [(max(positions) - min(positions))/2.0]

    # write file headers
    # regress_file.write("#radius\tr^2\tp-val\tstd-err\n")
    cfp_file.write("#indID\tmin(CFP1,CFP2)\tgroup\n")

    # iterate radii from center
    for rad in radii:

        # compute CFP scores
        left_idx, right_idx = get_boundary_idx(positions, center_col, rad)
        hap_scores = cfp.haplotype_CFP_scores(hap_mat[:, left_idx:right_idx],
                                              col_freqs[left_idx:right_idx], norm=norm)

        for pheno, pheno_lab in phenotypes_to_regress.iteritems():
            for f_name, f in {"min": np.min}.iteritems():  # "max": np.max, "mean": np.mean,

                # paired haplotype CFP scores -> individual CFP scores
                indv_scores = compound_hap_CFPs(zip(hap_scores[0::2], hap_scores[1::2]), c_func=f)

                # reorder phenotype values to match CFP scores
                ind_ids = [hid[:-2] for hid in hap_ids[0::2]]
                reord_pheno = np.array([pheno_df.xs(iid)[pheno] for iid in ind_ids])
                reord_group = np.array([pheno_df.xs(iid)["Group"] for iid in ind_ids])

                # regress on phenotypes
                # slope, intercept, r_val, p_val, std_e = stats.linregress(reord_pheno, indv_scores)
                # mod = sm.OLS(reord_pheno, sm.add_constant(indv_scores, prepend=False))
                # res = mod.fit()
                # print res.summary()

                # outliers
                # outl = sorted(izip(res.resid, ind_ids, indv_scores, reord_pheno, reord_group),
                #               reverse=True)
                # print "pos e:", ["(%s,%.2f,%g,%g,%s)" %
                #                  (iid, resid, phen, i_scr, grp)
                #                  for resid, iid, i_scr, phen, grp in outl[:4]]
                # print "neg e:", ["(%s,%.2f,%g,%g,%s)" %
                #                  (iid, resid, phen, i_scr, grp)
                #                  for resid, iid, i_scr, phen, grp in outl[::-1][:4]]

                # plot it
                # plot_lin_reg(reord_pheno, indv_scores,
                #              slope, intercept, r_val, p_val, std_e,
                #              sweep_name, pheno_lab, f_name, reord_group)

                # regress_file.write("{pheno}\t{func}\t{r2}\t{p}\t{stderr}\n".format(pheno=pheno,
                #                                                                    func=f_name,
                #                                                                    r2=r_val**2,
                #                                                                    p=p_val,
                #                                                                    stderr=std_err))

            cfp_file.write("\n".join(["%s\t%g\t%s" % (i_id, i_scr, i_grp)
                           for i_id, i_scr, i_grp in izip(ind_ids, indv_scores, reord_group)]))

    # regress_file.close()
    cfp_file.close()


###############################################################################
def plot_lin_reg(x, y, m, b, r, p, std_err, sweep_name, pheno_lab, func_name, group):

    # init plot
    fig = plt.figure(figsize=(8, 6))  # width, height
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)
    # gs = gridspec.GridSpec(3, 3, width_ratios=[1, 1], left=0.11, bottom=0.06,
    #                        right=0.92, top=0.97, wspace=0.3, hspace=0.2)
    # ax = fig.add_subplot(gs[0, 0])

    fig.subplots_adjust(bottom=0.09, wspace=0.3, left=0.1, right=0.92, top=0.94)
    ax = fig.add_subplot(111)

    # plot data points
    ax.scatter(rand_jitter(x[group == "CMS"], max(x) - min(x)),
               rand_jitter(y[group == "CMS"], max(y) - min(y)),
               facecolor=c1[0], edgecolor='#262626', linewidths=0.5, alpha=0.85, s=30)
    ax.scatter(rand_jitter(x[group == "Control"], max(x) - min(x)),
               rand_jitter(y[group == "Control"], max(y) - min(y)),
               facecolor=c1[1], edgecolor='#262626', linewidths=0.5, alpha=0.85, s=30)

    # plot fitted line
    xi = np.arange(min(x), max(x))
    ax.plot(xi, m*xi+b, color='#262626', linestyle='--', linewidth=0.75)

    # set axis range
    ax.set_ylim([None, max(y)*1.15])

    # plot info
    if m > 0:
        # bottom right
        x_text, y_text = 0.10, 0.94
    else:
        # top right
        x_text, y_text = 0.90, 0.94

    ax.text(x_text, y_text,
            r"\noindent $R^2={:.2f}$\\$P\,\,={:.1e}$".format(r**2, p),
            horizontalalignment='center',
            verticalalignment='center',
            transform=ax.transAxes)

    # title
    ax.set_title(r"$\mathbf{%s}$" % sweep_name)
    ax.set_xlabel(r"$\mathbf{%s}$" % pheno_lab)
    ax.set_ylabel(r"$\mathbf{%s(CFP1, CFP2)}$" % func_name)

    # pretify
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#262626')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#262626')

    # ticks only bottom and left
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', colors='#262626')
    ax.tick_params(axis='y', colors='#262626')
    ax.xaxis.label.set_color('#262626')
    ax.yaxis.label.set_color('#262626')

    # save figure
    plt.savefig('%s/%s.png' % (save_to_dir, sweep_name), dpi=300)
    plt.savefig('%s/pdf/%s.pdf' % (save_to_dir, sweep_name))
    plt.show()
    plt.close(fig)


###############################################################################
def compound_hap_CFPs(cfp_score_pairs, c_func=np.mean):
    """ For each pair of CFP scores, compute a single (compound) score with the given function.
    """

    ans = []
    for hap1_cfp, hap2_cfp in cfp_score_pairs:
        ans.append(c_func([hap1_cfp, hap2_cfp]))

    return np.array(ans)


###############################################################################
def go_focal_site_known(sweep_name, hapmap_chr_file, focal_rsid):
    ''' Run full CFP classification on given sweep '''

    global chimp_polar, no_polar

    print "============================================"
    print "Working on (known focal site) sweep %s..." % sweep_name
    print "============================================\n"

    # init allele polatization stats
    chimp_polar, no_polar = 0.0, 0.0

    # read haplotype matrix, and chimp-polarize alleles (updates polarization stats)
    hap_mat, col_freqs, ba_col, positions = read_phased_hapmap_chr(hapmap_chr_file, focal_rsid)

    if not ba_col:
        print "\nError: focal site not in data!\n"
        sys.exit(1)

    # clean haplotype matrix, removes all-0 columns (non-segregating)
    # may also remove sites with no Chimp data (if artificially set to all-0)
    hap_mat, col_freqs, ba_col, positions = hap_read.remove_zero_cols(hap_mat,
                                                                      col_freqs,
                                                                      ba_col,
                                                                      positions)

    # sanity check
    if len(hap_ids) != len(hap_mat):
        print "Warning: # haplotype IDs != # haplotypes"

    # find and report the focal SNP state
    carrier_status = hap_mat[:, ba_col]
    print "Focal SNP (%s) freq. %g (of %i haplotypes)" % (focal_rsid,
                                                          col_freqs[ba_col],
                                                          len(hap_mat))
    # print np.array(hap_mat[:, ba_col], dtype=int), "\n"

    # output files
    acc_rad_fname = "acc_vs_radius_%s.txt" % sweep_name
    scores_fname = "cfp_scores_labels_%s.txt" % sweep_name
    acc_file, cfp_file = open(acc_rad_fname, 'w'), open(scores_fname, 'w')

    # file headers
    cfp_file.write("#cfp\tpred-label\tP(carr)\tP(non-carr)\ttrue-label\n")
    if report_balanced_acc:
        acc_file.write("#radius\tnum-snps\tb-acc\tsil-score\tbic\taic\tp-val\n")
    else:
        acc_file.write("#radius\tnum-snps\tacc\tsil-score\tbic\taic\tp-val\n")

    # CFP scores at increasing radii around focal SNP
    final_scores_clusters = []
    acc_best, acc_best_r, acc_best_nsnps = 0, None, None

    for r in radii:
        left_idx, right_idx = get_boundary_idx(positions, ba_col, r)

        # 1. CFP scores
        hap_scores = cfp.haplotype_CFP_scores(hap_mat[:, left_idx:right_idx],
                                              col_freqs[left_idx:right_idx], norm=norm)

        # 2. cluster & predict
        pred_status, p_ycarr, p_ncarr, bic, aic, sil_score = cfp.cluster_CFP_scores_GMM(hap_scores)
        # pred_status, sil_score = cfp.cluster_CFP_scores(hap_scores)

        # sanity check
        if len(hap_scores[pred_status == 0]) > 0 and len(hap_scores[pred_status == 1]) > 0:
            if not max(hap_scores[pred_status == 0]) <= min(hap_scores[pred_status == 1]):
                print "Warning: radius=%i, max-0 %g min-1 %g" % (r,
                                                                 max(hap_scores[pred_status == 0]),
                                                                 min(hap_scores[pred_status == 1]))

        # 3. P-value
        zscore, pval = stats.ranksums(hap_scores[carrier_status == 0],
                                      hap_scores[carrier_status == 1])

        logp = -np.log10(pval)  # -np.log2(pval)

        # generate report
        acc, b_acc, tpr, fpr = cfp.clustering_report(pred_status, carrier_status)

        # remember best
        # if( acc >= acc_best ):
        if r == 25000:
            if report_balanced_acc:
                acc_best, acc_best_r, acc_best_nsnps = b_acc, r, right_idx - left_idx
            else:
                acc_best, acc_best_r, acc_best_nsnps = acc, r, right_idx - left_idx

            final_scores_clusters = zip(hap_scores, pred_status, p_ycarr, p_ncarr, carrier_status)

        # report accuracy & radius
        if report_balanced_acc:
            acc_file.write("%i\t%i\t%g\t%g\t%g\t%g\t%g\n" % (r, right_idx-left_idx, b_acc,
                                                             sil_score, bic, aic, logp))
        else:
            acc_file.write("%i\t%i\t%g\t%g\t%g\t%g\t%g\n" % (r, right_idx-left_idx, acc,
                                                             sil_score, bic, aic, logp))

    # report best
    for tup in final_scores_clusters:
        cfp_file.write("%g\t%i\t%g\t%g\t%i\n" % tup)

    acc_file.close()
    cfp_file.close()

    print "\nBest accuracy: %g (radius %i bp)" % (acc_best, acc_best_r)
    print "\nOutput in '%s' and '%s'\n" % (acc_rad_fname, scores_fname)


###############################################################################
def find_center_with_offset(positions, offset=0):
    ''' returns the index in 'positions' that is closest to the
        center of the interval, plus 'offset'
    '''

    real_center = (min(positions) + max(positions)) / 2.0 + offset
    center_loci = min(positions, key=lambda x: abs(x-real_center))
    return np.where(positions == center_loci)


###############################################################################
def get_boundary_idx(pos, ba_col, radius):

    left = bisect.bisect_left(pos, pos[ba_col] - radius)
    right = bisect.bisect_right(pos, pos[ba_col] + radius)

    return left, right


###############################################################################
def read_chimp_allele_table(chimp_allele_file):
    ''' Read Chimpanzee allele for all HapMap loci into dict '''
    global chimp_alleles

    sys.stdout.write("\nReading Chimp alleles table...")
    sys.stdout.flush()

    fh = open(chimp_allele_file, 'r')
    for line in fh:
        # skip header
        if line.startswith("#"):
            continue

        # parse allele
        line_spl = line.rstrip().split()
        rs_id, strand = line_spl[4], line_spl[6]
        obs_alleles = set(line_spl[8].split("/"))
        chimp_a = line_spl[13]

        # save
        if(strand == '-'):
            chimp_alleles[rs_id] = [flip_revcomp(chimp_a), flip_revcomp_set(obs_alleles)]
        else:
            chimp_alleles[rs_id] = [chimp_a, obs_alleles]

    fh.close()

    print "Done.\n"


###############################################################################
def read_phased_hapmap_chr(f_path, focal_rsid):
    ''' Read HapMap3 phased chromosome file
        Updates the global variable 'hap_ids' with the identities of haplotypes.
    '''
    global hap_ids

    print "Reading phased data from '%s' (focal: %s)" % (f_path, focal_rsid)
    sys.stdout.flush()

    # init
    snp_mat, positions, ba_col = [], [], None

    # read phased HapMap3 file
    with open(f_path, 'r') as hapmap_fh:
        for i, line in enumerate(hapmap_fh):

            # header
            if line.startswith("rsID"):
                hap_ids = line.rstrip().split()[2:]
                continue

            # parse SNP info
            line_spl = line.rstrip().split()
            rsid, pos, alleles = line_spl[0], int(line_spl[1]), line_spl[2:]

            # save column of focal site
            if focal_rsid == rsid:
                ba_col = i-1

            # polarize alleles based on Chimp allele
            alleles = chimp_0(rsid, line_spl[2:])

            snp_mat.append(alleles)  # save SNP as row
            positions.append(pos)    # save position

    # haplotype matrix & positions
    snp_mat, positions = np.array(snp_mat), np.array(positions)
    hap_mat = snp_mat.T

    # allele frequencies
    col_freqs = np.sum(hap_mat, axis=0) / float(len(hap_mat))

    # report summary
    print ("\tSNPs read: %i (chimp-polarized %.2f, discarded %.2f).\n" %
           (i, chimp_polar/float(i), no_polar/float(i)))

    return hap_mat, col_freqs, ba_col, positions


###############################################################################
def allele_sanity_and_flip(chimp_a, chimp_set, loc_allele_list):
    ''' Check sanity of observed HapMap alleles vs. Chimp data
        Return the sanity {True, False} & Chimp allele (flipped, if necessary)
    '''

    obs_loc_set = set(loc_allele_list)
    if obs_loc_set.issubset(chimp_set):
        # all good
        return True, chimp_a
    else:
        # observed HapMap alleles not a subset of Chimp
        if DEBUG:
            print ("HapMap alleles (%s) not subset of Chimp (%s)" %
                   (','.join(e for e in obs_loc_set), ','.join(e for e in chimp_set)))

        # check if flip allele issue
        if obs_loc_set.issubset(flip_revcomp_set(chimp_set)):
            if DEBUG:
                print ("HapMap subset after flipping chimp (%s)" %
                       ','.join(e for e in flip_revcomp_set(chimp_set)), "\n")
            # chimp allele was flipped, flip back
            return True, flip_revcomp(chimp_a)
        else:
            # just bad data
            return False, None


###############################################################################
def chimp_0(rsID, locus_alleles):
    ''' convert based on chimp allele if possible, otherwise major/minor '''
    global chimp_polar, no_polar

    if rsID in chimp_alleles:
        # variant has Chimp allele data

        # sanity check
        sane, chimp_a = allele_sanity_and_flip(chimp_alleles[rsID][0],
                                               chimp_alleles[rsID][1],
                                               locus_alleles)
        if sane:
            # good data
            chimp_polar += 1
        else:
            # bad Chimp data for site, set to all-0 (removed later)
            no_polar += 1
            return np.zeros(len(locus_alleles))

        # convert
        for i, a in enumerate(locus_alleles):
            if a == chimp_a:
                locus_alleles[i] = 0.0
            else:
                locus_alleles[i] = 1.0

        return np.array(locus_alleles)

    else:
        # no Chimp data for site
        if DISCARD_NO_CHIMP:
            # set to all-0 (removed later)
            no_polar += 1
            return np.zeros(len(locus_alleles))
        elif rsID in ref_alleles:
            # use reference to set major/minor
            ref_a = ref_alleles[rsID]
            for i, a in enumerate(locus_alleles):
                if a == ref_a:
                    locus_alleles[i] = 0.0
                else:
                    locus_alleles[i] = 1.0

            return np.array(locus_alleles)
        else:
            print "Cannot set reference allele without reference data. Missing rsID %s" % rsID
            sys.exit(1)


###############################################################################
def flip_revcomp(base):
    ''' reverse complement base '''

    if base == 'A':
        return 'T'
    elif base == 'G':
        return 'C'
    elif base == 'C':
        return 'G'
    elif base == 'T':
        return 'A'
    else:
        "Error: flip_revcomp() got NON-DNA!"


###############################################################################
def flip_revcomp_set(bases):
    ''' reverse complement set '''
    ans = set()

    for b in bases:
        ans.add(flip_revcomp(b))

    return ans


###############################################################################
def read_ref_allele_table(f_path):
    """ Read reference alleles data from file.
        File expected in the HapMap2 or IMPUTE SNP 'legend' format.
    """

    global ref_alleles, DISCARD_NO_CHIMP

    DISCARD_NO_CHIMP = False  # reference alleles file was supplied, will be used

    sys.stdout.write("Reading reference alleles table...")
    sys.stdout.flush()

    with open(f_path, 'r') as ref_allele_fh:
        for line in ref_allele_fh:
            if line.startswith("ID"):
                continue

            rsID, loc, ref_a, alt_a = line.rstrip().split()
            ref_alleles[rsID] = ref_a

    print "Done.\n"


###############################################################################
def major_0(locus_alleles):
    ''' !! DEPRECATED !!
        Convert allele list from {A,C,G,T} to {0,1} based on observed major & minor allele.
        This effectively fold the site's frequency, i.e. sets it to max of (f, 1-f).
    '''

    tally = Counter()
    for a in locus_alleles:
        tally[a] += 1
    major_a = max(tally.iteritems(), key=operator.itemgetter(1))[0]
    minor_a = min(tally.iteritems(), key=operator.itemgetter(1))[0]

    if(minor_a == major_a):
        # fixed in the sample, all 0
        for i in range(len(locus_alleles)):
            locus_alleles[i] = 0.0
    else:
        # convert minor/major to 0/1
        for i, a in enumerate(locus_alleles):
            if(a == minor_a):
                locus_alleles[i] = 1.0
            elif(a == major_a):
                locus_alleles[i] = 0.0
            else:
                print "Error: >2 alleles in HapMap chromsome row\n"
                sys.exit(1)

    return np.array(locus_alleles)


###############################################################################
def rand_jitter(arr, range_size):
    return arr + np.random.randn(len(arr)) * (JITTER_FRAC*range_size)


###############################################################################
#                                  MAIN                                       #
###############################################################################
if __name__ == '__main__':

    if len(sys.argv) not in [2, 3]:
        print ("\n\tusage: %s <path/to/chimp-alleles.txt> [path/to/ref-alleles.txt]\n"
               % (os.path.basename(sys.argv[0])))
        sys.exit(1)

    # first, read chimp allele data
    read_chimp_allele_table(sys.argv[1])

    if len(sys.argv) == 3:
        # read reference allele data
        read_ref_allele_table(sys.argv[2])

    # iterate sweeps
    for sweep_name, (hapmap_file, focal_rsID) in sweeps.iteritems():

        if focal_rsID:
            # run CFP classification
            go_focal_site_known(sweep_name, hapmap_file, focal_rsID)
        else:
            # run CFP correlation
            go_focal_site_unknown(sweep_name, hapmap_file)
