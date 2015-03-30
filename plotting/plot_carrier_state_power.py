#!/usr/bin/env python

''' plot power of carrier-state prediction by CFP score '''

import sys
import os
import math
import operator
import random
import time
import datetime
import matplotlib
import pylab
import brewer2mpl
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib import rcParams

''' internal imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p
import cfp_score as cfp
import hap_reader_ms as hread


###############################################################################
#                            PARAMETER SETTINGS
###############################################################################
tall_fig = False
lab_fontsize = 14

f, s = 0.3, 0.05
f_step_size = 0.1
norm = 1
times = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800]

# make frequency bins
f_bins = np.append(np.arange(0.0, 1.0, f_step_size), [1.0])

prediction_stats_file_to_read = None
prediction_stats_file_to_write = "/home/rronen/Desktop/carrier_status_stats_f%g_s%g.txt" % (f, s)

# dir to save plots
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/carrier_stats"

c1 = brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors


###############################################################################
#                                    GO
###############################################################################
def go():

    if(prediction_stats_file_to_read):
        # ########### read prediction stats from file ###########
        print "Reading prediction stats..."
        pred_stats = read_prediction_stats_file(prediction_stats_file_to_read)

    else:
        # ###### generate prediction stats, write & read ########
        pred_stats_fh = open(prediction_stats_file_to_write, 'w')

        # header
        pred_stats_fh.write("#f\ts\tt\tsim\tba-freq\tacc\tTPR\tFPR\n")

        for t in times:
            print "working on t=%i" % t

            for sim in range(p.last_sim):
                # haplotype data
                hap_mat_s, col_freqs_s, mut_pos_s, bacol = hread.ms_hap_mat(f, s, t, sim, "s")

                # skip simulation if beneficial allele fixed
                if(col_freqs_s[bacol] == 1.0 or col_freqs_s[bacol] == 0.0):
                    continue

                # carrier status
                carrier_status = hap_mat_s[:, bacol]

                # CFP score for each haplotype
                hap_scores = cfp.haplotype_CFP_scores(hap_mat_s, col_freqs_s, norm=norm)

                # cluster
                acc, b_acc, tpr, fpr = cfp.cluster_CFP_scores_report(hap_scores, carrier_status)
                predicted_status, bic = cfp.cluster_CFP_scores_GMM(hap_scores)

                # write stats for current sim
                pred_stats_fh.write("%g\t%g\t%i\t%i\t%g\t%g\t%g\t%g\n"
                                    % (f, s, t, sim, col_freqs_s[bacol], b_acc, tpr, fpr))

        pred_stats_fh.close()

        # read data from file
        pred_stats = read_prediction_stats_file(prediction_stats_file_to_write)

    # ###### plot clustering stats per freq bin #########

    # init plot
    fig = plt.figure(figsize=(10, 5.75))  # width, height
    fig.subplots_adjust(bottom=0.09, left=0.09, right=0.96, top=0.97)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)
    ax1 = fig.add_subplot(111)

    # plot axis
    plot_prediction_stats_on_axis(ax1, pred_stats)

    # save figure
    plt.savefig('%s/carrier_pred_stats_f%.1f_s%g.png' % (save_to_dir, f, s), dpi=300)
    plt.savefig('%s/pdf/carrier_pred_stats_f%.1f_s%g.pdf' % (save_to_dir, f, s))
    plt.show()
    plt.close(fig)


###############################################################################
def plot_2_pred_stats(file1, file2):
    global f  # determines first bin to plot

    print "Reading prediction stats..."
    pred_stats1 = read_prediction_stats_file(file1)
    pred_stats2 = read_prediction_stats_file(file2)

    # ###### plot clustering stats per freq bin #########

    # init plot
    if tall_fig:
        fig = plt.figure(figsize=(8.75, 7.75))  # width, height
    else:
        fig = plt.figure(figsize=(10, 4))  # width, height

    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)

    if tall_fig:
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
    else:
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    # plot axis
    f = 0.0
    if tall_fig:
        plot_prediction_stats_on_axis(ax1, pred_stats1, xaxis_label=False, yaxis_label=True)
    else:
        plot_prediction_stats_on_axis(ax1, pred_stats1, xaxis_label=True, yaxis_label=True)
    f = 0.3
    if tall_fig:
        plot_prediction_stats_on_axis(ax2, pred_stats2, xaxis_label=True, yaxis_label=True)
    else:
        plot_prediction_stats_on_axis(ax2, pred_stats2, xaxis_label=True, yaxis_label=False)

    # subplot labels
    if tall_fig:
        fig.text(0.01, 0.968, r"$\mathbf{A}$", fontsize=21)
        fig.text(0.01, 0.505, r"$\mathbf{B}$", fontsize=21)
    else:
        fig.text(0.007, 0.94, r"$\mathbf{A}$", fontsize=21)
        fig.text(0.507, 0.94, r"$\mathbf{B}$", fontsize=21)

    if tall_fig:
        plt.tight_layout(pad=1.5, h_pad=1.25)
    else:
        plt.tight_layout(w_pad=3.5)

    # save figure
    plt.savefig('%s/carrier_pred_stats.png' % (save_to_dir), dpi=300)
    plt.savefig('%s/pdf/carrier_pred_stats.pdf' % (save_to_dir))
    plt.show()
    plt.close(fig)


###############################################################################
def plot_prediction_stats_on_axis(ax, pred_stats, xaxis_label=True, yaxis_label=True):

    OFFSET = 1
    MIN_DATA_FOR_BOX = 100

    # generate data
    data, pos, ticks, tick_l = [], [], [], []
    for b, (lab, acc_list, tpr_list, fpr_list) in pred_stats.iteritems():
        # bin edges
        bin_start, bin_end = bin_freqs_from_tick_label(lab)

        # sufficient data check
        if(len(acc_list) > MIN_DATA_FOR_BOX and bin_start >= f):
            data.append(acc_list)
            pos.append(OFFSET + 1.5*b)
            tick_l.append(lab)
        else:
            tick_l.append("")

        ticks.append(OFFSET + 1.5*b)

    # make boxplots
    bp = ax.boxplot(data, notch=1, positions=pos, vert=True, patch_artist=True)

    # format boxplots
    linecol, facecol = '#262626', ''  # c[1]
    pylab.setp(bp['whiskers'], color='#262626', alpha=0.75)
    pylab.setp(bp['caps'],     color='#262626', alpha=0.75)
    pylab.setp(bp['fliers'],   color='#262626', alpha=0.25, marker='o')  # c1[1]
    pylab.setp(bp['boxes'],    color='#262626', alpha=0.75, facecolor='none')
    pylab.setp(bp['medians'],  color='#262626', alpha=0.75)

    # set axis range
    ax.set_xlim([0, max(ticks) + min(ticks)])
    ax.set_ylim([-0.03, 1.03])

    # set ticks
    ax.set_xticks(ticks)
    ax.set_xticklabels([polish_tick_label(x) for x in tick_l])

    # set axis labels
    if yaxis_label:
        ax.set_ylabel(r"$\mathbf{Balanced \,\, Acc.}$", fontsize=lab_fontsize)
    if xaxis_label:
        ax.set_xlabel(r"$\mathbf{Adaptive \,\, allele \,\, frequency}$", fontsize=lab_fontsize)

    # beutify
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#262626')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#262626')
    ax.get_yaxis().tick_left()
    ax.get_xaxis().tick_bottom()


def read_prediction_stats_file(fname):
    ''' Read clustering stats from given file into bin dict.
    '''

    # prediction stats dict
    prediction_stats = {}  # [bin] -> [bin, label, [Accuracies..], [TPRs..], [FPRs..]]

    # initialize individual bin lists
    for b, (low_f, high_f) in enumerate(zip(f_bins, f_bins[1:])):
        # bin label, accuracy, TPR, FPR
        prediction_stats[b] = ["%g-%g" % (low_f, high_f), [], [], []]

    stats_file = open(fname, 'r')
    for line in stats_file:

        if(line.startswith("#")):
            continue

        # get data from line
        f, s, t, sim, ba_f, acc, tpr, fpr = [float(s) for s in line.rstrip().split()]

        # save in dict
        bin = np.digitize([ba_f], f_bins)[0] - 1
        bin = max(0, bin)  # as soft sweeps may start at freq (slightly) < f
        prediction_stats[bin][1].append(acc)
        prediction_stats[bin][2].append(tpr)
        prediction_stats[bin][3].append(fpr)

    stats_file.close()
    return prediction_stats


def polish_tick_label(lab):
    ''' 1. reduce whitespace in latex math string, i.e. '0 - 100' to '0-100'
        2. convert generations to units of 2N_e
    '''

    if not lab:
        return ""

    # parse out frequencies
    bin_left, bin_right = bin_freqs_from_tick_label(lab)

    # convert to 2N_e if necessary
    bin_left = (bin_left / 2000.0) if bin_left > 1.0 else bin_left
    bin_right = (bin_right / 2000.0) if bin_right > 1.0 else bin_right

    if tall_fig:
        return r"$%g\,\mbox{-}\,%g$" % (bin_left, bin_right)
    else:
        return r"$%.1f$" % bin_left + "\n" + r"$%.1f$" % bin_right


def bin_freqs_from_tick_label(lab):
    bin_left, bin_right = [float(x) for x in lab.split("-")]
    return bin_left, bin_right


###############################################################################
#                                  MAIN
###############################################################################
if __name__ == '__main__':

    if len(sys.argv) == 2:
        if(sys.argv[1] == 'read-data'):
            go()
        else:
            prediction_stats_file_to_read = sys.argv[1]
            go()
    elif len(sys.argv) == 3:
        plot_2_pred_stats(sys.argv[1], sys.argv[2])
    else:
        print ("\n\tusage: %s \'read-data\' OR <pred-stats> OR <pred-stats1> <pred-stats2>\n"
               % os.path.basename(sys.argv[0]))
        sys.exit(1)
