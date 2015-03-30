#!/usr/bin/env python

''' plot results from real sweep data '''

import sys
import os
import operator
import math
import random
import itertools
import pylab
import matplotlib
import brewer2mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from collections import defaultdict
from matplotlib import rcParams

lab_fontsize = 12

c1 = brewer2mpl.get_map('Set1',  'Qualitative', 9).mpl_colors
c3 = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors

# known selective sweeps Fig.
sweeps = {
    # old format (no posterior)
    "LCT":   ("acc_vs_radius_LCT_CEU.txt",   "cfp_scores_labels_LCT_CEU.txt", "CEU"),
    "TRPV6": ("acc_vs_radius_TRPV6_CEU.txt", "cfp_scores_labels_TRPV6_CEU.txt", "CEU"),
    "PSCA":  ("acc_vs_radius_PSCA_YRI.txt",  "cfp_scores_labels_PSCA_YRI.txt",  "YRI"),
    "ADH1B": ("acc_vs_radius_ADH1B_CHB+JPT.txt", "cfp_scores_labels_ADH1B_CHB+JPT.txt",
              "CHB\mbox{+}JPT"),
    "EDAR":  ("acc_vs_radius_EDAR_CHB+JPT.txt",  "cfp_scores_labels_EDAR_CHB+JPT.txt",
              "CHB\mbox{+}JPT"),

    # new format (with posterior)
    # "LCT":   ("acc_vs_radius_LCT.txt",   "cfp_scores_labels_LCT.txt", "CEU"),
    # "TRPV6": ("acc_vs_radius_TRPV6.txt", "cfp_scores_labels_TRPV6.txt", "CEU"),
    # "PSCA":  ("acc_vs_radius_PSCA.txt",  "cfp_scores_labels_PSCA.txt",  "YRI"),
    # "ADH1B": ("acc_vs_radius_ADH1B.txt", "cfp_scores_labels_ADH1B.txt",
    #           "CHB\mbox{+}JPT"),
    # "EDAR":  ("acc_vs_radius_EDAR.txt",  "cfp_scores_labels_EDAR.txt",
    #           "CHB\mbox{+}JPT"),
}

# supp. Fig., PSCA in East Asia (CHB)
sweeps = {
    "PSCA":  ("acc_vs_radius_PSCA_CHB.txt",  "cfp_scores_labels_PSCA_CHB.txt",  "CHB"),
}

# dir to save plots
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/real_sweeps"
plot_all_sweeps = False  # True
x_max_noise = 0.5
show = True
normalize_cfp = False
plot_posterior = False
new_fformat = False


###############################################################################
def go():

    if plot_all_sweeps:
        # plot all genes as joint fig
        plot_all()
    else:
        # plot each gene as separate fig
        for sweep_name, (rad_acc_file, score_lab_file, pop) in sweeps.iteritems():
            plot_single_gene(sweep_name, rad_acc_file, score_lab_file, pop)


##############################################################################
def plot_all():

    # init plot
    fig = plt.figure(figsize=(8, 9.5))  # width, height
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)
    if new_fformat and plot_posterior:
        gs = gridspec.GridSpec(5, 2, width_ratios=[1, 2.25], left=0.09, bottom=0.06,
                               right=0.92, top=0.97, wspace=0.3, hspace=0.3)
    else:
        gs = gridspec.GridSpec(5, 2, width_ratios=[1, 3.75], left=0.11, bottom=0.06,
                               right=0.92, top=0.97, wspace=0.3, hspace=0.2)

    # LCT
    plot_cfp_lab_on_axis(fig.add_subplot(gs[0, 0]), sweeps["LCT"][1])
    plot_acc_rad_on_axis(fig.add_subplot(gs[0, 1]), sweeps["LCT"][0], "LCT", sweeps["LCT"][2])

    # TRPV6
    plot_cfp_lab_on_axis(fig.add_subplot(gs[1, 0]), sweeps["TRPV6"][1])
    plot_acc_rad_on_axis(fig.add_subplot(gs[1, 1]), sweeps["TRPV6"][0], "TRPV6", sweeps["TRPV6"][2])

    # PSCA
    plot_cfp_lab_on_axis(fig.add_subplot(gs[2, 0]), sweeps["PSCA"][1])
    plot_acc_rad_on_axis(fig.add_subplot(gs[2, 1]), sweeps["PSCA"][0], "PSCA",  sweeps["PSCA"][2])

    # ADH1B
    plot_cfp_lab_on_axis(fig.add_subplot(gs[3, 0]), sweeps["ADH1B"][1])
    plot_acc_rad_on_axis(fig.add_subplot(gs[3, 1]), sweeps["ADH1B"][0], "ADH1B", sweeps["ADH1B"][2])

    # EDAR
    plot_cfp_lab_on_axis(fig.add_subplot(gs[4, 0]), sweeps["EDAR"][1], show_xaxis_label=True)
    plot_acc_rad_on_axis(fig.add_subplot(gs[4, 1]), sweeps["EDAR"][0], "EDAR", sweeps["EDAR"][2],
                         show_xaxis_label=True)

    # subfig labels
    if new_fformat and plot_posterior:
        col2_x = 0.36
    else:
        col2_x = 0.28
    subfig_label(fig, 0.01, 0.998, "A")
    subfig_label(fig, col2_x, 0.998, "B")
    subfig_label(fig, 0.01, 0.810, "C")
    subfig_label(fig, col2_x, 0.810, "D")
    subfig_label(fig, 0.01, 0.620, "E")
    subfig_label(fig, col2_x, 0.620, "F")
    subfig_label(fig, 0.01, 0.430, "G")
    subfig_label(fig, col2_x, 0.430, "H")
    subfig_label(fig, 0.01, 0.240, "I")
    subfig_label(fig, col2_x, 0.240, "J")

    # save figure
    plt.savefig('%s/all.png' % (save_to_dir), dpi=300)
    plt.savefig('%s/pdf/all.pdf' % (save_to_dir))
    if show:
        plt.show()
    plt.close(fig)


##############################################################################
def plot_single_gene(sweep_name, radius_vs_acc_fname, scores_fname, pop):

    # init plot
    fig = plt.figure(figsize=(8, 3.5))  # width, height
    fig.subplots_adjust(bottom=0.15, wspace=0.3, left=0.1, right=0.92, top=0.9)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)
    if new_fformat and plot_posterior:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2.25])
    else:
        gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3.5])
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])

    # plot axis
    plot_cfp_lab_on_axis(ax1, scores_fname)
    plot_acc_rad_on_axis(ax2, radius_vs_acc_fname, sweep_name, pop, show_xaxis_label=True)

    # letter
    fig.text(0.01, 0.96, r"$\mathbf{A}$", transform=ax1.transAxes,
             fontsize=21, verticalalignment='top')
    fig.text(0.30, 0.96, r"$\mathbf{B}$", transform=ax2.transAxes,
             fontsize=21, verticalalignment='top')

    # save figure
    plt.savefig('%s/%s.png' % (save_to_dir, sweep_name), dpi=300)
    plt.savefig('%s/pdf/%s.pdf' % (save_to_dir, sweep_name))
    if show:
        plt.show()
    plt.close(fig)


##############################################################################
def plot_acc_rad_on_axis(ax, fname, sweep_name, pop_name, show_xaxis_label=False):

    # read data
    rad_nsnps_acc = np.loadtxt(fname)
    rad, nsnps, acc = rad_nsnps_acc[:, 0], rad_nsnps_acc[:, 1], rad_nsnps_acc[:, 2]
    if new_fformat:
        logp = rad_nsnps_acc[:, 6]
    else:
        logp = rad_nsnps_acc[:, 4]

    ax2 = ax.twinx()

    # convert radii to window sizes in Mbp
    w_size_mb = rad*2/1000000.0

    # plot line with markers
    acc_line, = ax.plot(w_size_mb, acc, '-', linewidth=1.1, marker='o', c='#262626', ms=3,
                        markerfacecolor='#262626', markeredgecolor='none')

    # plot log p values
    lp_line, = ax2.plot(w_size_mb, logp, '-', linewidth=1.1, marker='o', c=c1[1], ms=3,
                        markerfacecolor=c1[1], markeredgecolor='none')

    # circle data point used
    x, y = 50000/1000000.0, acc[rad == 25000]
    ax.scatter([x], [y], c='none', s=55, alpha=1.0, linewidths=1.0, edgecolor=c1[0], marker='o')

    ax.set_ylim([0.49, 1.03])  # Balanced Acc.
    ax.set_xlim([0, 1.45])  # Mbp
    if show_xaxis_label:
        ax.set_xlabel(r"$\mathbf{window \,\, size \,\, (Mbp)}$", fontsize=lab_fontsize)

    # sweep name
    ax.text(0.99, 0.96, r"$\mathbf{%s \,\, , %s}$" % (sweep_name, pop_name),
            transform=ax2.transAxes, fontsize=10,
            horizontalalignment='right', verticalalignment='top')

    ax.set_ylabel(r"$\mathbf{Balanced \,\, Acc.}$", fontsize=lab_fontsize)
    if new_fformat:
        ax2.set_ylabel(r"$\mathbf{-log_{10} P}$", fontsize=lab_fontsize)
    else:
        ax2.set_ylabel(r"$\mathbf{-log_{2} P}$", fontsize=lab_fontsize)

    # pretify
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#262626')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#262626')
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['top'].set_color('#262626')
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['right'].set_color('#262626')

    # ticks only bottom and left
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', colors='#262626')
    ax.tick_params(axis='y', colors='#262626')
    ax.xaxis.label.set_color('#262626')
    ax.yaxis.label.set_color('#262626')
    # ax.xaxis.set_ticks_position('none') # remove x ticks
    # ax.yaxis.set_ticks_position('none') # remove y ticks

    # replace first tick position
    # xticks = ax.get_xticks()
    # xticks[0] = rad[0]*2/1000000.0
    xticks = np.insert(np.arange(0.25, 1.51, 0.25), 0, 0.05)
    ax.set_xticks(xticks)

    if not show_xaxis_label:
        ax.set_xticklabels([])  # no x ticks labels

    if(sweep_name == "LCT"):
        legend = ax.legend([acc_line, lp_line],
                           [r"$\mathbf{accuracy}$", r"$\mathbf{-log_2 P}$"],
                           loc='lower left', markerscale=1.2, labelspacing=0.16,
                           borderpad=0.3, handletextpad=0.2, prop={'size': 11},
                           shadow=False, fancybox=True)

        rect = legend.get_frame()
        rect.set_facecolor(np.array([float(247)/float(255)]*3))
        rect.set_linewidth(0.0)  # remove line around legend
        for t in legend.texts:
            t.set_color('#262626')


##############################################################################
def plot_cfp_lab_on_axis(ax, fname, show_xaxis_label=False):

    # read data
    cfp_pred_real = np.loadtxt(fname)
    if new_fformat:
        cfp = cfp_pred_real[:, 0]
        pred_lab = cfp_pred_real[:, 1]
        p_ycarr = cfp_pred_real[:, 2]
        p_ncarr = cfp_pred_real[:, 3]
        real_lab = cfp_pred_real[:, 5]

        # posterior of predicted label
        p_pred_lab = np.maximum.reduce([p_ycarr, p_ncarr])
    else:
        cfp = cfp_pred_real[:, 0]
        pred_lab = cfp_pred_real[:, 1]
        real_lab = cfp_pred_real[:, 2]

    # CFP normalization
    if normalize_cfp:
        cfp = cfp / len(cfp_pred_real)

    # stratify CFP scores on real label
    if new_fformat and plot_posterior:
        cfp_y_c = add_random_nosie(cfp[real_lab == 1], est_mag=True)
        cfp_n_c = add_random_nosie(cfp[real_lab == 0], est_mag=True)
    else:
        cfp_y_c = cfp[real_lab == 1]
        cfp_n_c = cfp[real_lab == 0]

    if new_fformat and plot_posterior:
        # x-values as log(posterior-of-carrier-class)
        py = p_ycarr[real_lab == 1]
        py[py == .0] = 10**-100
        x_y_c = add_random_nosie(-np.log10(py), est_mag=True)

        pn = p_ycarr[real_lab == 0]
        pn[pn == .0] = 10**-100
        x_n_c = add_random_nosie(-np.log10(pn), est_mag=True)
    else:
        # x-values with random noise
        x_y_c = add_random_nosie([1]*len(cfp_y_c))
        x_n_c = add_random_nosie([1]*len(cfp_n_c))

    ax.scatter(x_y_c, cfp_y_c, c=c1[0], s=32, alpha=0.80, linewidths=0.35,
               edgecolor=c3[7], marker='o')
    ax.scatter(x_n_c, cfp_n_c, c=c1[1], s=32, alpha=0.80, linewidths=0.35,
               edgecolor=c3[7], marker='o')

    # limits
    if not plot_posterior:
        ax.set_xlim([0.0, 2.0])
    ax.set_ylim([-0.03*max(cfp), 1.22*max(cfp)])

    if normalize_cfp:
        ax.set_ylabel(r"$\mathbf{1\mbox{-}CFP/n}$", fontsize=lab_fontsize)
    else:
        ax.set_ylabel(r"$\mathbf{1\mbox{-}CFP}$", fontsize=lab_fontsize)

    if new_fformat and plot_posterior and show_xaxis_label:
        ax.set_xlabel(r"$\mathbf{-log_{10} P(carrier)}$", fontsize=lab_fontsize)

    y_slack = max(cfp[pred_lab == 1]) * 0.05
    min_clust_y = min(cfp[pred_lab == 1]) - y_slack
    max_clust_y = max(cfp[pred_lab == 1]) + y_slack

    ax.axhspan(min_clust_y, max_clust_y, facecolor='gray', edgecolor='none', alpha=0.25)

    # plot num carriers / total haplotypes
    ax.text(0.5, 0.96,
            r"$n\!=\!%i \,\, (%i)$" % (len(cfp), len(cfp_y_c)),
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='center')

    # pretify
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#262626')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#262626')
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['top'].set_color('#262626')
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['right'].set_color('#262626')

    # ticks only bottom and left
    if not plot_posterior:
        # remove x ticks
        ax.xaxis.set_ticks_position('none')
        ax.get_xaxis().set_ticklabels([])
    else:
        ax.get_xaxis().tick_bottom()

    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', colors='#262626')
    ax.tick_params(axis='y', colors='#262626')
    ax.xaxis.label.set_color('#262626')
    ax.yaxis.label.set_color('#262626')


###############################################################################
def add_random_nosie(x_vals, est_mag=False):
    ''' adds randome noise (at most max_noise) to each point in given list
    '''
    x_r = []

    if est_mag:
        max_noise = np.abs(max(x_vals)-min(x_vals))*0.1
    else:
        max_noise = x_max_noise

    # add random noise
    for i in range(len(x_vals)):
        x_r.append(x_vals[i] + random.uniform(-1.0, 1.0) * max_noise)

    return x_r


###############################################################################
def subfig_label(fig, x, y, text):
    fig.text(x, y, r"$\mathbf{%s}$" % text, fontsize=20, verticalalignment='top')


###############################################################################
if __name__ == '__main__':

    go()
