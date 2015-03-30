#!/usr/bin/env python

''' plot results from real sweep data, nonCMS '''

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
from scipy import stats

save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/real_sweeps_CMS"
lab_fontsize = 11
max_noise = 0.5

sweeps = {
    "ANP32D": ("cfp_scores_variables_ANP32D.txt",),
    # "ARID1B": ("cfp_scores_variables_ARID1B.txt",),
    "CD3E": ("cfp_scores_variables_CD3E.txt",),
    # "CNNM1": ("cfp_scores_variables_CNNM1.txt",),
    # "DUOX": ("cfp_scores_variables_DUOX.txt",),
    # "GUSBP4": ("cfp_scores_variables_GUSBP4.txt",),
    # "PBX4": ("cfp_scores_variables_PBX4.txt",),
    "SENP1": ("cfp_scores_variables_SENP1.txt",),
    "SUSD5": ("cfp_scores_variables_SUSD5.txt",),
}

c1 = brewer2mpl.get_map('Set1',  'Qualitative', 9).mpl_colors
c3 = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors


def go():
    ''' Plots a single figure visualizing the CFP scores for nonCMS sweeps.
    '''

    # init plot
    fig = plt.figure(figsize=(7, 2.75))  # width, height
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)
    gs = gridspec.GridSpec(1, 4,  # width_ratios=[1, 3]
                           left=0.13, bottom=0.03,
                           right=0.96, top=0.9,
                           wspace=0.57, hspace=0.25)

    for i, (sweep_name, (cfp_file,)) in enumerate(sweeps.iteritems()):
        ax = fig.add_subplot(gs[i])
        i_ids, i_cfps, i_grps = read_file(cfp_file)
        plot_sweep_cfps_on_ax(ax, i_cfps, i_grps, sweep_name, i)

    # save figure
    plt.savefig('%s/all.png' % (save_to_dir), dpi=300)
    plt.savefig('%s/pdf/all.pdf' % (save_to_dir))
    plt.show()
    plt.close(fig)


def plot_sweep_cfps_on_ax(ax, i_cfps, i_grps, sweep_name, plot_num):
    '''
    '''

    # stratify CFP scores on real label
    cfp_CMS = i_cfps[i_grps == "CMS"]
    cfp_nonCMS = i_cfps[i_grps == "Control"]

    zscore, pval = stats.ranksums(cfp_CMS, cfp_nonCMS)

    # X-values with random noise
    x_CMS = add_random_nosie([1]*len(cfp_CMS), only_positive=True)
    x_nonCMS = add_random_nosie([1]*len(cfp_nonCMS), only_positive=True)

    ax.scatter(x_nonCMS, cfp_nonCMS, c=c1[0], s=32, alpha=0.80, linewidths=0.35,
               edgecolor=c3[7], marker='o')
    ax.scatter(x_CMS, cfp_CMS, c=c1[1], s=32, alpha=0.80, linewidths=0.35,
               edgecolor=c3[7], marker='o')

    # limits
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([None, 1.18*max(i_cfps)])
    if plot_num == 0:
        ax.set_ylabel(r"$\mathbf{min({CFP1}^1,{CFP2}^1)}$", fontsize=lab_fontsize)

    # sweep name
    ax.set_title(r"$\mathbf{%s}$" % sweep_name, fontsize=lab_fontsize)

    # p-value
    ax.text(0.5, 0.96, r"$P={:.1e}$".format(pval).replace("-", "\,\mbox{--}\,"), fontsize=10,
            transform=ax.transAxes, verticalalignment='top', horizontalalignment='center')

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
    ax.get_xaxis().set_ticklabels([])
    ax.get_yaxis().tick_left()
    # ax.get_xaxis().tick_bottom()
    ax.tick_params(axis='x', colors='#262626')
    ax.tick_params(axis='y', colors='#262626')
    ax.xaxis.label.set_color('#262626')
    ax.yaxis.label.set_color('#262626')
    ax.xaxis.set_ticks_position('none')  # remove x ticks
    # ax.yaxis.set_ticks_position('none')  # remove y ticks


def add_random_nosie(vals, only_positive=False):
    ''' adds randome noise (at most x_max_noise) to each point in given list
    '''
    val_r = []

    # add random noise
    for i in range(len(vals)):
        v_r = vals[i] + random.uniform(-1.0, 1.0) * max_noise
        if only_positive and v_r < 0:
            # undo if turned out negative and we want strictly positive
            v_r = vals[i]

        val_r.append(v_r)

    return val_r


def read_file(fpath):
    ''' Reads CFP scores from file 'fpath', formatted as:
            ------------------------------
            #indID  min(CFP1,CFP2)  group
            103-RQ  4621    Control
            109-EP  4596    Control
            ...
            ...
            ------------------------------

        Returns a dict of individual ID's/str -> (cfp/float, group/str).
    '''
    print "reading from %s" % fpath
    i_ids, i_cfps, i_grps = [], [], []
    with open(fpath, 'r') as fh:
        for line in fh:
            if line.startswith("#"):
                continue

            i_id, i_cfp, i_grp = line.rstrip().split()

            i_ids.append(i_id)
            i_cfps.append(float(i_cfp))
            i_grps.append(i_grp)

    return np.array(i_ids), np.array(i_cfps), np.array(i_grps)

###############################################################################
if __name__ == '__main__':

    go()
