#!/usr/bin/env python

''' plotting utility for haplotype CFP progression in a sweep '''

import sys
import os
import operator
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from matplotlib import rcParams
import brewer2mpl

''' internal imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p
import cfp_score as cfp
import hap_reader_ms as hread

###############################################################################
# ############################## SETTINGS #####################################
###############################################################################

x_max_noise = 0.35
y_max = {1: 0, 2: 0}  # max CFP-score per norm

# (f,s,t) -> scatters
scatter_dict_s = {}  # holds ( x_init, y_init, x_new, y_new, x_benef, y_benef )
scatter_dict_n = {}  # holds ( x_init, y_init, x_new, y_new )

# sweep parameters
f = 0.3
times = [0, 50, 100, 150, 200, 300, 400, 500, 1500, 3000, 4000]

# f = 0.0
# times  = [25, 200, 250, 300, 350, 400, 450, 500, 600, 700, 1000, 1500, 2000, 3000]

# break x-axis before these times
breaks_after = [4, 7]

s = 0.05

# times  = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# times += [i for i in range(600,2001,100)]

c1 = brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors
c2 = brewer2mpl.get_map('Set2', 'Qualitative', 8).mpl_colors
c3 = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors
cp = brewer2mpl.get_map('YlGn', 'Sequential', 9).mpl_colors

# dir to save plots
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/cfp_1d_progress"


###############################################################################
# ############################ MAIN FUNCTION ##################################
###############################################################################
def go():

    # for sim in range(p.last_sim):
    for sim in [18]:

        for t in times:

            # read popuation sample files
            hap_mat_s, col_freqs_s, mut_pos_s, bacol = hread.ms_hap_mat(f, s, t, sim, "s")
            compute_hap_cfp_dist(f, s, t, hap_mat_s, col_freqs_s, mut_pos_s,
                                 bacol=bacol, norm=1, recomb=True)

            hap_mat_s, col_freqs_s, mut_pos_s, bacol = hread.ms_hap_mat(f, s, t, sim, "s-noR")
            compute_hap_cfp_dist(f, s, t, hap_mat_s, col_freqs_s, mut_pos_s,
                                 bacol=bacol, norm=1, recomb=False)

            # mutation scatter
            # compute_hap_cfp_dist(f, s, t, hap_mat_n1, col_freqs_n1,
            #                      mut_pos_n1, bacol=None, norm=1)
            # compute_hap_cfp_dist(f, s, t, hap_mat_n1, col_freqs_n1,
            #                      mut_pos_n1, bacol=None, norm=2)
            # compute_hap_cfp_dist(f, s, t, hap_mat_s, col_freqs_s,
            #                      mut_pos_s, bacol=bacol, norm=1)
            # compute_hap_cfp_dist(f, s, t, hap_mat_s, col_freqs_s,
            #                      mut_pos_s, bacol=bacol, norm=2)

        # plot scatter for this simulation
        plot_hap_cfp_progress(sim)


###############################################################################
def compute_hap_cfp_dist(f, s, t, hap_mat, col_freqs, mut_pos, bacol=None, norm=1, recomb=True):
    ''' Computes a 1D scatter of haplotype CFP-scores in the given matrix.
        Stores in appropriate dict.
    '''
    global scatter_dict_s, scatter_dict_n  # used in subsequent call to plot_hap_cfp_progress

    # CFP-scores (y-values) for carriers & non-carriers of the b-allele
    y_yes_car, y_non_car = [], []

    # compute CFP score for each haplotype
    hap_scores = cfp.haplotype_CFP_scores(hap_mat, col_freqs, norm=norm)

    # update the maximal CFP score encountered so far
    update_y_max(hap_scores, norm)

    if bacol:
        # sweep, separate carriers and non-carriers
        for i in range(len(hap_mat)):
            if hap_mat[i, bacol] == 1.0:
                y_yes_car.append(hap_scores[i])
            else:
                y_non_car.append(hap_scores[i])
    else:
        # neutral, everyone is a non-carrier
        y_non_car = hap_scores

    # store in dict
    if bacol is None:
        scatter_dict_n[f, s, t, norm, recomb] = (y_yes_car, y_non_car)
    else:
        scatter_dict_s[f, s, t, norm, recomb] = (y_yes_car, y_non_car)


###############################################################################
def plot_hap_cfp_progress(sim):
    ''' make progression (1D) scatters of haplotype CFP-scores '''

    # initialize plot
    fig = plt.figure(figsize=(9, 6.5))  # width, height
    fig.subplots_adjust(bottom=0.09, hspace=0.17, left=0.09, right=0.96, top=0.96)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)

    # without recombination
    ax1 = fig.add_subplot(211)
    plot_progress_axis(ax1, 1, recomb=False, mean_cfp=(48*199.0/2.0), xlab=False)

    # with recombination
    ax2 = fig.add_subplot(212)
    plot_progress_axis(ax2, 1, recomb=True, mean_cfp=(48*199.0/2.0), xlab=True)

    # subplot labels
    fig.text(0.015, 0.970, r"$\mathbf{A}$", fontsize=17)
    fig.text(0.015, 0.510, r"$\mathbf{B}$", fontsize=17)

    # save figure
    plt.savefig('%s/hap_cfp_progress_f%.1f_s%g_sim%i.png' % (save_to_dir, f, s, sim), dpi=300)
    plt.savefig('%s/pdf/hap_cfp_progress_f%.1f_s%g_sim%i.pdf' % (save_to_dir, f, s, sim))
    plt.show()
    plt.close(fig)


###############################################################################
def plot_progress_axis(ax, norm, recomb, mean_cfp, xlab=True):
    ''' plot progression CFP-scores on axis '''

    # cummulative offset for breaking x-axis
    cum_off = 0
    x_ticks = []
    x_breaks = []

    for i, t in enumerate(times):

        # set y-values for this time point
        y_yes_car, y_non_car = scatter_dict_s[f, s, t, norm, recomb]

        # sanity
        assert len(y_yes_car) + len(y_non_car) == 200

        # add offset if x-axis broke at last time point
        if i-1 in breaks_after:
            cum_off = cum_off + 0.5
            x_breaks.append(i+cum_off)

        # set x-values for this time point, with random noise
        x_yes_car = add_random_nosie([i+cum_off]*len(y_yes_car))
        x_non_car = add_random_nosie([i+cum_off]*len(y_non_car))

        # remember for x-ticks
        x_ticks.append(i+cum_off)

        # boundary
        if (i+1 <= len(times)-1) and (i not in breaks_after):
            ax.axvline(((i+cum_off)*2+1)/2.0, color='#262626', linewidth=0.25)

        if len(y_yes_car) > len(y_non_car):
            # scatter yes-carriers, then non-carriers
            if len(y_yes_car) > 0:
                sct_yes_car = ax.scatter(x_yes_car, y_yes_car, c=c1[0], s=17, alpha=0.75,
                                         linewidths=0.3, edgecolor=c3[7], marker='o')
                # scats.append(sct_yes_car)
                # labels.append(r"$\mathbf{}$")

            if len(y_non_car) > 0:
                sct_non_car = ax.scatter(x_non_car, y_non_car, c=c1[1], s=17, alpha=0.75,
                                         linewidths=0.3, edgecolor=c3[7], marker='o')
                # scats.append(sct_non_car)
                # labels.append(r"$\mathbf{}$")
        else:
            # scatter non-carriers, then yes-carriers
            if len(y_non_car) > 0:
                sct_non_car = ax.scatter(x_non_car, y_non_car, c=c1[1], s=17, alpha=0.75,
                                         linewidths=0.3, edgecolor=c3[7], marker='o')
                # scats.append( sct_non_car )
                # labels.append( r"$\mathbf{}$" )

            if len(y_yes_car) > 0:
                sct_yes_car = ax.scatter(x_yes_car, y_yes_car, c=c1[0], s=17, alpha=0.75,
                                         linewidths=0.3, edgecolor=c3[7], marker='o')
                # scats.append( sct_yes_car )
                # labels.append( r"$\mathbf{}$" )

    # limits
    ax.set_xlim([0.0-x_max_noise-(i+cum_off)*0.02, (i+cum_off)+x_max_noise+(i+cum_off)*0.02])
    ax.set_ylim([-0.02*y_max[norm], y_max[norm]*1.02])

    # break x-axis
    d = .015  # how big to make the diagonal lines in axes coordinates
    x_breaks_axis = [0.43, 0.72]
    kwargs = dict(transform=ax.transAxes, color='#262626',
                  clip_on=False, linewidth=0.75, zorder=101)
    for x_b in x_breaks_axis:
        # two balck 'breaking' lines
        ax.plot((x_b-d*0.5, x_b+d*0.5), (-d, +d), **kwargs)
        ax.plot((x_b+(1.15*d)-d*0.5, x_b+(1.15*d)+d*0.5), (-d, +d), **kwargs)

        # white line over the broken part
        ax.plot((x_b, x_b+(1.14*d)), (0, 0),
                transform=ax.transAxes,
                color='white', linewidth=1.25,
                clip_on=False, zorder=100)

    # plot mean CFP as dotted line
    d = .03
    kwargs = dict(linestyle='dotted', linewidth=1.25, color='#262626')
    ax.axhline(mean_cfp, xmin=0, xmax=x_breaks_axis[0]-d*0.55, **kwargs)
    ax.axhline(mean_cfp, xmin=x_breaks_axis[0]+d*1.1, xmax=x_breaks_axis[1]-d*0.55, **kwargs)
    ax.axhline(mean_cfp, xmin=x_breaks_axis[1]+d*1.1, xmax=1,  **kwargs)

    if xlab:
        ax.set_xlabel(r"$\mathbf{Time\,\,(generations)}$", fontsize=12)
    else:
        # text annotations on top
        y_pos = 1.03
        kwargs = dict(transform=ax.transAxes, fontsize=12,
                      verticalalignment='bottom', horizontalalignment='center')
        ax.text((x_breaks_axis[0])/2, y_pos,
                r"$selective \,\, sweep$", **kwargs)
        ax.text((x_breaks_axis[0]+x_breaks_axis[1]+0.01)/2, y_pos,
                r"$post\mbox{-}fixation$", **kwargs)
        ax.text((x_breaks_axis[1]+1+0.02)/2, y_pos,
                r"$recovery$", **kwargs)

    ax.set_ylabel(r"$\mathbf{%i\mbox{-}CFP}$" % norm, fontsize=12)

    # pretify
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_linewidth(0.5)
    # ax.spines['top'].set_color('#262626')
    # ax.spines['right'].set_linewidth(0.5)
    # ax.spines['right'].set_color('#262626')
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
    # ax.xaxis.set_ticks_position('none') # remove x ticks
    # ax.yaxis.set_ticks_position('none') # remove y ticks

    # set x-axis ticks & labels
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(["$%i$" % x for x in times])

    # axis legend
    # legend = ax.legend(scats, labels, loc='upper right', scatterpoints=1,
    #                    markerscale=1.2, labelspacing=0.16,  # fontsize='x-small'
    #                    borderpad=0.3, handletextpad=0.06, prop={'size': 13},
    #                    shadow=False, fancybox=True)
    # rect = legend.get_frame()
    # rect.set_facecolor(np.array([float(247)/float(255)]*3))
    # rect.set_linewidth(0.0)  # remove line around legend
    # for t in legend.texts:
    #     t.set_color('#262626')


###############################################################################
def add_random_nosie(x_vals):
    ''' adds randome noise (at most x_max_noise) to each point in given list
    '''
    x_r = []

    # add random noise
    for i in range(len(x_vals)):
        x_r.append(x_vals[i] + random.uniform(-1.0, 1.0) * x_max_noise)

    return x_r


###############################################################################
def update_y_max(cfp_scores, norm):
    ''' updates y_max if necessary '''

    global y_max

    curr_max = max(cfp_scores)
    if(curr_max > y_max[norm]):
        y_max[norm] = curr_max


###############################################################################
if __name__ == '__main__':

    go()
