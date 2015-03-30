#!/usr/bin/env python

''' Plot CFP-score distribution (boxplot/violin) as function of:
    (i) beneficial allele frequency, and
    (ii) time since fixation
'''

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
import cPickle as pck
import scipy.stats as stats
from scipy.stats import gaussian_kde
from collections import defaultdict
from matplotlib import rcParams

''' internal imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p
import cfp_score as cfp
import hap_reader_ms as hread

###############################################################################
# ############################## SETTINGS #####################################
###############################################################################
axis_fontsize = 13  # 17

MIN_DATA_FOR_VIOLIN = 2000
DATA_TO_PLOT = p.last_sim

y_max = {1: 0, 2: 0}  # holds max observed CFP per norm
norm = 1

# sweep parameters
# f, s, f_step_size = 0.3, 0.05, 0.15
f, s, f_step_size = 0.0, 0.05, 0.10
times = p.times

scatter_dict_s = {}  # sim CFP scatters (f,s,t,sim,norm,"yes/no")
violin_dict = {}     # CFP violins (v)[0/1], 0=carrier scores, 1=noncarrier scores
violin_labels = []   # label for each violin

c1 = brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors
c2 = brewer2mpl.get_map('Set2', 'Qualitative', 8).mpl_colors
c3 = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors
cp = brewer2mpl.get_map('YlGn', 'Sequential', 9).mpl_colors

# dir to save plots
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/cfp_1d_progress_density"

# violins
violin_file_to_read, violin_info_to_read = None, None
violin_file_to_write = "/home/rronen/Desktop/freq_time_cfp_rank_violin_f%.1f_s%g.pck" % (f, s)
violin_info_to_write = "/home/rronen/Desktop/freq_time_cfp_rank_violin_info_f%.1f_s%g.pck" % (f, s)

# plot customization
post_fixation = True  # False

show_violin = True
violin_cut_top_1pc = True
show_bplot = True
w_p_vals = True
w_percentiles = True

# no p-values or rank percentiles for post fixation
if post_fixation:
    w_p_vals = False
if post_fixation:
    w_percentiles = False

# violins to plot
if post_fixation:
    # POST FIXATION (f=0.0, s=0.05)
    # regular intervals of 2N generations
    v_2_plot = [10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40, 44]
    fname_suff = "post_fix"
else:
    # PRE FIXATION (f=0.0, s=0.05)
    # [0.0-0.1, 0.1-0.2,..., 0.9-1.0]
    v_2_plot = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    fname_suff = "pre_fix"

# DEPRECATED
# 0.3,0.05 -> [0.3-0.45, 0.45-0.6, 0.6-0.75, 0.75-0.9, 0.9-1, 100, 200, 500, 3000]
# f=0.0, step=0.2
# v_2_plot = [0,1,2,3,4,6,9,30,40]
# fname_suff = "post_fix"

# DEPRECATED
# 0.0,0.04 -> [0-0.15, 0.15-0.3, 0.3-0.45, 0.45-0.6, 0.6-0.75, 0.75-0.9, 0.9-1, 100, 200, 500, 3000]
# f=0.3, step=0.15
# v_2_plot = [0,1,2,3,4,5,6  7,8,10,38]
# fname_suff = "post_fix"


###############################################################################
# ############################ MAIN FUNCTION ##################################
###############################################################################
def go():
    global violin_dict, violin_labels

    if violin_file_to_read:
        # read violins from file
        print "Reading violin data..."
        with open(violin_file_to_read, mode='r') as v_data_fh:
            violin_dict = pck.load(v_data_fh)
        with open(violin_info_to_read, mode='r') as v_info_fh:
            violin_labels = pck.load(v_info_fh)
        print "Read violins:", violin_labels
    else:
        # make violins, write to file

        # ############### 1 ##################
        # for each simulation
        # (1) compute haplotype CFP's
        # (2) get beneficial allele frequency
        # (3) get min-max time since fixation
        print "\n%s -- STARTED MAKING VIOLINS!\n" % datetime.datetime.fromtimestamp( time.time() ).strftime('%Y-%m-%d %H:%M:%S')
        print "Generating CFP data..."
        sim_freqs, min_t_since_fix, first_fix_t = [], [], {}
        for t in times:
            print "t=%i" % t,  # update
            for sim in range(DATA_TO_PLOT):

                # read haplotype data
                hap_mat_s, col_freqs_s, mut_pos_s, bacol = hread.ms_hap_mat(f, s, t, sim, "s")

                # compute haplotype CFP scores & save in 'scatter_dict_s'
                compute_hap_cfp_dist(f, s, t, sim, hap_mat_s, col_freqs_s, mut_pos_s, bacol=bacol, norm=1)

                # save b-allele frequency
                sim_freqs.append(col_freqs_s[bacol])

                # note time, if this is first time fixed
                if (col_freqs_s[bacol] == 1.0) and (sim not in first_fix_t):
                    first_fix_t[sim] = t

                # if fixed, find min. generations (100,200,...) since fixation
                # max. generations since fixation = +100 from what's found
                if sim in first_fix_t:
                    min_t_since_fix.append(t - first_fix_t[sim])
                else:
                    min_t_since_fix.append(-1)

        # ############### 2 ##################
        # bins beneficial allele frequencies and times since fixation,
        # then assign each simulation to its proper bin
        print "\nBinning simulations for freq. and time..."

        # set frequency bins for beneficial allele & assign simulations to bin
        f_bins = np.append( np.arange(f, 1.0, f_step_size), [1.0] )
        sim_freqs_bin_indices = np.digitize( sim_freqs, f_bins ) - 1
        print "freq bins:", f_bins
        n_freq_bins = len(f_bins) - 1

        # set time since fixation bins & assign simulations to bin
        t_bins = np.arange(0, 4001, 100)
        print "time bins:", t_bins
        sim_t_since_fix_bin_indices = np.digitize(min_t_since_fix, t_bins) - 1

        # init freq. violin dicts
        for v, (low_f, high_f) in enumerate(zip(f_bins, f_bins[1:])):
            print "Init freq violin %i (%g-%g)" % (v, low_f, high_f)
            violin_dict[v, "CFP"] = [[], [], []]   # lists of carrier & non-carrier CFP
            violin_dict[v, "CFP-rank"] = [[], []]  # lists of carrier & non-carrier CFP rank
            violin_dict[v, "Wilcoxon"] = []        # Wilcoxon rank-sum P-values
            violin_labels.append("%g-%g" % (low_f, high_f))

        # init time violin dicts
        for v, (low_t, high_t) in enumerate(zip(t_bins, t_bins[1:])):
            print "Init time violin %i (%i-%i)" % (v+n_freq_bins, low_t, high_t)
            violin_dict[v+n_freq_bins, "CFP"] = [[]]  # no non-carrier list
            violin_labels.append("%i-%i" % (low_t, high_t))

        # ############### 3 ##################
        # for each simulation, collect its (carrier & non-carrier)  CFP scores from 'scatter_dict_s'
        # into the appropriate (numbered) violin list
        print "Dispersing CFP data to appropriate violins..."
        i = 0
        for t in times:
            print "t=%i" % t,  # update
            for sim in range(DATA_TO_PLOT):
                if sim_freqs[i] < 1.0:
                    # sweep ongoing, place in appropriate frequency bin violin(s)
                    v = sim_freqs_bin_indices[i]
                    assert v >=-1 and v < len(f_bins)-1, "Error: bad freq bin for (non-fixed) sim (bin=%i, freq=%g)\n" % (v, sim_freqs[i])
                    if(v == -1): v = 0 # start-f is set in the population, so possibly samp freq < start-f
                                       # (it will always be close.) assign to first frequency bin
                    # CFP scores
                    violin_dict[v, "CFP"][0].extend( scatter_dict_s[ f, s, t, sim, norm, "yes" ] ) # carrier CFP scores
                    violin_dict[v, "CFP"][1].extend( scatter_dict_s[ f, s, t, sim, norm, "non" ] ) # non-carrier CFPs scores
                    
                    # CFP ranks
                    yes_pc, non_pc = rank_pc( scatter_dict_s[ f, s, t, sim, norm, "yes" ], scatter_dict_s[ f, s, t, sim, norm, "non" ] )
                    violin_dict[v, "CFP-rank"][0].extend( yes_pc ) # carrier CFP ranks/percentiles
                    violin_dict[v, "CFP-rank"][1].extend( non_pc ) # non-carrier CFP ranks/percentiles

                    # Wilcoxon rank sum test
                    # for early time points (t=0,25,50) there may be 0 carriers in the sample.
                    # in this case the test is meaningless, and stats.ranksums returns np.nan (handled later)
                    zscore, pval = stats.ranksums(scatter_dict_s[ f, s, t, sim, norm, "yes" ], scatter_dict_s[ f, s, t, sim, norm, "non" ])
                    violin_dict[v, "Wilcoxon"].append( pval )
                else:
                    # sweep fixed, place in appropriate time since fixation violin
                    v = sim_t_since_fix_bin_indices[i]
                    assert v >=0 and v < len(t_bins)-1, "Error: out of bounds time bin for fixed simulation\n"
                    violin_dict[v+n_freq_bins, "CFP"][0].extend( scatter_dict_s[ f, s, t, sim, norm, "yes" ] ) # only carriers

                i += 1

        # pickle computed violins & labels, for later use
        with open(violin_file_to_write, mode='wb') as v_data_fh: pck.dump(violin_dict, v_data_fh)
        with open(violin_info_to_write, mode='wb') as v_info_fh: pck.dump(violin_labels, v_info_fh)
        print "\n%s -- DONE MAKING VIOLINS!\n" % datetime.datetime.fromtimestamp( time.time() ).strftime('%Y-%m-%d %H:%M:%S')

    # FINALLY, plot violins...
    plot_cfp_progress_violins( with_p_values=w_p_vals, with_percentiles=w_percentiles )

###############################################################################
def rank_pc( cfp_yes, cfp_non ):
    ''' given two lists of CFP scores (carriers and non-carriers), returns
        two lists of the corresponding percentiles in the sorted CFP scores
    '''

    cfp_scores, pc_yes, pc_non = [], [], []

    for cfp in cfp_yes: cfp_scores.append( [cfp, "yes"] )
    for cfp in cfp_non: cfp_scores.append( [cfp, "non"] )
    
    # snaity check
    assert len(cfp_scores) == 200, "Error: unexpected number of CFP scores (%i)" % len(cfp_scores)

    n = float( len(cfp_scores) )
    for i, (cfp, status) in enumerate( sorted(cfp_scores) ):
        if( status == "yes" ):
            pc_yes.append( 100.0 * i / n )
        elif( status == "non" ):
            pc_non.append( 100.0 * i / n )

    return pc_yes, pc_non


###############################################################################
def compute_hap_cfp_dist(f, s, t, sim, hap_mat, col_freqs, mut_pos, bacol=None, norm=1):
    ''' Computes haplotype CFP-scores in the given matrix & stores in dict '''

    global scatter_dict_s  # used in subsequent call to plot_hap_cfp_progress

    # CFP-scores for carriers & non-carriers of the b-allele
    y_yes_car, y_non_car = [], []

    # compute CFP score for each haplotype
    hap_scores = cfp.haplotype_CFP_scores(hap_mat, col_freqs, norm=norm)

    # separate carriers and non-carriers
    for i in range(len(hap_mat)):
        if hap_mat[i, bacol] == 1.0:
            y_yes_car.append(hap_scores[i])
        else:
            y_non_car.append(hap_scores[i])

    scatter_dict_s[f, s, t, sim, norm, "yes"] = y_yes_car
    scatter_dict_s[f, s, t, sim, norm, "non"] = y_non_car


###############################################################################
def plot_cfp_progress_violins(with_p_values=False, with_percentiles=False):
    ''' make progression violins of haplotype CFP-scores '''

    OFFSET = 2

    # init plot
    if with_p_values and with_percentiles:
        fig = plt.figure(figsize=(10.5, 8.5))  # width, height
        fig.subplots_adjust(bottom=0.07, hspace=0.1, left=0.09, right=0.96, top=0.97)
    elif with_p_values:
        fig = plt.figure(figsize=(10.5, 7.5))  # width, height
        fig.subplots_adjust(bottom=0.09, hspace=0.1, left=0.09, right=0.96, top=0.97)
    else:
        fig = plt.figure(figsize=(10.5, 6))  # width, height
        fig.subplots_adjust(bottom=0.09, left=0.09, right=0.96, top=0.97)

    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)

    if with_p_values:
        # ax1 = fig.add_subplot( 211 ) # p-values
        # ax2 = fig.add_subplot( 212 ) # violins

        ax2 = fig.add_subplot(311)  # violins
        ax3 = fig.add_subplot(312)  # violins - rank
        ax1 = fig.add_subplot(313)  # p-values

    else:
        ax2 = fig.add_subplot(111)  # violins

    # plot violins
    data, pos, ticks, tick_l, colors = [], [], [], [], []
    data_rank = []

    # plot p-value scatter
    data_pval = []

    for i, v in enumerate(v_2_plot):
        print "plotting violin %s" % violin_labels[v]
        y_yes_car, y_non_car = [], []
        y_rank_yes_car, y_rank_non_car = [], []

        # process carrier violin
        y_yes_car = violin_dict[v, "CFP"][0]   # CFP scores
        if not post_fixation:
            y_rank_yes_car = violin_dict[v, "CFP-rank"][0]  # CFP score ranks

        # sufficient data check
        if len(y_yes_car) > MIN_DATA_FOR_VIOLIN:
            data.append(y_yes_car)
            if not post_fixation:
                data_rank.append(y_rank_yes_car)
            colors.append(c1[1])

        # update the maximal CFP score encountered so far
        update_y_max(y_yes_car, norm)

        # process non-carrier violin
        if len(violin_dict[v, "CFP"]) > 1:
            y_non_car = violin_dict[v, "CFP"][1]  # CFP scores
            y_rank_non_car = violin_dict[v, "CFP-rank"][1]  # CFP score ranks

            # sufficient data check
            if len(y_non_car) > MIN_DATA_FOR_VIOLIN:
                data.append(y_non_car)
                data_rank.append(y_rank_non_car)
                colors.append(c1[0])

            # plot P values from Wilcoxon rank test
            data_pval.append(violin_dict[v, "Wilcoxon"])

        if len(y_yes_car) > MIN_DATA_FOR_VIOLIN and len(y_non_car) > MIN_DATA_FOR_VIOLIN:
            pos.append(OFFSET + 3.5*i - 0.75)
            pos.append(OFFSET + 3.5*i + 0.75)
            ticks.append(OFFSET + 3.5*i)
        else:
            pos.append(OFFSET + 3.5*i)
            ticks.append(OFFSET + 3.5*i)

        tick_l.append(violin_labels[v])

    # plot CFP-scores as violins
    violin_plot_ax(ax2, data, pos, ticks, tick_l, colors,
                   boxp=show_bplot, violin=show_violin, is_raw_cfp=True)

    # plot p-values as scatter
    if with_p_values:
        scatter_p_val(ax1,  data_pval, pos, ticks, tick_l)

    # plot CFP score ranks as violins
    if with_percentiles:
        violin_plot_ax(ax3, data_rank, pos, ticks, tick_l, colors,
                       boxp=show_bplot, violin=show_violin, is_raw_cfp=False)

    # add letters
    if w_p_vals:
        fig.text(0.015, 0.97, r"$\mathbf{A}$", fontsize=17)
        fig.text(0.015, 0.67, r"$\mathbf{B}$", fontsize=17)
        fig.text(0.015, 0.37, r"$\mathbf{C}$", fontsize=17)
        if not post_fixation and norm == 1:
            fig.text(0.060, 0.969, r"$10^3$", fontsize=10)
            fig.text(0.079, 0.965, r"$\mathbf{x}$", fontsize=8)

    # save figure
    plt.savefig('%s/hap_%icfp_prog_%s_f%.1f_s%g.png' % (save_to_dir, norm, fname_suff, f, s),
                dpi=300)
    plt.savefig('%s/pdf/hap_%icfp_prog_%s_f%.1f_s%g.pdf' % (save_to_dir, norm, fname_suff, f, s))
    plt.show()
    plt.close(fig)


###############################################################################
def violin_plot_ax(ax, data, pos, ticks, tick_l, colors, violin=True, boxp=False, is_raw_cfp=True):
    ''' Plot a violin for each np.array in data
        'ax'    : the axis to plot on
        'data'  : list of np.array with data to visualize
        'pos'   : list with x-axis positions for respective violins
        'ticks' : list of tick positions
        'tick_l': list of labels for ticks
        'colors': list of colors for the violin
        'boxp'  : boolean, if True overlays boxplot for each violin [False]
    '''

    # plot neutral expectation
    if norm == 1 and is_raw_cfp:
        neutral_expec = 48.0*199.0 / 2.0
        if not post_fixation:
            neutral_expec = neutral_expec/1000.0  # normalize
        ax.axhline(neutral_expec, linestyle='dotted', linewidth=1.0, color='#262626', alpha=0.6)

    # violins
    dist = max(pos) - min(pos)
    w = min(0.15 * max(dist, 1.0), 0.5)

    # for boxplot, save all non-empty (valid) entries
    non_empty_data, non_empty_pos = [], []

    for i, (d, p) in enumerate(zip(data, pos)):

        d = np.array(d)

        # sanity check
        if len(d) == 0:
            continue

        # normalize
        if not post_fixation and is_raw_cfp and norm == 1:
            d = d / 1000.0

        mean_val = d.mean()

        # clean top 1% for raw CFP
        if violin_cut_top_1pc and is_raw_cfp:
            d = cut_top_1pc_data(d)

        non_empty_data.append(d)
        non_empty_pos.append(p)

        if violin:
            # compute kernel density
            k = gaussian_kde(d, 'silverman')  # 'scott'

            # support for violin
            x = np.linspace(k.dataset.min(), k.dataset.max(), 100)

            # violin profile (density curve)
            v = k.evaluate(x)

            # scaling the violin to the available space
            v = v/v.max()*w

            edgecol = colors[i]  # 'none'
            # ax.fill_betweenx(x, p,  v+p, facecolor=colors[i], edgecolor=edgecol, alpha=0.5)
            # ax.fill_betweenx(x, p, -v+p, facecolor=colors[i], edgecolor=edgecol, alpha=0.5)
            ax.fill_betweenx(x, p,  v+p, facecolor=colors[i], edgecolor='none', alpha=0.5)
            ax.fill_betweenx(x, p, -v+p, facecolor=colors[i], edgecolor='none', alpha=0.5)

            # show mean as single star
            ax.scatter([p], [mean_val], color='#262626', alpha=0.8, s=15, marker='*')

    # make boxplot
    if boxp:
        bp = ax.boxplot(non_empty_data, notch=1, positions=non_empty_pos,
                        vert=True, patch_artist=True, sym='')
        # pylab.setp(bp['fliers'], marker='None')

        from itertools import izip
        a = iter(range(len(non_empty_data)*2))
        for i, j in izip(a, a):
            # boxplot double components (above/below median)
            linecol, a = colors[i/2], 0.5
            if(violin and boxp):
                a = 0.75
                linecol = '#262626'

            pylab.setp(bp['whiskers'][i], color=linecol, alpha=0.75)
            pylab.setp(bp['whiskers'][j], color=linecol, alpha=0.75)
            pylab.setp(bp['caps'][i],     color=linecol, alpha=0.75)
            pylab.setp(bp['caps'][j],     color=linecol, alpha=0.75)
            # pylab.setp(bp['fliers'][i], color=linecol, marker='None', alpha=0.25)
            # pylab.setp(bp['fliers'][j], color=linecol, marker='None', alpha=0.25)

        for i in range(len(non_empty_data)):
            # boxplot single components
            facecol, a = colors[i], 0.5
            if(violin and boxp):
                a = 0.75
                facecol = 'none'

            pylab.setp(bp['boxes'][i],   color='#262626', facecolor=facecol, alpha=a)
            pylab.setp(bp['medians'][i], color='#262626', alpha=a)

    # axis ranges
    ax.set_xlim([0, max(pos) + min(pos)])
    if is_raw_cfp:
        if not post_fixation and norm == 1:
            ax.set_ylim([-0.01*(y_max[1]/1000.0), 17.0])
        else:
            ax.set_ylim([-0.01*y_max[1], None])
    else:
        ax.set_ylim([-2, 102])

    # pretify
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_linewidth(0.5)
    # ax.spines['top'].set_color('#262626')
    # ax.spines['right'].set_linewidth(0.5)
    # ax.spines['right'].set_color('#262626')

    if is_raw_cfp and (not w_p_vals):
        ax.spines['bottom'].set_linewidth(0.5)
        ax.spines['bottom'].set_color('#262626')
    else:
        ax.spines['bottom'].set_visible(False)
        ax.spines['bottom'].set_visible(False)

    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#262626')

    ax.get_yaxis().tick_left()

    if is_raw_cfp:
        # ticks only bottom & left
        ax.get_xaxis().tick_bottom()
        ax.tick_params(axis='x', colors='#262626')
        ax.tick_params(axis='y', colors='#262626')
        ax.xaxis.label.set_color('#262626')
        ax.yaxis.label.set_color('#262626')
    else:
        # tick only left
        ax.xaxis.set_ticks_position('none')  # remove x ticks
        ax.xaxis.set_major_locator(plt.NullLocator())  # remove x tick labels

    # ticks & tick labels
    if is_raw_cfp and not w_p_vals:
        ax.set_xticks(ticks)
        ax.set_xticklabels([r"$%s$" % polish_tick_label(x) for x in tick_l])
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])

    # x-axis label
    if is_raw_cfp and not w_p_vals:
        if fname_suff == 'pre_fix':
            ax.set_xlabel(r"$\mathbf{Adaptive \,\, allele \,\, frequency}$",
                          fontsize=axis_fontsize)
        else:
            ax.set_xlabel(r"$\mathbf{Time \,\, (2N \,\, generations) \,\, since \,\, fixation}$",
                          fontsize=axis_fontsize)

    # y-axis label
    if is_raw_cfp:
        if norm == 1:
            ax.set_ylabel(r"$\mathbf{%i\mbox{-}CFP}$" % norm, fontsize=axis_fontsize)
        else:
            ax.set_ylabel(r"$\mathbf{%i\mbox{-}CFP}$" % norm, fontsize=axis_fontsize)
    else:
        ax.set_ylabel(r"$\mathbf{Rank \,\, \%\mbox{-}tile}$", fontsize=axis_fontsize)


###############################################################################
def scatter_p_val(ax, data_p_vals, pos, ticks, tick_l):
    ''' '''
    max_noise = 0.5

    pos = np.array(pos)
    pos_centers = 0.5*(pos[0:-1:2]+pos[1::2])

    for i, (p_vals, x_pos) in enumerate(zip(data_p_vals, pos_centers)):

        p_vals = np.array(p_vals)

        # remove any np.nan, should only happen in first violin where it's
        # possible to have 0 carriers of the adaptive allele in the sample
        p_vals = p_vals[~np.isnan(p_vals)]

        # get % significant
        pc_sig = ( len(p_vals[ p_vals < 0.05 ]) / float(len(p_vals)) )

        # log transform p-values
        p_vals = -np.log2( p_vals )

        # get stutter noise
        x_pos_noise = [ random.uniform( x_pos - max_noise, x_pos + max_noise ) for p in p_vals ]

        # plot scatter
        ax.scatter( x_pos_noise, p_vals, facecolor=c1[1], alpha=0.25, edgecolor='#262626', linewidths=0.1, s=15)

        # show % significant over scatter
        if( pc_sig == 1.0 ):
            pc_sig = "1.0"
        else:
            pc_sig = "%g" % round(pc_sig,2)
        ax.text(x_pos, np.amax(p_vals)+2.5, r"$%s$" % pc_sig, fontsize=12, horizontalalignment='center', verticalalignment='bottom')


    # axis range
    ax.set_xlim([0, max(pos) + min(pos)])
    ax.set_ylim([-5, None])

    # line for statistical significance
    ax.axhline( -np.log2( 0.05 ), linewidth=1.0, linestyle='dotted', color='k', alpha=0.85)
    
    # pretify
    ax.spines['top'].set_visible( False )
    ax.spines['right'].set_visible( False )
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

    # ticks, tick labels, and axis labels
    ax.set_xticks( ticks )
    ax.set_xticklabels( [r"$%s$" % polish_tick_label(x) for x in tick_l] )
    ax.set_ylabel( r"$\mathbf{-log_2 P}$", fontsize=axis_fontsize )
    ax.set_xlabel( r"$\mathbf{Adaptive \,\, allele \,\, frequency}$", fontsize=axis_fontsize )

###############################################################################
def plot_hap_cfp_progress_heatmap():
    ''' make progression (1D) heatmapts of haplotype CFP-scores '''

    # initialize plot
    fig = plt.figure( figsize=(9, 6.5) ) # width, height
    fig.subplots_adjust(bottom=0.05, hspace=0.15, left=0.09, right=0.96,top=0.95)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex = True)
    
    # plot 1-CFP-scores
    for i,t in enumerate(times):
        ax = fig.add_subplot( 2, len(times), i+1 )
        plot_progress_axis( t, ax, 1, ylab=False )
    
    # plot 2-CFP-scores
    for i,t in enumerate(times):
        ax = fig.add_subplot( 2, len(times), len(times)+i+1 )
        plot_progress_axis( t, ax, 2, ylab=False )

    # save figure
    plt.savefig( '%s/hap_%icfp_progress_heat_f%.1f_s%g.png' % (save_to_dir,norm,f,s) , dpi=300 )
    plt.savefig( '%s/pdf/hap_%icfp_progress_heat_f%.1f_s%g.pdf' % (save_to_dir,norm,f,s) )
    plt.show()
    plt.close( fig )

###############################################################################
def plot_progress_axis( t, ax, norm, ylab=True ):
    ''' plot density (1D pcolor) of CFP-scores on axis '''

    # set y-values for this time point
    y_yes_car, y_non_car = scatter_dict_s[ f, s, t, norm ][0], scatter_dict_s[ f, s, t, norm ][1]

    # joint density of yes-carriers and non-carriers
    y_all = y_yes_car + y_non_car
    hist, binedges = np.histogram(y_all, bins=30, normed=True, range=None)
    hist_2d = np.vstack([hist]).T
    # im = ax.imshow(hist_2d, interpolation='nearest', alpha=0.9, cmap=plt.cm.Blues, origin='high', 
                    # extent=[binedges[0], binedges[-1],binedges[0], binedges[-1]])
    ax.pcolor(hist_2d, alpha=0.9, cmap=plt.cm.Blues)#, extent=[binedges[0], binedges[-1],binedges[0], binedges[-1]])

    # title & axis labels
    ax.set_title(r"$\mathbf{\tau=%i}$" % t)
    if(ylab): ax.set_ylabel( r"$\mathbf{%i\mbox{-}CFP \,\, score}$" % norm, fontsize=12 )

    # pretify
    # ax.spines['top'].set_visible( False )
    # ax.spines['right'].set_visible( False )
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['top'].set_color('#262626')
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['right'].set_color('#262626')
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
    ax.xaxis.set_ticks_position('none') # remove x ticks
    ax.xaxis.set_major_locator(plt.NullLocator())
    # ax.yaxis.set_ticks_position('none') # remove y ticks

    # legend
    # legend = fig.legend( objects, labels, loc='upper right', scatterpoints=1, markerscale=1.2, labelspacing=0.16, #fontsize='x-small'
    #                     borderpad=0.3, handletextpad=0.06, prop={'size':13}, shadow=False, fancybox=True )
    # rect = legend.get_frame()
    # rect.set_facecolor( np.array([float(247)/float(255)]*3) )
    # rect.set_linewidth(0.0) # remove line around legend
    # for t in legend.texts: t.set_color('#262626')


###############################################################################
def cut_top_1pc_data(data):
    ''' remove top 1 percentile of data '''
    data = np.array(data)
    cutoff = np.percentile(data, 98)
    return data[data < cutoff]


###############################################################################
def polish_tick_label(lab):
    ''' reduce latex math whitespace in string, '0 - 100' -> '0-100'
    '''

    r1, r2 = [float(x) for x in lab.split("-")]
    if r1 > 1.0:
        r1 = r1 / 2000.0
    if r2 > 1.0:
        r2 = r2 / 2000.0

    # return lab.replace('-', '\mbox{-}')
    return "%g\,\mbox{-}\,%g" % (r1, r2)


###############################################################################
def update_y_max(cfp_scores, norm):
    ''' update y_max, if necessary '''

    global y_max

    curr_max = max(cfp_scores)
    if curr_max > y_max[norm]:
        y_max[norm] = curr_max

###############################################################################
if __name__ == '__main__':

    if len(sys.argv) in [2, 3]:
        if len(sys.argv) == 3:
            violin_file_to_read = sys.argv[1]
            violin_info_to_read = sys.argv[2]
            go()
        elif sys.argv[1] == 'read-data':
            go()
    else:
        print "\n\tusage: %s \'read-data\' OR <violin-data.pck> <violin-labels.pck>\n" % sys.argv[0]
        sys.exit(1)
