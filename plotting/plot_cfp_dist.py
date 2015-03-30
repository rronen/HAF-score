#!/usr/bin/env python

import sys
import os
import math
import matplotlib
import brewer2mpl
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rcParams
from scipy.stats import gaussian_kde

''' internal imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p
import cfp_score as cfp
import hap_reader_ms as hread

###############################################################################
#                                   PARAMS                                    #
###############################################################################

# expectation computation
N = 1000 * 2.0          # effective population size, haplotypes
n = 200                 # sample size, haplotype
wsize = 50000.0         # window size, base pairs
bp_mu = 2.4*(10.0**-7)  # from paper (scaled for N=1000, 10x of truth), half in future
bp_theta = 2.0*N*bp_mu  # 2N\mu, where N is *haploid* population size
window_theta = bp_theta * wsize

# CFP-score norm
cfp_norm = 1

# directory
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/cfp_scores"

c1 = brewer2mpl.get_map('Set1', 'Qualitative', 7).mpl_colors

# location of neutral simulations
ms_neut_dirname = "sim_ms_neut_rho0"
ms_neut_dirname = "sim_ms_neut_rho0_alpha80.00"

# filename to save
if p.fold_freq:
    # file_name = "neutral_cfp_scores_folded"
    file_name = "neutral_growth_cfp_scores_folded"    # !!!!! CHANGED FOR EXP. GROWTH !!!!!
else:
    # file_name = "neutral_cfp_scores"
    file_name = "neutral_growth_cfp_scores"    # !!!!! CHANGED FOR EXP. GROWTH !!!!!

# cutoff for historgram values (s.t. range isn't dominated by outliers)
top_val_cutoff = 30000
top_val_cutoff = 700    # !!!!! CHANGED FOR EXP. GROWTH alpha=80 !!!!!


# empirical distribution
nbins = 150
nbins = 250  # !!!!! CHANGED FOR EXP. GROWTH alpha=80 !!!!!


###############################################################################
def plot_score_hist(scores):
    ''' plot distribution of scores as historgram '''

    # prep plot
    matplotlib.rc('text', usetex=True)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.08, hspace=0.28, left=0.12, right=0.95, top=0.94)

    # so outliers don't dominate the range
    scores = np.array(scores)
    scores2plot = scores[scores < top_val_cutoff]

    # plot historgram
    ax = fig.add_subplot(111)
    ax.hist(scores2plot, bins=nbins, normed=True, color='w', linewidth=0.75, edgecolor='#262626')
    # ax.set_title("$\mathbf{CFP^%i, \,\, Neutral \,\, Coalescent}$" % cfp_norm)
    ax.set_xlabel(r"$\mathbf{%i\mbox{-}CFP}$" % cfp_norm)
    ax.set_ylabel(r"$\mathbf{Density}$")
    plt.xlim((-0.02*max(scores2plot), None))

    # pretify
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#262626')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#262626')

    # ticks only on bottom & left
    ax.tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', colors='#262626')
    ax.tick_params(axis='y', colors='#262626')
    ax.xaxis.label.set_color('#262626')
    ax.yaxis.label.set_color('#262626')

    # plot 10 & 90 percentile
    # ax.axvline(np.percentile(scores, 10), color='gray', linestyle='dotted', linewidth=1.5)
    # ax.axvline(np.percentile(scores, 90), color='gray', linestyle='dotted', linewidth=1.5)

    # expectated CFP
    if cfp_norm == 1:
        exp = CFP_neutral_expectation_1norm()
        exp = 126.86  # !!!!! CHANGED FOR EXP. GROWTH !!!!!
    elif cfp_norm == 2:
        exp = CFP_neutral_expectation_2norm()

    # plot expectation (blue)
    exp_line = ax.axvline(exp, color=c1[1], linestyle='dashed', linewidth=1.75)

    # plot mean (red)
    mean = np.mean(scores, dtype=np.float64)
    mean_line = ax.axvline(mean, color=c1[0], linestyle='dotted', linewidth=2.25)

    # legend
    lines = [exp_line, mean_line]
    labels = [r"$\mathbf{Expectation \,\, (%.1f)}$" % exp,
              r"$\mathbf{Mean \,\, (%.1f)}$" % mean]
    legend = ax.legend(lines, labels, 'upper right', prop={'size': 12},
                       handlelength=3.5, columnspacing=1, labelspacing=0.1,
                       handletextpad=0.25, fancybox=True, shadow=False)
    # bbox_to_anchor=[0.5, -0.005], ncol=len(stats_2_plot), # ncol=4,

    rect = legend.get_frame()
    rect.set_facecolor(np.array([float(247)/float(255)]*3))
    rect.set_linewidth(0.0)  # remove edge
    texts = legend.texts
    for t in texts:
        t.set_color('#262626')

    plt.savefig('%s/%s.png' % (save_to_dir, file_name), dpi=300)
    plt.savefig('%s/pdf/%s.pdf' % (save_to_dir, file_name))
    plt.show()
    plt.close(fig)


###############################################################################
def plot_theta_est(theta_Ws):

    # prep plot
    matplotlib.rc('text', usetex=True)
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.08, hspace=0.28, left=0.10, right=0.95, top=0.94)

    # plot historgram
    ax = fig.add_subplot(111)

    ax.hist(theta_Ws, bins=nbins, normed=True, color='w', linewidth=0.75, edgecolor='#262626')
    ax.set_title(r"$\mathbf{Watterson\mbox{'}s \,\, \theta, \,\, Neutral \,\, Coalescent}$")
    ax.set_xlabel(r"$\mathbf{\theta_w}$")
    ax.set_ylabel(r"$\mathbf{Density}$")

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

    ax.tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    # text & vertical lines
    ax.text(0.8, 0.90, r"$\mathbf{\theta=%g}$" % window_theta,
            transform=ax.transAxes, fontsize=15)
    ax.text(0.8, 0.85, r"$\mathbf{\bar{\theta}_w=%g}$" %
            np.mean(theta_Ws, dtype=np.float64), transform=ax.transAxes, fontsize=15)

    ax.axvline(window_theta, color=c1[1],
               linestyle='dashed', linewidth=1.75)  # blue
    ax.axvline(np.mean(theta_Ws, dtype=np.float64), color=c1[0],
               linestyle='dashed', linewidth=1.75)  # red

    # save plot
    # file name
    if p.fold_freq:
        file_name = "wattersons_theta_folded"
    else:
        file_name = "wattersons_theta"
    plt.savefig('%s/%s.png' % (save_to_dir, file_name), dpi=300)
    plt.savefig('%s/pdf/%s.pdf' % (save_to_dir, file_name))
    plt.show()
    plt.close(fig)


###############################################################################
def get_neutral_scores():
    ''' get many haplotype CFP-scores from *neutrally evolving* samples '''

    num_neut_sim = 20000
    h_cfps = []
    theta_Ws = []

    for sim in range(num_neut_sim):

        # progress
        if sim % 1000 == 0:
            print "read %i simulations" % sim

        # read haplotype matrix
        hap_mat, col_freqs, positions = hread.ms_neut_mat(sim, ms_neut_dirname)

        # haplotype CFP-scores
        cfps = cfp.haplotype_CFP_scores_ms(hap_mat, col_freqs, cfp_norm)

        # save haplotype CFP scores
        h_cfps.extend(cfps)

        # save Watterson's theta
        theta_Ws.append(len(col_freqs) / harmonic_sum(n))

    print "\nComputing histogram over %i scores" % len(h_cfps)

    return np.array(h_cfps), np.array(theta_Ws)


###############################################################################
def CFP_neutral_expectation_1norm():
    ''' expected haplotype 1-CFP score under neutral coalescent '''

    return window_theta * (n-1.0) / 2.0


###############################################################################
def CFP_neutral_expectation_2norm():
    ''' expected haplotype 2-CFP score under neutral coalescent '''

    return window_theta * (n- 1.0) * (2*n - 1.0) / 6.0


###############################################################################
def harmonic_sum(last_term):
    ''' returns the sum of 1/i for i=(1,...,last_term-1) '''

    h_sum = 0.0
    for i in range(1, last_term):
        h_sum += 1.0/i
    return h_sum


###############################################################################
# ################################# MAIN ######################################
###############################################################################
if __name__ == '__main__':

    # print "theta=%g" % window_theta
    print "E(CFP^1)=%g" % CFP_neutral_expectation_1norm()
    print "E(CFP^2)=%g" % CFP_neutral_expectation_2norm()

    # read simulated data & compute stats
    h_CFPs, theta_Ws = get_neutral_scores()
    print "avg(CFP^%i)=%g" % (cfp_norm, h_CFPs.mean())

    # plot stats of interest
    plot_score_hist(h_CFPs)  # plot_theta_est(theta_Ws)


###############################################################################
# ################################## OLD ######################################
###############################################################################
def plot_cfp_dists(scores_s, scores_n1, scores_n2, s, t):
    ''' plot distributions of CFP scores as 3 lines (s, n1, n2) '''

    # compute means and variances
    mean_s, var_s = np.mean(scores_s), np.var(scores_s)
    mean_n1, var_n1 = np.mean(scores_n1), np.var(scores_n1)
    mean_n2, var_n2 = np.mean(scores_n2), np.var(scores_n2)
    lab_s = r"$\mathbf{Sweep[1]} \,    (\mu=%.2f, \, \sigma^2=%.2f)$" % (mean_s, var_s)
    lab_n1 = r"$\mathbf{Sweep[0]} \,    (\mu=%.2f, \, \sigma^2=%.2f)$" % (mean_n1, var_n1)
    lab_n2 = r"$\mathbf{Neutral} \,\,\, (\mu=%.2f, \, \sigma^2=%.2f)$" % (mean_n2, var_n2)

    # compute histograms
    hist_s, bin_edges_s = np.histogram(scores_s, bins=50, density=True)
    centers_s = (bin_edges_s[:-1]+bin_edges_s[1:])/2

    hist_n1,  bin_edges_n1 = np.histogram(scores_n1, bins=50, density=True)
    centers_n1 = (bin_edges_n1[:-1]+bin_edges_n1[1:])/2

    hist_n2,  bin_edges_n2 = np.histogram(scores_n2, bins=50, density=True)
    centers_n2 = (bin_edges_n2[:-1]+bin_edges_n2[1:])/2

    # prep plot
    lines, labels = [], []
    matplotlib.rc('text', usetex=True)
    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.05, hspace=0.35, left=0.05, right=0.97, top=0.96)
    ax = fig.add_subplot(111)

    # plot distributions
    line, = ax.plot(centers_s,  hist_s,  color='red', linewidth=1)
    lines.append( line )
    labels.append( lab_s )
    
    line, = ax.plot(centers_n1, hist_n1, color='blue', linewidth=1)
    lines.append( line )
    labels.append( lab_n1 )
    
    line, = ax.plot(centers_n2, hist_n2, color='green', linewidth=1.05)
    lines.append( line )
    labels.append( lab_n2 )
    
    # finalize plot
    ax.set_title(r"$\mathbf{CFP\, score\, distribution \, (s=%g, t=%g)}$" % ( s, t ) )
    ax.set_xlabel(r"$\mathbf{CFP \,\, Score}$")
    ax.set_ylabel(r"$\mathbf{Density}$")
    ax.legend(lines, labels, 'upper right', prop={'size':13}, fancybox=True, shadow=True)
    plt.savefig('%s/cfp_dist_s%g_t%g.pdf' % (save_to_dir, s, t) , dpi=350)
    plt.show()
    plt.close(fig)


###############################################################################
def violin_plot( data, pos, ticks, tick_l, s, t, boxp=False ):
    ''' Plot a violin for each np.array in data
        
        'data'  : list of np.array with data to visualize
        'pos'   : list with x-axis positions for respective violins
        'ticks' : list of tick positions
        'tick_l': labels for ticks
        'boxp'  : boolean, if True overlays boxplot for each violin [False]
    '''
    
    # prep figure
    matplotlib.rc('text', usetex = True)
    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.05, hspace=0.35, left=0.05, right=0.97,top=0.96)
    ax = fig.add_subplot(111)
    
    # violins
    dist = max(pos) - min(pos)
    w = min( 0.15 * max(dist,1.0), 0.5)
    
    # for boxplot, save all non-empty (valid) entries
    non_empty_data, non_empty_pos = [], []
    
    for i, (d, p) in enumerate( zip(data,pos) ):
        
        # sanity check
        if len(d) == 0: continue
        non_empty_data.append(d)
        non_empty_pos.append(p)
        
        # compute kernel density
        k = gaussian_kde(d, 'scott') # 'silverman'
        
        # support for violin
        x = np.linspace( k.dataset.min(), k.dataset.max(), 100 )
        
        # violin profile (density curve)
        v = k.evaluate(x)
        
        # scaling the violin to the available space
        v = v/v.max()*w
        
        # color fill density
        color = 'green'
        if i % 3 == 0 :
            color = 'red'
        elif (i-1) % 3 == 0: 
            color = 'yellow'
        
        ax.fill_betweenx(x, p,  v+p, facecolor=color, alpha=0.3)
        ax.fill_betweenx(x, p, -v+p, facecolor=color, alpha=0.3)

    # make boxplot
    if boxp: ax.boxplot( non_empty_data, notch=1, positions=non_empty_pos, vert=1 )

    # finalize figure
    ax.set_ylabel(r"$\mathbf{CFP \,\, Score}$")
    ax.set_xlim([0, max(pos) + min(pos)])
    ax.set_title(r"$\mathbf{CFP\, score\, distributions \, (s=%g, t=%g)}$" % ( s, t ) )
    plt.xticks(ticks, tick_l)
    
    # save plot
    if( len( data ) > 3):
        save_to = '%s/cfp_dist_violin_fbins_s%g_t%g.pdf' % (save_to_dir, s, t)
    else:
        save_to = '%s/cfp_dist_violin_s%g_t%g.pdf' % (save_to_dir, s, t)
    
    plt.savefig(save_to , dpi=350)
    
    # show
    plt.show()
    plt.close(fig)
