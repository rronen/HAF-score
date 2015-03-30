#!/usr/bin/env python

''' plotting utility for haplotype CFP peak and trough durring a selective sweep '''

import sys
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pck
from collections import defaultdict
from matplotlib import rcParams
import brewer2mpl

''' internal imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p
import cfp_score as cfp
import hap_reader_ms as hread

###############################################################################
###############################################################################
###############################################################################

last_sim_expgr = 10000
last_sim = 200
f = 0.0
# selection = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
selection = [0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
norm = 1

# data structs
cfp_scores = {}        # (f, s, t, sim): [cfp_1, cfp_2, ..., cfp_n]
expgr_cfp_scores = {}  # (s): [cfp_1, cfp_2, ..., cfp_n]
fixation_times = {}    # s: [t_1, t_2, ..., t_{last_sim}]

# cPickle files to write
cfp_file_to_write_sweep = "/home/rronen/Desktop/cfp_data_rho=0_sweep_TAKE2.pck"
cfp_file_to_write_expgr = "/home/rronen/Desktop/cfp_data_rho=0_expgr_TAKE2.pck"

# sweep data (fixation times) paths
path_pref = "/home/rronen/Documents/selection/data/sim_soft_500_2.4e-07_s"
path_suff = "_f0.00_rho0/fixation_times.txt"

# exp. growth data (ms files) paths
expgr_pref = "/home/rronen/Documents/selection/data/sim_ms_neut_rho0_"
s2dir = {
    0.01: ("sim_ms_neut_rho0_s0.01_alpha18.04",  18.04),
    0.02: ("sim_ms_neut_rho0_s0.02_alpha32.35",  32.35),
    0.03: ("sim_ms_neut_rho0_s0.03_alpha45.59",  45.59),
    0.04: ("sim_ms_neut_rho0_s0.04_alpha55.86",  55.86),
    0.05: ("sim_ms_neut_rho0_s0.05_alpha67.06",  67.06),
    0.06: ("sim_ms_neut_rho0_s0.06_alpha79.37",  79.37),
    0.07: ("sim_ms_neut_rho0_s0.07_alpha89.96",  89.96),
    0.08: ("sim_ms_neut_rho0_s0.08_alpha100.31", 100.31),
}

# mean fixation time for s=0.05
# /home/rronen/Documents/selection/data/sim_soft_500_2.4e-07_s0.050_f0.00_rho0/Log
mean_fixation_time_s0_05 = 453.4

# dir to save plots
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/alpha_s_CFPs"
c1 = brewer2mpl.get_map('Set1', 'Qualitative', 9).mpl_colors

# conditional expectation values
# to be used instead of observed mean form 'ms'
# point (alpha=18.0352, cfp=477.709511) excluded
# due to our bad estimating of trough CFP when s is low (the 477.7)
glenn_cond_exp = {
    "alphas": [32.3506, 45.5874, 55.8623,  # 18.0352
               67.0569, 79.3735, 89.9607, 100.314],
    "cfps": [289.332045, 212.913737, 176.91418,  # 477.709511
             149.501505, 127.796829, 113.653657, 102.575707],
}


###############################################################################
###############################################################################
###############################################################################
def go(data_dict_file_sweep, data_dict_file_expgr):

    # (1) ### EXPONENTIAL GROWTH ###
    read_cfp_data_expgr(data_dict_file_expgr)

    # (2) ####### SELECTION ########
    # read selective sweep CFP data
    read_cfp_data_sweep(data_dict_file_sweep)

    # read fixation times
    read_fixation_times()

    # (3) ##### SUMARRY STATS ######
    s_2_mean_peak_CFPs, s_2_mean_trough_CFPs, alpha_2_mean_CFPs = peak_trough_stats()

    # (4) ######### PLOT ##########

    # initialize plot
    fig = plt.figure(figsize=(9, 4.5))  # width, height
    fig.subplots_adjust(bottom=0.14, wspace=0.2, left=0.09, right=0.97, top=0.87)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex=True)

    ax1 = fig.add_subplot(121)
    plot_alpha_s_CFPs(ax1, s_2_mean_peak_CFPs, s_2_mean_trough_CFPs, alpha_2_mean_CFPs)

    ax2 = fig.add_subplot(122)
    plot_peak_trough_diff(ax2, s_2_mean_peak_CFPs, s_2_mean_trough_CFPs)

    # letter
    fig.text(0.03, 0.96, r"$\mathbf{A}$", transform=ax1.transAxes,
             fontsize=21, verticalalignment='top')
    fig.text(0.52, 0.96, r"$\mathbf{B}$", transform=ax2.transAxes,
             fontsize=21, verticalalignment='top')

    # save figure
    plt.savefig('%s/relation.png' % (save_to_dir), dpi=300)
    plt.savefig('%s/pdf/relation.pdf' % (save_to_dir))
    plt.show()
    plt.close(fig)


###############################################################################
def plot_peak_trough_diff(ax, s_2_mean_peak_CFPs, s_2_mean_trough_CFPs):

    peaks, troughs, diffs = [], [], []
    for s in selection:
        peaks.append(s_2_mean_peak_CFPs[s])
        troughs.append(s_2_mean_trough_CFPs[s])

    peaks = np.array(peaks)
    troughs = np.array(troughs)
    diffs = peaks - troughs

    ind = np.arange(len(peaks))  # x locations for the groups
    width = 0.2                  # width of the bars

    rects1 = ax.bar(ind+0*width, peaks,   width, color=c1[0], alpha=0.7, linewidth=0.5)  # yerr=std)
    rects2 = ax.bar(ind+1*width, diffs,   width, color=c1[1], alpha=0.7, linewidth=0.5)  # yerr=std)
    rects3 = ax.bar(ind+2*width, troughs, width, color=c1[2], alpha=0.7, linewidth=0.5)  # yerr=std)

    # axis limits
    ax.set_xlim(ind[0]-0.5, ind[-1]+3*width+0.5)
    ax.set_ylim(0, 1.3*max(peaks))

    # neutral expectation and observed peak
    ax.axhline(48*199, linestyle='dotted', linewidth=1.25, color='#262626')

    # ticks
    ax.set_xticks(ind+1.5*width)
    ax.set_xticklabels([r'$0.02$', r'$0.03$', r'$0.04$', r'$0.05$',
                        r'$0.06$', r'$0.07$', r'$0.08$'])

    # axis label
    ax.set_xlabel(r"$\mathbf{selection \,\, coefficient \,\, (s)}$", fontsize=13)

    # legend
    legend = ax.legend([rects1[0], rects2[0], rects3[0]],
                       [r"$\mathbf{Peak}$", r"$\mathbf{Peak - Trough}$",
                        r"$\mathbf{Trough}$"],
                       loc='upper right', markerscale=1.1,
                       labelspacing=0.2,  handlelength=3.0,  # fontsize='x-small'
                       handleheight=0.3,
                       borderpad=0.3, handletextpad=0.7, prop={'size': 11},
                       shadow=False, fancybox=True)
    rect = legend.get_frame()
    rect.set_facecolor(np.array([float(247)/float(255)]*3))
    rect.set_linewidth(0.0)  # remove line around legend
    for t in legend.texts:
        t.set_color('#262626')

    # pretify
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
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


###############################################################################
def plot_alpha_s_CFPs(ax, s_2_mean_peak_CFPs, s_2_mean_trough_CFPs, alpha_2_mean_CFPs):

    # prep data
    x_s, y_trough_cfp = [], []
    for (s, mean_trough) in sorted(s_2_mean_trough_CFPs.iteritems()):
        x_s.append(s)
        y_trough_cfp.append(mean_trough)

    x_alpha, y_cfp = [], []
    if glenn_cond_exp:
        x_alpha = glenn_cond_exp["alphas"]
        y_cfp = glenn_cond_exp["cfps"]
    else:
        for (alpha, mean_cfp) in sorted(alpha_2_mean_CFPs.iteritems()):
            x_alpha.append(alpha)
            y_cfp.append(mean_cfp)

    ax2 = ax.twiny()

    # TRY WITH r RATHER THAN alpha
    # x_alpha = np.array(x_alpha) / 4000  # !!!! CHANGED FOR R !!!!

    # plot relationship
    s_line, = ax.plot(x_s, y_trough_cfp, color=c1[0], marker='.', linewidth=1.5)
    alpha_line, = ax2.plot(x_alpha, y_cfp, color=c1[1], marker='.', linewidth=1.5)

    # axis limits
    ax.set_xlim(0.015, 0.085)
    ax2.set_xlim(27, 105)
    # ax2.set_xlim(0.007, 0.026)  # !!!! CHANGED FOR R !!!!
    ax.set_ylim(min(y_trough_cfp + y_cfp)-0.1*max(y_trough_cfp + y_cfp),
                1.25*max(y_trough_cfp + y_cfp))

    # axis labels
    ax.set_ylabel(r"$\mathbf{1\mbox{-}CFP}$", fontsize=13)
    ax.set_xlabel(r"$\mathbf{selection \,\, coefficient \,\, (s)}$", fontsize=13)
    ax2.set_xlabel(r"$\mathbf{growth \,\, rate \,\, (\alpha)}$", fontsize=13)

    ax2.xaxis.labelpad = 8
    for tick in ax2.get_xaxis().get_major_ticks():
        tick.set_pad(0.75)
        tick.label1 = tick._get_text1()

    # tick labels
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=12)

    # legend
    lines = [alpha_line, s_line]
    labels = [r"$\mathbf{\,\,\,\, Expected \,\, exp. \,\, growth}$",
              r"$\mathbf{\,\,\,\, Mean \,\, `trough'}$"]
    legend = ax.legend(lines, labels, loc='upper right', markerscale=1.1,
                       labelspacing=0.2,  handlelength=3.0,  # fontsize='x-small'
                       borderpad=0.3, handletextpad=0.7, prop={'size': 11},
                       shadow=False, fancybox=True)
    rect = legend.get_frame()
    rect.set_facecolor(np.array([float(247)/float(255)]*3))
    rect.set_linewidth(0.0)  # remove line around legend
    for t in legend.texts:
        t.set_color('#262626')
# pretify
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['top'].set_color('#262626')
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['right'].set_color('#262626')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['bottom'].set_color('#262626')
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['left'].set_color('#262626')

    # ticks only bottom and left
    # ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='x', colors='#262626')
    ax.tick_params(axis='y', colors='#262626')
    ax.xaxis.label.set_color('#262626')
    ax.yaxis.label.set_color('#262626')


###############################################################################
def read_fixation_times():
    '''
        read fixation times from files
    '''
    global fixation_times

    for s in selection:

        if s == 0.05:
            # DEBUG
            # mean_t = mean_fixation_time_s0_05
            # print "s=%g, t=%g, alpha=%g" % (s, mean_t, np.log(2000)/(mean_t/4000.0))
            continue  # special case, only have the mean

        f_path = path_pref + "%.3f" % s + path_suff
        times = []
        with open(f_path, 'r') as fix_t_fh:
            for line in fix_t_fh:
                times.append(float(line.rstrip()))

        fixation_times[s] = np.array(times)

        # DEBUG
        # mean_t = np.mean(fixation_times[s])
        # print "s=%g, t=%g, alpha=%g" % (s, mean_t, np.log(2000)/(mean_t/4000.0))


###############################################################################
def read_cfp_data_expgr(data_dict_file_expgr):
    ''' read CFP score data from
            - ms files, or
            - input cPickle file
    '''
    global expgr_cfp_scores

    if data_dict_file_expgr:
        # load from pck file
        print "\nReading (expgr) CFP data from '%s'...\n" % data_dict_file_expgr
        with open(data_dict_file_expgr, mode='r') as cfp_data_fh:
            expgr_cfp_scores = pck.load(cfp_data_fh)

    else:
        for s, (dirname, alpha) in s2dir.iteritems():

            print "\nReading (expgr) data for s=%g" % s
            h_cfps = []

            for sim in range(last_sim_expgr):

                # progress
                if sim % 1000 == 0 and sim > 0:
                    print "\tRead %i expgr simulations" % sim

                # read haplotype matrix
                hap_mat, col_freqs, positions = hread.ms_neut_mat(sim, dirname)

                # haplotype CFP-scores
                cfps = cfp.haplotype_CFP_scores_ms(hap_mat, col_freqs, norm)

                # save haplotype CFP scores
                h_cfps.extend(cfps)

            expgr_cfp_scores[s] = np.array(h_cfps)

        # save data dict
        with open(cfp_file_to_write_expgr, mode='wb') as cfp_data_fh:
            pck.dump(expgr_cfp_scores, cfp_data_fh)


###############################################################################
def read_cfp_data_sweep(data_dict_file_sweep):
    ''' read CFP score data from
            - mpop files, or
            - input cPickle file
    '''

    global cfp_scores

    if data_dict_file_sweep:
        # load from pck file
        print "\nReading (sweep) CFP data from '%s'...\n" % data_dict_file_sweep
        with open(data_dict_file_sweep, mode='r') as cfp_data_fh:
            cfp_scores = pck.load(cfp_data_fh)

    else:
        # read from mpop files
        for s in selection:

            print "\nReading (sweep) data for s=%g" % s

            for sim in range(last_sim):
                sys.stdout.write("%i," % sim)
                sys.stdout.flush()

                for t in p.times:
                    try:
                        hap_mat_s, col_freqs_s, mut_pos_s, bacol = hread.ms_hap_mat(f, s, t,
                                                                                    sim, "s-noR")

                        # DEBUG
                        # print "Watt. theta=%g" % (len(col_freqs_s)/harmonic_sum(len(hap_mat_s)))
                        # int_freq = col_freqs_s * len(hap_mat_s)
                        # print int_freq
                        # print len(int_freq)
                        # print len(int_freq[int_freq > 185])

                    except IOError:
                        break

                    # compute CFP score for each haplotype
                    cfps = cfp.haplotype_CFP_scores(hap_mat_s, col_freqs_s, norm=norm)
                    cfp_scores[f, s, t, sim] = cfps

                    # DEBUG
                    # print "%g" % np.mean(cfps)

        # save data dict
        with open(cfp_file_to_write_sweep, mode='wb') as cfp_data_fh:
            pck.dump(cfp_scores, cfp_data_fh)


###############################################################################
def peak_trough_stats():

    # init return values
    s_2_mean_peak_CFPs = {}
    s_2_mean_trough_CFPs = {}
    alpha_2_mean_CFPs = {}

    for s in selection:

        print "s=%g" % s

        peak_mean_CFPs, trough_mean_CFPs = [], []
        peak_times, trough_times = [], []

        for sim in range(last_sim):

            # DEBUG
            # print "\nsim: %i" % sim

            # extreme (mean) CFPs for current simulation
            peak_mean, trough_mean = 0, np.inf
            peak_mean_t, trough_mean_t = None, None

            for t in p.times:
                if (f, s, t, sim) not in cfp_scores:
                    assert t > 0, "Not even 1 simulation??? (s=%g, sim=%i)" % (s, sim)
                    break  # times greater than this not simulated

                hap_scores = cfp_scores[f, s, t, sim]

                mean_cfp = np.mean(hap_scores)

                if mean_cfp > peak_mean:
                    peak_mean = mean_cfp
                    peak_mean_t = t
                    # DEBUG
                    # print "peak-mean=%g (time=%i)" % (peak_mean, peak_mean_t)

                if mean_cfp < trough_mean:
                    trough_mean = mean_cfp
                    trough_mean_t = t

            # save extreme (mean) CFPs for current simulation
            if peak_mean > 0:
                peak_mean_CFPs.append(peak_mean)
                peak_times.append(peak_mean_t)

            if trough_mean < np.inf:
                trough_mean_CFPs.append(trough_mean)
                trough_times.append(trough_mean_t)

        s_2_mean_peak_CFPs[s] = np.mean(peak_mean_CFPs)
        print "\tPeak CFP: mean=%g, sd=%g" % (s_2_mean_peak_CFPs[s], np.std(peak_mean_CFPs)),
        print "(time: mean=%g, sd=%g)" % (np.mean(peak_times), np.std(peak_times))

        if s == 0.05:
            print "\tFixation time: mean=%g" % mean_fixation_time_s0_05
        else:
            print "\tFixation time: mean=%g, sd=%g" % (np.mean(fixation_times[s]),
                                                       np.std(fixation_times[s]))

        s_2_mean_trough_CFPs[s] = np.mean(trough_mean_CFPs)
        print "\tTrough CFP: mean=%g, sd=%g" % (s_2_mean_trough_CFPs[s],
                                                np.std(trough_mean_CFPs)),
        print "(time: mean=%g, sd=%g)" % (np.mean(trough_times), np.std(trough_times))

        alpha = s2dir[s][1]
        alpha_2_mean_CFPs[alpha] = np.mean(expgr_cfp_scores[s])
        print "\tExp. Growth corresponding: mean=%g, sd=%g" % (alpha_2_mean_CFPs[alpha],
                                                               np.std(expgr_cfp_scores[s]))

    return s_2_mean_peak_CFPs, s_2_mean_trough_CFPs, alpha_2_mean_CFPs


###############################################################################
def harmonic_sum(last_term):
    ''' returns the sum of 1/i for i=(1,...,last_term-1) '''

    h_sum = 0.0
    for i in range(1, last_term):
        h_sum += 1.0/i
    return h_sum


###############################################################################
if __name__ == '__main__':

    if len(sys.argv) not in [2, 3]:
        print "\n\tusage:\n"
        print "\t\t" + "%s 'read-data'" % sys.argv[0]
        print "\t\tOR"
        print "\t\t" + "%s <cfp_data_sweep.pck> <cfp_data_expgr.pck>\n" % sys.argv[0]
    else:
        if sys.argv[1] == 'read-data':
            go(data_dict_file_sweep=None, data_dict_file_expgr=None)
        else:
            go(data_dict_file_sweep=sys.argv[1], data_dict_file_expgr=sys.argv[2])
