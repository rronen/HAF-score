#!/usr/bin/env python

import sys, os, matplotlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import NullLocator
import brewer2mpl

times = [450,700,1000,1500]

c1 = brewer2mpl.get_map( 'Set1', 'Qualitative', 7 ).mpl_colors
# t2col = {450:'blue', 700:'black', 1000:'red', 1500:'green'}
t2col = {450:c1[1], 700:'#262626', 1000:c1[0], 1500:c1[2]}

t2lab = { 450 : r'$\mathbf{\tau=450}$' ,
          700 : r'$\mathbf{\tau=700}$' ,
         1000 : r'$\mathbf{\tau=1000}$',
         1500 : r'$\mathbf{\tau=1500}$'}

tests = {"Spi":121, "EHH":122}
test_names = {"Spi":r"$\mathbf{S_{\pi}}$", "EHH":r"$\mathbf{EHH}$"}

test_t_ss_2_pow = {} # 450, 5 -> 0.5

max_ss = 80 # 80, 100
s_sizes = range(2,max_ss+1,2)

save_to = "/home/rronen/Dropbox/UCSD/HA_selection/Grant_NIH/SampleSizeFig/samp_size_EHH_Spi_to%i.pdf" % max_ss

###############################################################################
def plot_samp_size():

    # pretify: latex font, outward ticks
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex = True)

    # figure
    fig = plt.figure(figsize=(12.5, 5.5))
    fig.subplots_adjust(hspace=0.2, wspace=0.13,left=0.06, right=0.98, top=0.95, bottom=0.10)

    for test,subp in tests.iteritems():
        read_power(test) # read power files
        labels, lines = [], []
        ax = fig.add_subplot( subp )
        ax.set_ylim(0, 105)
        ax.set_xlim(0, max_ss+2)

        for t in times:
            x, y = [x for x in s_sizes], [test_t_ss_2_pow[test,t,ss] for ss in s_sizes]
            line, = ax.plot(x, y, linestyle='-', linewidth=1.15, color=t2col[t], label=t2lab[t] ) # marker=get_marker(stat)

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
        # ax.xaxis.set_ticks_position('none') # remove x ticks
        # ax.yaxis.set_ticks_position('none') # remove y ticks

        # axis labels
        if(subp == 121):
            ax.set_ylabel(r"$\mathbf{\% \,\, Optimal \,\, Power}$", fontsize=13)
        ax.set_xlabel(r'$\mathbf{Sample \,\, Size\,\,(haplotypes)}$', fontsize=13)
        ax.set_title( test_names[test] )

        ax.axvline(x=20, ymin=0, ymax=100, linewidth=1.5, color='0.5', linestyle='dotted')
        ax.axvline(x=40, ymin=0, ymax=100, linewidth=1.5, color='0.5', linestyle='dotted')
        legend = ax.legend(loc='lower right')
        rect = legend.get_frame()
        rect.set_facecolor( np.array([float(247)/float(255)]*3) )
        rect.set_linewidth(0.0) # remove edge
        texts = legend.texts
        for t in texts: t.set_color('#262626')

    # panel labels
    fig.text(0.05, 1.0, "A", fontsize=15)
    fig.text(0.55, 1.0, "B", fontsize=15)

    # save file
    plt.savefig(save_to, dpi=550)
    plt.show()

###############################################################################
def read_power(test):

    for t in times:

        F = open("/home/rronen/Documents/selection/data/Sample_size_experiments/s0.02_t%i_%s.txt" % (t,test))
        for line in F:
            if line.startswith('#') : continue
            spl = line.rstrip().split()
            t, ss, p = float(spl[1]), float(spl[2]), min(float(spl[4]), 100.0)
            test_t_ss_2_pow[test,t,ss] = p

        F.close()

###############################################################################
################################## MAIN #######################################
###############################################################################
if __name__ == '__main__':
    
    plot_samp_size()