#!/usr/bin/env python

''' plotting HFS historgrams '''

import os,sys, cPickle, matplotlib
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
from matplotlib import rcParams

''' internal imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p
import hap_reader_ms as hread
import hfs_utils as hfs

###############################################################################
################################### PARAMS ####################################
###############################################################################

use_saved = True # use pickled data

save_to_dir  = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/mean_HFS"       # dir to save plots
pck_data_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/mean_HFS/pck" # dir to save/read pickles

clust_2_name = { "exact":"Exact", "flt-f":"Freq.\, filter", "flt-f-cfp":"Freq.\, \&\, CFP\, filter" }
group_methods = [ "exact", "flt-f", "flt-f-cfp" ]

y_axis_subplots = [ (3,3,1), (3,3,4), (3,3,7), (4,3,1), (4,3,4), (4,3,7), (4,3,10) ]
x_axis_subplots = [ (3,3,7), (3,3,8), (3,3,9), (4,3,10), (4,3,11), (4,3,12) ]

c1 = brewer2mpl.get_map( 'Set1', 'Qualitative', 7 ).mpl_colors

################################################
#### mHFS different f,s,t comparison plots #####
################################################
f, s, times  = [0.0, 0.2, 0.4], 0.05, [75, 150, 400, 1000, 2000]
f_t_title = True

sp_indx_l  = [ ( 3, len(times), i+1 ) for i in range( len(times)*3 ) ]
sp_spec_l1 = [ ( f[0], s, times[i], group_methods[2] ) for i in range( len(times) ) ]
sp_spec_l2 = [ ( f[1], s, times[i], group_methods[2] ) for i in range( len(times) ) ]
sp_spec_l3 = [ ( f[2], s, times[i], group_methods[2] ) for i in range( len(times) ) ]

subplot = dict( zip( sp_spec_l1 + sp_spec_l2 + sp_spec_l3, sp_indx_l ) )

file_name = 'mHFS_f%.1f-%.1f_s%.2f_t%i-%i' % ( f[0], f[-1], s, times[0], times[-1] )

################################################
#### clustering technique comparison plots #####
################################################
# f, s, times  = 0.3, 0.05, [1200, 1500, 2000, 2500]
# f_t_title = False
# subplot = {
#     (f, s, times[0], group_methods[0]):(4,3,1),
#     (f, s, times[0], group_methods[1]):(4,3,2),
#     (f, s, times[0], group_methods[2]):(4,3,3),

#     (f, s, times[1], group_methods[0]):(4,3,4),
#     (f, s, times[1], group_methods[1]):(4,3,5),
#     (f, s, times[1], group_methods[2]):(4,3,6),

#     (f, s, times[2], group_methods[0]):(4,3,7),
#     (f, s, times[2], group_methods[1]):(4,3,8),
#     (f, s, times[2], group_methods[2]):(4,3,9),

#     (f, s, times[3], group_methods[0]):(4,3,10),
#     (f, s, times[3], group_methods[1]):(4,3,11),
#     (f, s, times[3], group_methods[2]):(4,3,12),
# }

# file_name = 'mHFS_f%.1f_s%.2f_t%i-%i' % ( f, s, times[0], times[-1] )

###############################################################################
def save_to_pck( f, s, t, group, hfs_s, hfs_n, mean_ba_freq, bin_edges ):
    ''' Writes given data to a dict and cPickle it '''

    dict_to_save = { "hfs_s": hfs_s, "hfs_n": hfs_n, "mean_ba_freq": mean_ba_freq, "bin_edges": bin_edges }
    
    with open( get_pck_fname( f, s, t, group ), mode='wb' ) as pck_fh: 
        cPickle.dump( dict_to_save, pck_fh )

###############################################################################
def get_from_pck( f, s, t, group ):
    ''' Reads HFS-s, HFS-n, and mean ba freq. from cPickle '''
    
    with open( get_pck_fname( f, s, t, group ), mode='rb' ) as pck_fh:
        saved_dict = cPickle.load( pck_fh )

    return saved_dict["hfs_s"], saved_dict["hfs_n"], saved_dict["mean_ba_freq"], saved_dict["bin_edges"]

###############################################################################
def get_pck_fname( f, s, t, group ):
    return pck_data_dir + "/" + "f%.2f_s%.3f_t%i_%s.pck" % ( f, s, t, group )

###############################################################################
def plot_mean_hfs_on_ax( ax, HFS_s_mean, HFS_n1_mean, bin_edges, group, f, s, t ):
    ''' plots mean HFS as lines on given axis. '''

    centers = (bin_edges[:-1]+bin_edges[1:])/2.0
    
    n_line, = ax.plot(centers, HFS_n1_mean, linewidth=1.1, color=c1[1], marker='^', markersize=4, 
                markeredgecolor=c1[1], mew=0.35, label=r"$\mathbf{Neutral}$" ) # blue

    s_line, = ax.plot(centers, HFS_s_mean,  linewidth=1.1, color=c1[0], marker='*', markersize=6, 
                markeredgecolor=c1[0], mew=0.35, label=r"$\mathbf{Sweep}$" ) # red

    if(HFS_s_mean[-1] > 0.05): ax.text(0.75, 0.9, "$%.2f$" % HFS_s_mean[-1], transform=ax.transAxes)

    return s_line, n_line

###############################################################################
def read_all_hfs( f, s, t, group ):
    ''' returns lists of HFS/mHFS vectors of sweep/neutral simulations under given parameters.
        also, returns the mean frequency of the beneficial allele in these simulations.
    '''

    HFS_s, HFS_n1, HFS_n2, mean_ba_freq = [], [], [], .0

    for sim in range( p.last_sim ):

        # read haplotype matrices
        hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
        hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )
        #hap_mat_n2, col_freqs_n2, mut_pos_n2,   _  = hread.ms_hap_mat( f, s, t, sim, "n2" )

        # compute HFS
        HFS_s.append(  hfs.get_hfs( hap_mat_s,  col_freqs_s,  group ) )
        HFS_n1.append( hfs.get_hfs( hap_mat_n1, col_freqs_n1, group ) )
        #HFS_n2.append( hfs.get_hfs( hap_mat_n2, col_freqs_n2, group ) )

        # mean b-allele freq.
        mean_ba_freq += col_freqs_s[ bacol ]

    return HFS_s, HFS_n1, mean_ba_freq / p.last_sim # ,HFS_n2

###############################################################################
def make_multi_plot():
    ''' Generates multi plot of HFS for given f, s, times and grouping methods.
    '''

    # prep plot
    matplotlib.rc('text', usetex = True)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    fig = plt.figure(figsize=(15, 8))
    fig.subplots_adjust(bottom=0.09, hspace=0.29, left=0.05, right=0.97,top=0.96)
    
    for ( f, s, t, group ), subp in sorted( subplot.iteritems() ):
        
        # read HFS
        if( use_saved and os.path.exists(get_pck_fname(f, s, t, group)) and group != 'flt-f-cfp' ): # use when clustering changes
            # use saved version
            hfs_s, hfs_n1, ba_freq_m, p.bin_edges = get_from_pck( f, s, t, group )
        else:
            # read and generate
            hfs_s, hfs_n1, ba_freq_m = read_all_hfs( f, s, t, group )
            save_to_pck( f, s, t, group, hfs_s, hfs_n1, ba_freq_m, p.bin_edges )

        # stack HFS arrays, and compute per-bin mean
        hfs_s, hfs_n1 = np.vstack( hfs_s ), np.vstack( hfs_n1 )
        hfs_s_m, hfs_n1_m = np.mean( hfs_s, axis=0 ), np.mean( hfs_n1, axis=0 )

        # plot mHFS
        ax = fig.add_subplot( subp[0], subp[1], subp[2] )
        s_line, n_line = plot_mean_hfs_on_ax( ax, hfs_s_m, hfs_n1_m, p.bin_edges, group, f, s, t )
        
        # plot mean b-allele freq.
        if( not p.fold_freq ):
            if( ba_freq_m < 1.0 and ba_freq_m > 0.0 ):
                ba_f_line = ax.axvline( ba_freq_m, color='#262626', linestyle='dotted', linewidth=1.3 )

        # ax.set_xscale('log', basex=2)
        ax.set_xlim( -0.01 , 1.000 )
        ax.set_ylim( -0.003, 0.055 ) # -0.003, None

        # title & axis labels
        if( f_t_title ):
            title = r"$\mathbf{f=%.1f, \tau=%i}$" % ( f, t )
        else:
            title = r"$\mathbf{%s, f=%.1f, \tau=%i}$" % ( clust_2_name[group], f, t )

        ax.set_title( title )
        if( subp in y_axis_subplots ): ax.set_ylabel( r"$\mathbf{Scaled\,\, Count}$" )
        if( subp in x_axis_subplots ): ax.set_xlabel( r"$\mathbf{Haolotype\,\, Freq.}$" )
        
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

        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        print subp

    # legend
    legend = fig.legend( [s_line, n_line ], [r"$\mathbf{Sweep}$", r"$\mathbf{Neutral}$"], 'lower center', ncol=2, fancybox=True, 
                         shadow=False, handlelength=3, handletextpad=0.2, prop={'size':13} )
    rect = legend.get_frame()
    rect.set_facecolor( np.array([247.0/255.0]*3) )
    rect.set_linewidth(0.0) # remove line around legend
    for t in legend.texts: t.set_color('#262626')

    # save plot
    plt.savefig( '%s/pdf/%s.pdf' % ( save_to_dir, file_name ), dpi=300 )
    plt.savefig( '%s/%s.png'     % ( save_to_dir, file_name ), dpi=300 )
    plt.show()
    plt.close(fig)

###############################################################################
################################## MAIN #######################################
###############################################################################
if __name__ == '__main__':
    
    make_multi_plot()

###############################################################################
############################ EXTERNALLY CALLED ################################
###############################################################################
def plot_mean_hfs( hfs_s, hfs_n1, hfs_n2, bins, bin_edges, scale_counts, group, f, s, t ):
    ''' Plots HFS histograms as lines for a single parametrization.
        Shows selection, neutral1 and neutral2.
    '''
   
    # compute mean & std-dev of HFS
    HFS_s_mean,  HFS_s_std  = np.mean(hfs_s, axis=0), np.std(hfs_s, axis=0)
    HFS_n1_mean, HFS_n1_std = np.mean(hfs_n1, axis=0), np.std(hfs_n1, axis=0)
    HFS_n2_mean, HFS_n2_std = np.mean(hfs_n2, axis=0), np.std(hfs_n2, axis=0)
       
    # plot type 
    errs, bars, lines = False, False, True
        
    # prep
    matplotlib.rc('text', usetex = True)
    fig = plt.figure(figsize=(14, 8))
    fig.subplots_adjust(bottom=0.05, hspace=0.35, left=0.05, right=0.97,top=0.96)
    ax = fig.add_subplot(111)
        
    # bin width
    width = 0.75*(bin_edges[1]-bin_edges[0])
    width = width/3.0

    center = (bin_edges[:-1]+bin_edges[1:])/2
    
    # plot mean historgrams
    if(lines): ax.plot(center, HFS_s_mean, color='red', linewidth=1.1, label=r"$\mathbf{sweep}$", marker='^')
    if(bars ): ax.bar(center+width*-1, HFS_s_mean, align='center', width=width, color='red'  , label=r"$\mathbf{sweep}$")
    if(errs ): ax.errorbar(center+width*-1, HFS_s_mean, yerr=HFS_s_std, marker='None', linestyle='None', ecolor='red', color='red')

    if(lines): ax.plot(center, HFS_n1_mean, color='blue', linewidth=1.1, label=r"$\mathbf{neutral1}$", marker='+')
    if(bars ): ax.bar(center+width*0, HFS_n1_mean, align='center', width=width, color='blue' , label=r"$\mathbf{neutral1}$")
    if(errs ): ax.errorbar(center+width*0, HFS_n1_mean, yerr=HFS_n1_std, marker='None', linestyle='None', ecolor='blue', color='blue')

    if(lines): ax.plot(center, HFS_n2_mean, color='green', linewidth=1.1, label=r"$\mathbf{neutral2}$", marker='*')
    if(bars ): ax.bar(center+width*1, HFS_n2_mean, align='center', width=width, color='green', label=r"$\mathbf{neutral2}$")
    if(errs ): ax.errorbar(center+width*1, HFS_n2_mean, yerr=HFS_n2_std, marker='None', linestyle='None', ecolor='green', color='green')

    # title and axis labels
    ax.set_title( r"$\mathbf{f=%g, s=%g, \tau=%i}$" % (f,s,t) )
    ax.set_xlabel( r"$\mathbf{Haplotype \,\, Frequency}$" )
    ax.set_ylabel( r"$\mathbf{Mean \,\, Scaled \,\, Count}$" )
    
    # get last frequency bin
    maximal = bins[-1]
    if(not scale_counts): maximal = max( max(np.nonzero(HFS_s_mean )[0]), max(np.nonzero(HFS_n1_mean)[0]), max(np.nonzero(HFS_n2_mean)[0]) )
    
    # axis range
    ymin, ymax = ax.get_ylim()
    ax.set_ylim(-0.03*ymax, 1.03*ymax)
    if(bars): ax.set_xlim(1.0-width*3, maximal-width*2)
    
    # legend
    ax.legend()

    # save plot
    plt.savefig( '%s/pdf/hfs_%s_f%g_s%g_t%i.pdf' % ( save_to_dir, group, f, s, t ), dpi=300 )
    plt.savefig( '%s/hfs_%s_f%g_s%g_t%i.png' % ( save_to_dir, group, f, s, t ), dpi=300 )
    plt.show()
    plt.close(fig)


