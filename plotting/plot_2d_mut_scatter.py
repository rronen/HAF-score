#!/usr/bin/env python

''' plotting utility for 2D mutation scatters '''

import sys, os, math, operator, random
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
################################ SETTINGS #####################################
###############################################################################

size_const = 15 # point size
max_noise_frac = 0.05 # max noise on x,y as fraction of max value

# (f,s,t) -> scatters
scatter_dict_s = {} # holds ( x_init, y_init, x_new, y_new, x_benef, y_benef )
scatter_dict_n = {} # holds ( x_init, y_init, x_new, y_new )

# sweep parameters
f = 0.3
s = 0.05
#times, h, w = [0, 25,  50, 100, 200, 400, 2000, 3000, 4000], 3, 3
times, h, w = [0, 25, 50, 75, 100, 150, 200, 250, 400, 600, 800, 1000, 1300, 1600, 1900, 2500], 4, 4

spp_tup, spl_tup = [ (f,s,times[i]) for i in range( len(times) ) ], [ (h,w,i+1) for i in range(h*w) ]
plot_layout = dict( zip( spp_tup, spl_tup ) )

c1 = brewer2mpl.get_map('Set1', 'Qualitative', 7).mpl_colors
c2 = brewer2mpl.get_map('Set2', 'Qualitative', 7).mpl_colors
cp = brewer2mpl.get_map('YlGn', 'Sequential' , 9).mpl_colors

# dir to save plots
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/cfp_freq_2d_scatters"

###############################################################################
############################## MAIN FUNCTION ##################################
###############################################################################
def go():
    
    for sim in range( p.last_sim ):

        for t in times:
            
            # read popuation sample files
            hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
            hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )
            # hap_mat_n2, col_freqs_n2, mut_pos_n2,   _   = hread.ms_hap_mat( f, s, t, sim, "n2" )
            
            # sanity check
            assert p.sample_size == len(hap_mat_s) == len(hap_mat_n1) #== len(hap_mat_n2)

            # init-pop mutation positions
            init_mut_pos = hread.ms_mut_pos( f, s, sim )

            # mutation scatter
            compute_scatter( f, s, t, hap_mat_n1, col_freqs_n1, mut_pos_n1, init_mut_pos )
            compute_scatter( f, s, t, hap_mat_s,  col_freqs_s,  mut_pos_s,  init_mut_pos, bacol )

        # plot scatter for this simulation
        plot_scatter( sim )

###############################################################################
def compute_scatter( f, s, t, hap_mat, col_freqs, mut_pos, init_mut_pos, bacol=None ):
    ''' Computes a 2D scatter for each mutation in given matrix, where x-axis is the 
        frequency and y-axis is the (mutation) CFP score. Stores in appropriate dict.
    '''
    global scatter_dict_s, scatter_dict_n # used in subsequent call to plot-scatter
    
    # x,y lists of current scatter
    x_init , y_init  = [], [] # old
    x_new  , y_new   = [], [] # new
    x_benef, y_benef = [], [] # benef

    # it = 3
    # mut_scores_dict, hap_scores_dict = cfp.score_mutations_and_haplotypes( hap_mat, cfp.h_score, cfp.m_score, it) # NEW

    # compute CFP score for each individual/row
    hap_scores = cfp.haplotype_CFP_scores( hap_mat, col_freqs )
    
    # make scatter (x,y per mutation)
    for j in range( len( col_freqs ) ):

        # mutation frequency
        af = col_freqs[j]

        # if ignoring fixed variants, ignore
        if( p.cfp_exclude_fixed and af == 1.0 ): continue
        
        # frequency (possibly folded)
        x_val = af
        # x_val = mut_scores_dict[0][j]
        
        # mutation CFP score
        y_val = cfp.mut_score_from_carrier_scores( hap_scores[ hap_mat[:, j] == 1.0 ] )
        # y_val = cfp.mut_score_from_carrier_scores( hap_scores_dict[0][ hap_mat[:, j] == 1.0 ] )
        # y_val = mut_scores_dict[it-1][j]

        # append to appropriate list
        if( not bacol is None and j == bacol):
            # beneficial allele
            x_benef.append(x_val)
            y_benef.append(y_val)
        elif( mut_pos[j] in init_mut_pos ):
            # old mutation
            x_init.append(x_val)
            y_init.append(y_val)
        else:
            # recent mutation
            x_new.append(x_val)
            y_new.append(y_val)

    # store in dict
    if( not bacol is None ):
        scatter_dict_s[ f, s, t ] = ( x_init, y_init, x_new, y_new, x_benef, y_benef )
    else:
        scatter_dict_n[ f, s, t ] = ( x_init, y_init, x_new, y_new )

###############################################################################
def plot_scatter( sim ):
    ''' make quad scatter plot of frequency vs. CFP score '''

    # initialize plot
    fig = plt.figure( figsize=(12.5, 11) ) # width, height
    fig.subplots_adjust(bottom=0.09, hspace=0.3, left=0.05, right=0.99,top=0.97)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex = True)

    # max-y for all subplots
    x_max, y_max, x_min, y_min = find_max_x_y()
    x_max_real, y_max_real = (1+max_noise_frac)*x_max, (1+max_noise_frac)*y_max

    for (f, s, t), subp in plot_layout.iteritems():

        ########
        # prep #
        ########
        scats, labels = [],[]
        ax = fig.add_subplot( subp[0], subp[1], subp[2] )

        ###################
        # neutral scatter #
        ###################
        x_init, y_init, x_new, y_new = scatter_dict_n[ f, s, t ]

        # add some random noise to scatter points
        x_neut,  y_neut = add_random_nosie( x_new+x_init, y_new+y_init, x_max*max_noise_frac, y_max*max_noise_frac )
        # x_new,  y_new  = add_random_nosie( x_new,  y_new, x_max*max_noise_frac, y_max*max_noise_frac )
        # x_init, y_init = add_random_nosie(x_init, y_init, x_max*max_noise_frac, y_max*max_noise_frac )

        # scatter
        if(x_neut):
            sct_samll = scatter( ax, x_new+x_init, y_new+y_init, size_const+3, alpha=.7, edgecolor=cp[6], marker='+', linewidths=1.05 ) # green
            scats.append( sct_samll )
            labels.append( r"$\mathbf{%i}$" % len( x_new+x_init ) )
        # if( x_init ):
        #     sct_samll = scatter( ax, x_init, y_init, size_const+3, alpha=.7, edgecolor=cp[6], marker='+', linewidths=1.05 ) # green
        #     scats.append( sct_samll )
        #     labels.append( r"$\mathbf{%i}$" % len( x_init ) )

        # if( x_new ):
        #     sct_samll = scatter( ax, x_new,  y_new,  size_const+3, alpha=1.0, edgecolor=cp[4], marker='+', linewidths=1.05 ) # lightgreen
        #     scats.append( sct_samll )
        #     labels.append( r"$\mathbf{%i}$" % len( x_new ) )

        #################
        # sweep scatter #
        #################
        x_init, y_init, x_new, y_new, x_benef, y_benef = scatter_dict_s[ f, s, t ]
        
        # scatter points list to set, with sizes
        x_new,  y_new  = add_random_nosie( x_new,  y_new,  x_max*max_noise_frac, y_max*max_noise_frac )
        x_init, y_init = add_random_nosie( x_init, y_init, x_max*max_noise_frac, y_max*max_noise_frac )

        # scatter
        if( x_init ):
            sct_small = scatter( ax, x_init, y_init, size_const, alpha=.7, edgecolor='0.97', color='blue', marker='o', linewidths=0.5) # c1[1]
            scats.append( sct_small )
            labels.append( r"$\mathbf{%i}$" % len( x_init ) )

        if( x_new ):
            sct_small = scatter( ax, x_new,  y_new,  size_const, alpha=.7, edgecolor='0.97', color=c1[0], marker='o', linewidths=0.5) # red
            scats.append( sct_small )
            labels.append( r"$\mathbf{%i}$" % len( x_new ) )

        if( x_benef ):
            sct = ax.scatter( x_benef , y_benef , s=180, linewidths=1.3, edgecolor='white', facecolor='black', marker='*' )
            # scats.append( sct )
            # labels.append( r"$\mathbf{1}$" )
        
        # save for figure legend
        # if( len( scats ) == 4 ): fig_legend_scats = scats
        if( len( scats ) == 3 ): fig_legend_scats = scats

        ############
        # finalize #
        ############
        # limits
        ax.set_xlim( [ x_min, x_max_real ] )
        ax.set_ylim( [ y_min, y_max_real*1.02 ] )

        # mutation filtering rules
        if( p.fold_freq ):
            ax.axhspan( -0.05, 1.75, xmin=-0.05, xmax=ax.transLimits.transform((0.25,0))[0], facecolor='none', alpha=0.4 )
        else:
            ax.axhspan( p.flt_cfp_h, y_max_real*1.1, xmin=-0.06, xmax=p.flt_freq_h, facecolor='0.9', alpha=1.0, 
                        edgecolor='none', zorder=-1000) # fill
            ax.axhspan( p.flt_cfp_h, y_max_real*1.1, xmin=-0.06, xmax=p.flt_freq_h, facecolor='none', alpha=.5, 
                        edgecolor='0.45',linewidth=0.8, zorder=-1000) # line

            ax.axhspan( -0.1, p.flt_cfp_l , xmin=-0.06, xmax=p.flt_freq_l, facecolor='0.9', alpha=1.0, 
                        edgecolor='none', zorder=-1000) # fill
            ax.axhspan( -0.1, p.flt_cfp_l , xmin=-0.06, xmax=p.flt_freq_l, facecolor='none', alpha=.5, 
                        edgecolor='0.45',linewidth=0.8, zorder=-1000) # line
        
        # title and axis labels
        ax.set_title( r"$\mathbf{\tau=%g}$" % t )
        if( ( subp[0] == subp[1] == 3 and subp[2] > 6 ) or ( subp[0] == subp[1] == 4 and subp[2] > 12 ) ):
            if( p.fold_freq ):
                ax.set_xlabel( r"$\mathbf{Folded \,\, Freq.}$" )
            else:
                ax.set_xlabel( r"$\mathbf{Freq.}$" )
        if( ( subp[0] == subp[1] == 3 and subp[2] in [1,4,7] ) or ( subp[0] == subp[1] == 4 and subp[2] in [1,5,9,13] ) ):
            ax.set_ylabel( r"$\mathbf{CFP\, score}$" )

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
        # ax.xaxis.set_ticks_position('none') # remove x ticks
        ax.yaxis.set_ticks_position('none') # remove y ticks

        # axis legend
        loc = 'lower right'
        if( not p.fold_freq ): 
            if( max( y_init + y_new ) < 4 and y_max > 5): loc = 'upper right'
        scats.append( sct ) # add the beneficial allele
        legend = ax.legend( scats, labels, loc=loc, scatterpoints=1, markerscale=1.2, labelspacing=0.16, #fontsize='x-small'
                            borderpad=0.3, handletextpad=0.06, prop={'size':11}, shadow=False, fancybox=True )
        rect = legend.get_frame()
        rect.set_facecolor( np.array([float(247)/float(255)]*3) )
        rect.set_linewidth(0.0) # remove line around legend
        for t in legend.texts: t.set_color('#262626')

    # figure legend
    # fig_leg_labs = [ r"$\mathbf{N_{ancient}}$", r"$\mathbf{N_{recent}}$", r"$\mathbf{S_{ancient}}$", 
                     # r"$\mathbf{S_{recent}}$", r"$\mathbf{beneficial}$" ]
    fig_leg_labs = [ r"$\mathbf{Neutral}$", r"$\mathbf{S_{ancient}}$", r"$\mathbf{S_{recent}}$", r"$\mathbf{Beneficial}$" ]
    legend = fig.legend( fig_legend_scats, fig_leg_labs, 'lower center', ncol=5, fancybox=True, scatterpoints=1, markerscale=1.2,
                            borderpad=0.4, labelspacing=0.175, shadow=False, handlelength=1, handletextpad=0.2, prop={'size':13} )

    rect = legend.get_frame()
    rect.set_facecolor( np.array([float(247)/float(255)]*3) )
    rect.set_linewidth(0.0) # remove line around legend
    for t in legend.texts: t.set_color('#262626')

    # save figure
    # plt.savefig( '%s/mut_scatter_f%.1f_s%g_t_sim%i.png' % ( save_to_dir,f, s, sim ) , dpi=300 )
    plt.savefig( '%s/pdf/mut_scatter_f%.1f_s%g_t_sim%i.pdf' % ( save_to_dir,f, s, sim ) )
    plt.show()
    plt.close( fig )

###############################################################################
def scatter( ax, x, y, sizes, alpha=1.0, edgecolor='w', color='b', marker='*', linewidths=1.5):
    ''' plot scatter '''

    sct = ax.scatter(x, y, c=color, s=sizes, linewidths=linewidths, edgecolor=edgecolor, marker=marker )
    return sct

###############################################################################
def add_random_nosie( x, y, x_max_noise, y_max_noise ):
    ''' Returns:
          (a) unique [x,y] points with linearly scaled sizes, or 
          (b) non-unique [x,y] points with random x,y noise added.
    '''
    assert len(x) == len(y) # sanity check

    x_r, y_r = [],[]

    # add random noise
    for i in range( len(x) ):

        # add at most max_noise to x and y coordinates
        x_r.append( x[i] + random.uniform(-1.0,1.0) * x_max_noise )
        y_r.append( y[i] + random.uniform(-1.0,1.0) * y_max_noise )

    return x_r, y_r

##############################################################################3
def find_max_x_y():
    
    x_max, y_max = 0, 0
    x_min, y_min = -0.05, -0.05

    for (f, s, t) in plot_layout:

        x_init_s, y_init_s, x_new_s, y_new_s, _ , _ = scatter_dict_s[ f, s, t ]
        x_init_n, y_init_n, x_new_n, y_new_n        = scatter_dict_n[ f, s, t ]

        curr_x_max, curr_x_min = max( x_init_n + x_new_n + x_init_s + x_new_s ), min( x_init_n + x_new_n + x_init_s + x_new_s )
        curr_y_max, curr_y_min = max( y_init_n + y_new_n + y_init_s + y_new_s ), min( y_init_n + y_new_n + y_init_s + y_new_s )
        
        if( curr_x_max > x_max ): x_max = curr_x_max
        if( curr_y_max > y_max ): y_max = curr_y_max
        
        if( curr_x_min < x_min ): x_min = curr_x_min
        if( curr_y_min < y_min ): y_min = curr_y_min

    
    return x_max, y_max, x_min, y_min

##############################################################################3
if __name__ == '__main__':

    go()
