#!/usr/bin/env python

''' plotting utility for 2D mutation scatters '''

import sys, os, math, operator, random
import matplotlib
import pylab
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
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

MIN_DATA_FOR_VIOLIN = 2000
DATA_TO_PLOT = p.last_sim

y_max = {1:0, 2:0} # max CFP-score per norm

# sweep parameters
f, s = 0.3, 0.05
#f, s = 0.0, 0.04

# times points post selection
# times  = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
# times += [i for i in range(600,2001,100)]

times  = [25, 50, 100, 150, 200, 500, 1500, 3000, 4000] # 400, 800 instead of 500
# times  = [0, 400, 800, 1600, 3200, 4000]

scatter_dict_s = {} # (f,s,t)[0] -> carrier CFP-scores, (f,s,t)[1] -> noncarrier CFP-scores
scatter_dict_n = {} # (f,s,t)[0] -> carrier CFP-scores, (f,s,t)[1] -> noncarrier CFP-scores

# init dicts
for t in times: 
    scatter_dict_s[f,s,t,1] = [ [],[] ]
    scatter_dict_s[f,s,t,2] = [ [],[] ]
    scatter_dict_n[f,s,t,1] = [ [],[] ]
    scatter_dict_n[f,s,t,2] = [ [],[] ]

c1 = brewer2mpl.get_map('Set1' , 'Qualitative', 9).mpl_colors
c2 = brewer2mpl.get_map('Set2' , 'Qualitative', 8).mpl_colors
cp = brewer2mpl.get_map('YlGn' , 'Sequential' , 9).mpl_colors
c3 = brewer2mpl.get_map('Dark2', 'Qualitative', 8).mpl_colors

# dir to save plots
save_to_dir = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/cfp_1d_progress_density"

###############################################################################
############################## MAIN FUNCTION ##################################
###############################################################################
def go():
    
    for t in times:

        for sim in range( DATA_TO_PLOT ):
        
            # read popuation sample files
            hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
            #hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )
            #hap_mat_n2, col_freqs_n2, mut_pos_n2,   _   = hread.ms_hap_mat( f, s, t, sim, "n2" )
            
            # sanity check
            #assert p.sample_size == len(hap_mat_s) == len(hap_mat_n1) #== len(hap_mat_n2)

            # init-pop mutation positions
            #init_mut_pos = hread.ms_mut_pos( f, s, sim )

            # mutation scatter
            #compute_hap_cfp_dist( f, s, t, hap_mat_n1, col_freqs_n1, mut_pos_n1, bacol=None,  norm=1 )
            #compute_hap_cfp_dist( f, s, t, hap_mat_n1, col_freqs_n1, mut_pos_n1, bacol=None,  norm=2 )
            compute_hap_cfp_dist( f, s, t,  hap_mat_s,  col_freqs_s,  mut_pos_s,  bacol=bacol, norm=1 )
            # compute_hap_cfp_dist( f, s, t,  hap_mat_s,  col_freqs_s,  mut_pos_s,  bacol=bacol, norm=2 )

    # plot scatter for this simulation
    # plot_hap_cfp_progress_heatmap()
    plot_hap_cfp_progress_violin()

###############################################################################
def compute_hap_cfp_dist( f, s, t, hap_mat, col_freqs, mut_pos, bacol=None, norm=1 ):
    ''' Computes a 1D scatter of haplotype CFP-scores in the given matrix.
        Stores in appropriate dict.
    '''
    global scatter_dict_s, scatter_dict_n # used in subsequent call to plot_hap_cfp_progress
    
    # CFP-scores (y-values) for carriers & non-carriers of the b-allele
    y_yes_car, y_non_car = [], []

    # compute CFP score for each haplotype
    hap_scores = cfp.haplotype_CFP_scores( hap_mat, col_freqs, norm=norm )
    
    # update the maximal CFP score encountered so far
    update_y_max(hap_scores, norm)

    if( bacol ):
        # sweep, separate carriers and non-carriers
        for i in range( len(hap_mat) ):
            if( hap_mat[i,bacol] == 1.0 ):
                y_yes_car.append( hap_scores[i] )
            else:
                y_non_car.append( hap_scores[i] )
    else:
        # neutral, everyone is a non-carrier
        y_non_car = hap_scores

    # store in dict
    if( bacol is None ):
        scatter_dict_n[ f, s, t, norm ][0].extend( y_yes_car )
        scatter_dict_n[ f, s, t, norm ][1].extend( y_non_car )
    else:
        scatter_dict_s[ f, s, t, norm ][0].extend( y_yes_car )
        scatter_dict_s[ f, s, t, norm ][1].extend( y_non_car )

###############################################################################
def plot_hap_cfp_progress_violin():
    ''' make progression violins of haplotype CFP-scores '''
    # initialize plot
    fig = plt.figure( figsize=(10.5, 6) ) # width, height
    fig.subplots_adjust(bottom=0.09, hspace=0.15, left=0.09, right=0.96,top=0.97)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex = True)

    OFFSET = 2
    norm = 1
    ax = fig.add_subplot( 111 )

    # generate violins
    data, pos, ticks, tick_l, colors = [], [], [], [], []
    for i,t in enumerate(times):
        y_yes_car, y_non_car = scatter_dict_s[ f, s, t, norm ][0], scatter_dict_s[ f, s, t, norm ][1]

        if( len(y_yes_car) > MIN_DATA_FOR_VIOLIN): 
            data.append( y_yes_car )
            colors.append(c1[0])
        if( len(y_non_car) > MIN_DATA_FOR_VIOLIN): 
            data.append( y_non_car )
            colors.append(c1[1])
        
        if( len(y_yes_car) > MIN_DATA_FOR_VIOLIN and len(y_non_car) > MIN_DATA_FOR_VIOLIN ): 
            pos.append( OFFSET + 3.5*i - 0.75 )
            pos.append( OFFSET + 3.5*i + 0.75 )
            ticks.append( OFFSET + 3.5*i   )
        else:
            pos.append( OFFSET + 3.5*i )
            ticks.append( OFFSET + 3.5*i )

        tick_l.append( r"$%i$" % t )

    # plot 1-CFP-scores as violins
    violin_plot_ax( ax, data, pos, ticks, tick_l, colors, boxp=True, violin=True )

    # plot neutral expectation as line
    if( norm == 1 ): ax.axhline( (48.0*199.0) / 2.0, linestyle='--', linewidth=0.85, color='#262626')

    # pretify
    ax.spines['top'].set_visible( False )
    ax.spines['right'].set_visible( False )
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
    # ax.xaxis.set_major_locator(plt.NullLocator()) # remove x tick labels
    # ax.yaxis.set_ticks_position('none') # remove y ticks
    # ax.yaxis.set_major_locator(plt.NullLocator()) # remove y tick labels

    # axis labels and range
    ax.set_xlabel( r"$\mathbf{Time \,\, (generations)}$" )
    ax.set_ylabel( r"$\mathbf{%i\mbox{-}CFP\mbox{-}core}$" % norm)
    # ax.set_title(r"$\mathbf{CFP\, score\, distributions \, (s=%g)}$" % s )
    ax.set_ylim( [ -0.02*y_max[1], None] ) # 14000

    # legend
    # legend = fig.legend( objects, labels, loc='upper right', scatterpoints=1, markerscale=1.2, labelspacing=0.16, #fontsize='x-small'
    #                     borderpad=0.3, handletextpad=0.06, prop={'size':13}, shadow=False, fancybox=True )
    # rect = legend.get_frame()
    # rect.set_facecolor( np.array([float(247)/float(255)]*3) )
    # rect.set_linewidth(0.0) # remove line around legend
    # for t in legend.texts: t.set_color('#262626')

    # save figure
    plt.savefig( '%s/hap_%icfp_progress_violin_f%.1f_s%g.png' % (save_to_dir,norm,f,s) , dpi=300 )
    plt.savefig( '%s/pdf/hap_%icfp_progress_violin_f%.1f_s%g.pdf' % (save_to_dir,norm,f,s) )
    plt.show()
    plt.close( fig )

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
    ''' plot density of CFP-scores on axis '''

    # set y-values for this time point
    y_yes_car, y_non_car = scatter_dict_s[ f, s, t, norm ][0], scatter_dict_s[ f, s, t, norm ][1]

    # joint density of yes-carriers and non-carriers
    y_all = y_yes_car + y_non_car
    hist, binedges = np.histogram(y_all, bins=30, normed=True, range=None)
    hist_2d = np.vstack([hist]).T
    # im = ax.imshow(hist_2d, interpolation='nearest', alpha=0.9, cmap=plt.cm.Blues, origin='high', 
                    # extent=[binedges[0], binedges[-1],binedges[0], binedges[-1]])
    ax.pcolor(hist_2d, alpha=0.9, cmap=plt.cm.Blues)#, extent=[binedges[0], binedges[-1],binedges[0], binedges[-1]])
    
    # limits
    # ax.set_xlim( [ None, None] )
    # ax.set_ylim( [ None, None] )

    # title & axis labels
    ax.set_title(r"$\mathbf{\tau=%i}$" % t)
    if(ylab): ax.set_ylabel( r"$\mathbf{%i\mbox{-}CFP\mbox{-}score}$" % norm, fontsize=12 )

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

###############################################################################
def violin_plot_ax(ax, data, pos, ticks, tick_l, colors, violin=True, boxp=False):
    ''' Plot a violin for each np.array in data
        'ax'    : the axis to plot on
        'data'  : list of np.array with data to visualize
        'pos'   : list with x-axis positions for respective violins
        'ticks' : list of tick positions
        'tick_l': list of labels for ticks
        'colors': list of colors for the violin
        'boxp'  : boolean, if True overlays boxplot for each violin [False]
    '''

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
        
        if( violin ):
            # compute kernel density
            k = gaussian_kde(d, 'silverman') # 'scott'
            
            # support for violin
            x = np.linspace( k.dataset.min(), k.dataset.max(), 100 )
            
            # violin profile (density curve)
            v = k.evaluate(x)
            
            # scaling the violin to the available space
            v = v/v.max()*w
            
            ax.fill_betweenx(x, p,  v+p, facecolor=colors[i], alpha=0.35)
            ax.fill_betweenx(x, p, -v+p, facecolor=colors[i], alpha=0.35)

    # make boxplot
    if( boxp ):
        bp = ax.boxplot( non_empty_data, notch=1, positions=non_empty_pos, sym='', vert=True )
        pylab.setp(bp['boxes'], color='#262626')
        pylab.setp(bp['whiskers'], color='#262626')
        pylab.setp(bp['fliers'], marker='None')

    # finalize
    ax.set_xlim([0, max(pos) + min(pos)])
    plt.xticks(ticks, tick_l)

##############################################################################3
def update_y_max( cfp_scores, norm ):
    ''' updates y_max if necessary '''

    global y_max

    curr_max = max(cfp_scores)
    if(curr_max > y_max[norm]): y_max[norm] = curr_max

##############################################################################3
if __name__ == '__main__':

    go()
