#!/usr/bin/env python

''' plotting HFS historgrams '''

import os,sys, matplotlib, operator
import matplotlib.pyplot as plt
import numpy as np
import brewer2mpl
from matplotlib import rcParams
from sklearn import svm

''' internal imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p
import cfp_score as cfp
import hap_reader_ms as hread
import learn

###############################################################################
################################### PARAMS ####################################
###############################################################################
nbins = 65 # for hexbin density

##############################################################
freq_strat, strat_freq = True, 0.00 # 0.05 -> 0.0 for all (but fixed) SNPs in learning/plot
cfp_or_freq_strat  = True # False for all SNPs in learning
##############################################################

# axis boundaries
if( freq_strat ): 
    min_x = strat_freq
else:
    min_x = 0.0

if( p.fold_freq ):
    max_x = 0.51
    max_y = 5
else:
    max_x = 1.01
    max_y = 7.5

# sweep parameters
f = 0.3
s = 0.05
#times, h, w = [0, 25,  50, 100, 200, 400, 2000, 3000, 4000], 3, 3
times, h, w = [0, 25, 50, 75, 100, 150, 200, 250, 400, 600, 800, 1000, 1300, 1600, 1900, 2500], 4, 4
times, h, w = [1000], 1, 1

# plot layout
spp_tup, spl_tup = [ (f,s,times[i]) for i in range( len(times) ) ], [ (h,w,i+1) for i in range(h*w) ]
plot_layout = dict( zip( spp_tup, spl_tup ) )

# dir to save plots 
save_to_dir  = "/home/rronen/Dropbox/UCSD/workspace/SoftSweep/plots/cfp_freq_2d_density"

# file name extension
if( p.fold_freq ):
    folded = "folded_"
else:
    folded = ""

###############################################################################
def make_densities():

    matplotlib.rc('text', usetex = True)
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    fig = plt.figure( figsize=(14, 10) )
    fig.subplots_adjust( bottom=0.06, hspace=0.29, left=0.05, right=0.96,top=0.96 )

    # general neutral class for learning ("0" label)
    neut_x, neut_y = get_2d_data( 0.0, 0.05, 0, "n1" )
    
    # startify for leraning, unless folded
    neut_x_strat, neut_y_strat = strat_on_cfp_or_freq( neut_x, neut_y, 5.0, 0.8 )

    for (f, s, t), subp in sorted( plot_layout.iteritems(), key=operator.itemgetter(1) ):

        plt.subplot( subp[0], subp[1], subp[2] )
        
        # get sweep data
        x,y = get_2d_data( f, s, t, "s" )

        # startify for leraning, unless folded
        x_strat, y_strat = strat_on_cfp_or_freq( x, y, 5.0, 0.8 )
        
        # lear classifier
        clf = learn.learn_2d_freq_cfp_classifier( neut_x_strat, neut_y_strat, x_strat, y_strat )

        # learn & plot decision bounds
        plot_decision_bounds( clf )

        # plot hexabin density
        plot_2d_density( x, y, subp[2], t )

    plt.savefig( '%s/pdf/%sfreq_cfp_2d_density_f%.1f_s%g.pdf' % ( save_to_dir, folded, f, s ) )
    plt.savefig( '%s/%sfreq_cfp_2d_density_f%.1f_s%g.png' % ( save_to_dir, folded, f, s ), dpi=400 )
    plt.show()
    plt.close( fig )

###############################################################################
def plot_decision_bounds( clf ):
    ''' plot to current axis the decision boundaries (sweep vs. neutral) of given classifier '''
    
    # plot decision boundaries
    ax = plt.gca()
    if( type(clf) == svm.LinearSVC ):
        # line for linear (2d) decision function
        w = clf.coef_[0]
        a = -w[0] / w[1]
        xx = np.linspace( min_x, max_x )
        yy = a * xx - ( clf.intercept_[0] ) / w[1]
        ax.plot( xx, yy, 'k-', linewidth=1.5 )
    else:
        # contour for non-linear decision function
        xx, yy = np.meshgrid( np.linspace(min_x, max_x, 300), np.linspace(0, max_y, 500) )

        # evaluate decision function for each grid point
        Z = clf.decision_function( np.c_[xx.ravel(), yy.ravel()] )
        Z = Z.reshape(xx.shape)

        # plot
        # ax.imshow(Z, interpolation='nearest', extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto', origin='lower', cmap=plt.cm.PuOr_r)
        contours = ax.contour( xx, yy, Z, levels=[0], linewidths=1.5, linetypes='--' )

###############################################################################
def get_2d_data( f, s, t, sim_type ):

    m_cfps, m_freqs = [], []

    for sim in range( p.last_sim ):

        # read haplotype matrix
        hap_mat, col_freqs, _ , _  = hread.ms_hap_mat( f, s, t, sim, sim_type )

        # compute mutation scores
        cfps = cfp.mutation_CFP_scores( hap_mat, col_freqs ) # cfp of b-allele might be np.nan

        # stratify by allele frequency
        col_freqs, cfps = strat_on_freq( col_freqs, cfps )

        # save
        m_cfps.extend( cfps )
        m_freqs.extend( col_freqs )

    print "(%.1f,%.2f,%i) " % (f,s,t) + "density over %i points" % len(m_cfps)
    
    return np.array( m_freqs ), np.array( m_cfps )

###############################################################################
def plot_2d_density( x, y, p_num, time ):
    
    ''' Plot a hex-bin density showing allele frequency (X-axis) versus CFP score (Y-axis) '''

    # make plot
    plt.hexbin( x, y, marginals=False, gridsize=nbins, bins='log', cmap=plt.cm.YlOrRd, extent=(min_x, max_x, 0, max_y) )

    # axis labels
    ax = plt.gca()
    ax.set_title( r"$\mathbf{\tau=%i}$" % time )
    if( p_num in [1 ,5 ,9 ,13] ): ax.set_ylabel(r"$\mathbf{CFP}$")
    if( p_num in [13,14,15,16] ): ax.set_xlabel(r"$\mathbf{Freq.}$")

    # pretify spines
    ax.spines['top'].set_visible( False )
    ax.spines['right'].set_visible( False )
    ax.spines['bottom'].set_visible( False )
    ax.spines['left'].set_visible( False )

    # ax.spines['bottom'].set_linewidth(0.5)
    # ax.spines['bottom'].set_color('#262626')
    # ax.spines['left'].set_linewidth(0.5)
    # ax.spines['left'].set_color('#262626')

    # compute x-ticks & their labels
    if( p.fold_freq ):
        xticks = np.arange(0.1,0.51,0.1)
    else:
        xticks = np.arange(0.2,1.01,0.2)
    ax.set_xticks( xticks )
    ax.set_xticklabels( ["$%g$" % tick for tick in xticks] )

    # tick only bottom & left
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    # pretify ticks
    ax.tick_params(axis='x', colors='#262626')
    ax.tick_params(axis='y', colors='#262626')
    ax.xaxis.label.set_color('#262626')
    ax.yaxis.label.set_color('#262626')

    cb = plt.colorbar()
    if( p_num in [4,8,12,16] ): cb.set_label('$\mathbf{Log_{10}(count)}$')

###############################################################################
def strat_on_freq( col_freqs, cfps ):
    ''' if freq_strat == True, stratify freqs & CFP-scores by allele freq '''

    if( freq_strat ):
        col_freqs = col_freqs[ (col_freqs > strat_freq) & (col_freqs < 1.0) ]
        cfps = cfps[ (col_freqs > strat_freq) & (col_freqs < 1.0) ]
    
    return col_freqs, cfps

###############################################################################
def strat_on_cfp_or_freq( col_freqs, cfps, below_cfp_keep, below_freq_keep ):
    ''' stratify on CFP-score OR frequency 
        effectively removes a top-right rectangle
    '''

    if( cfp_or_freq_strat and ( not p.fold_freq ) ):
        new_col_freqs = col_freqs[ (cfps < below_cfp_keep) | (col_freqs < below_freq_keep) ]
        new_cfps = cfps[ (cfps < below_cfp_keep) | (col_freqs < below_freq_keep) ]

        return new_col_freqs, new_cfps
    else:

        return col_freqs, cfps

###############################################################################
################################## MAIN #######################################
###############################################################################
if __name__ == '__main__':
    
    make_densities()
