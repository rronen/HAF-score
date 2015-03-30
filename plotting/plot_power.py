#!/usr/bin/env python

''' plotting power of selection statistics '''

import sys
import os
import math
import matplotlib
import brewer2mpl
import numpy as np
import matplotlib.pyplot as plt

from matplotlib import rcParams
from matplotlib.ticker import NullLocator
from os.path import basename

''' local imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p

###############################################################################
# ############################### Settings ####################################
###############################################################################
pdir = '/home/rronen/Dropbox/UCSD/workspace/SoftSweep/power/'
plot_fname_pref = '../plots/power/power'
power_dict = {}
fix_time_dict = {}
# stats_2_plot = [ 'H12', 'SFselect', 'iHS' ] # H1
# stats_2_plot = [ 'H12', 'SFselect', 'iHS', 'HFselect-s(-10x+2.0)', 'HFselect-s(-5.83x+1.75)', 'HFselect-s(-7x+1.75)', 'HFselect-s(rctn1.75-0.2)' ]
# stats_2_plot = [ 'H12', 'iHS', 'SFselect-s', 'SFselect-s(noF)', 'HFselect-s(-5.83x+1.75)', 'HFselect-s(-4x+2.2)FOLD', 'HFselect-s(-5.1x+4)UF']
# stats_2_plot = [ 'H12', 'iHS', 'HFselect-s(-5.1x+4)UF', 'HFselect-s-UFrule', 'HFselect-s(-5.83x+1.75)', 'SFselect-s(noF)' ]
# stats_2_plot = [ 'H12', 'iHS', 'SFselect-s(noF)', 'HFselect-s(-5.83x+1.75)','HFselect-s(-5.83x+1.75)-Hclst', 'HFselect-s-Bst-Hclst-F']
# stats_2_plot = [ 'H12', 'iHS', 'SFselect', 'CFP-ks', 'SFS-ks'] # 'SFselect-s(noF)'
stats_2_plot = [ 'H12', 'iHS', 'SFselect-s(noF)', 'CFPselect-s', 'CFPselect-s-grid' ] # 'CFPselect-s-f'

f_2_subf = { 0.0:221, 0.1:222, 0.3:223, 0.5:224 }
s = 0.05

colors1 = [(.15,.15,.15)] + brewer2mpl.get_map('Set1', 'qualitative', 8).mpl_colors
colors2 = [(.15,.15,.15)] + brewer2mpl.get_map('Set2', 'qualitative', 7).mpl_colors

# stat:(color,style,width)
stat_2_graphics = { 'iHS':[0,'-',1.5], 
                    'H12':[1,'-',1.5],
                    'SFselect-s(noF)':[2,'-',1.5],  # no fixed differences
                    'CFPselect-s':[3,'-',1.5],      # mu & sigma from neutral (better)
                    # 'CFPselect-s-v': [4,'-',1.5], # mu & sigma from sweep (not great)
                    # 'CFPselect-s-f': [4,'-',1.5],   # fixed bins 0,1000,2000,...,14000 
                    # 'CFPselect-s-RBF': [4,'-',1.5],   # mu & sigma from neutral
                    'CFPselect-s-grid': [4,'-',1.5],   # mu & sigma from neutral
                    'CFP-ks':[3,'-',1.5],
                    'SFS-ks':[4,'-',1.5],
                    'TD':[4,'-',1.5],
                    'H':[5,'-',1.5],

                    'SFselect-s':[4,'-',1.5], # with fixed diff
                    'SFselect-s(F)':[4,'-',1.5], # folded
                    'HFselect-s(-5.83x+1.75)':[5,'-',1.5],
                    'HFselect-s-Bst-Hclst-F': [3,'-',1.5],
                    'HFselect-s(-5.83x+1.75)-Hclst':[7,'-',1.5],
                    'HFselect-s(-4x+2.2)FOLD':[7,'-',1.5],
                    'HFselect-s(-5.1x+4)UF':[3,'-',1.5],
                    'HFselect-s-UFrule':[4,'-',1.5],

                    'H1':[3,'-',1.5],
                    'SFselect':[2,'-',1.5],
                    'HFselect-s(-5.83x+1.75)MED':[5,'-',1.5],
                    'HFselect-s(-5.83x+1.75)MEAN':[3,'-',1.5],
                    'HFselect-s(-10x+2.0)':[4,'-',1.5],
                    'HFselect-s(-7x+1.75)':[6,'-',1.5],
                    'HFselect-s(rctn1.75-0.2)':[7,'-',1.5],
                }

###############################################################################
###############################################################################
###############################################################################
def plot_power( logX ):

    # other tests
    read_power_from_file( pdir + "iHS.txt" )
    read_power_from_file( pdir + "petrov_H1.txt" )
    read_power_from_file( pdir + "petrov_H12.txt" )
    read_power_from_file( pdir + "SFselect.txt" )
    read_power_from_file( pdir + "SFselect_s_noFIX.txt" )
    read_power_from_file( pdir + "SFselect_s_w_FIX.txt" )
    read_power_from_file( pdir + "SFselect_s_FOLDED.txt" )

    # classics
    read_power_from_file( pdir + "classics_Tajimas_D.txt" )
    read_power_from_file( pdir + "classics_Fay_Wu_H.txt" )
    read_power_from_file( pdir + "classics_XPCLR.txt" )
    read_power_from_file( pdir + "classics_XPEHH.txt" )

    # CFPselect specific
    read_power_from_file( pdir + "CFP_select_s_10bins_softmax_htan_neutral_mu_sigma.txt" )
    read_power_from_file( pdir + "CFP_select_s_10bins_softmax_htan_variable_mu_sigma.txt" )
    # read_power_from_file( pdir + "CFP_select_s_14bins_fixed1000_norm1.txt" )
    read_power_from_file( pdir + "CFP_select_s_10bins_fixed1500_norm1.txt" )
    read_power_from_file( pdir + "CFP_select_s_10bins_softmax_htan_neutral_mu_sigma_RBFkernel_normalized.txt" )
    # read_power_from_file( pdir + "CFP_select_s_10bins_softmax_htan_neutral_mu_sigma_GRID_norm_scale_cv5.txt" )
    read_power_from_file( pdir + "CFP_select_s_10bins_softmax_htan_neutral_mu_sigma_GRID_norm_scale_cv10.txt" )

    # CFP Kolmogorov Smirnov test
    read_power_from_file( pdir + "kstest_SFS_140k_neut.txt" )
    read_power_from_file( pdir + "kstest_CFP_100k_neut.txt" )

    # HFselect-s, current best
    read_power_from_file( pdir + "HFselect_s_diag_-5.83x+1.75.txt" )

    read_power_from_file( pdir + "HFselect_s_learnGBoost_hclust.txt" )
    read_power_from_file( pdir + "HFselect_s_diag_-5.83x+1.75_hclst.txt" )

    read_power_from_file( pdir + "HFselect_s_FOLDED_diag_-4.0x+2.2.txt" )
    read_power_from_file( pdir + "HFselect_s_UNFOLD_diag_-5.1x+4.0.txt" )
    read_power_from_file( pdir + "HFselect_s_UNFOLD_diag_-5.1x+4.0_rule.txt" )

    # read HFselect-s, trials
    # read_power_from_file( pdir + "HFselect_s_10bins_MEAN_vs_MEDIAN/HFselect_s_diag_-5.83x+1.75_MEDIAN.txt" )
    # read_power_from_file( pdir + "HFselect_s_10bins_MEAN_vs_MEDIAN/HFselect_s_diag_-5.83x+1.75_MEAN.txt" )
    # read_power_from_file( pdir + "HFselect_s_20bins/HFselect_s_diag_-10x+2.0.txt" )
    # read_power_from_file( pdir + "HFselect_s_20bins/HFselect_s_diag_-5.83x+1.75.txt" )
    # read_power_from_file( pdir + "HFselect_s_20bins/HFselect_s_diag_-7x+1.75.txt" )
    # read_power_from_file( pdir + "HFselect_s_20bins/HFselect_s_rect_1.75_0.25.txt" )

    # read HFselect, current best
    # TODO

    # read HFselect, trials
    # TODO

    # read fixation times
    read_fixation_times( pdir + "fixation_t/times_f", ".txt")

    plot_quad( logX )

###############################################################################
def plot_quad( logX ):

    mid = ''
    if logX: mid = '_logX'
    plot_fname = plot_fname_pref + mid + '.pdf'
    print "Plotting power (saving to %s)" % plot_fname
    done_so_far = 0

    # pretify: latex font, outward ticks
    rcParams['xtick.direction'] = 'out'
    rcParams['ytick.direction'] = 'out'
    matplotlib.rc('text', usetex = True)

    # figure
    fig = plt.figure(figsize=(9.5, 6))
    fig.subplots_adjust(hspace=0.2, left=0.07, right=0.96, top=0.97, bottom=0.15)
    labels, lines = [], []

    for f,subf in sorted( f_2_subf.iteritems() ):

        ax = fig.add_subplot( subf )
        done_so_far += 1
        
        # ticks & tick-marks for top axis (units = ln(2Ns)/s)
        # ax_x2 = ax.twiny()
        # unit =  ( math.log(2*N*s) + math.log(math.log(2.0)) ) / s
        # lnNs_ticks_full = np.arange(unit , max(p.times)-150, unit )
        # interval = int( (len(lnNs_ticks_full) / 10.0) + 0.5 )
        # start = 0
        # while( lnNs_ticks_full[start] < 200 ): start += 1

        # axis limits
        ax.set_ylim(0, 1.0) 
        if( logX ):
            # ax_x2.set_xlim(45, max(p.times))
            # ax_x2.semilogx()
            # lnNs_ticks = list(lnNs_ticks_full[start:6]) + list(lnNs_ticks_full[7:-1:int(interval*2)])
            ax.set_xlim( 45, 3500 ) # 23.5 to include t=25
            ax.semilogx()
            ax.set_xticks([50,100,250,500,1000,3000])
            ax.xaxis.set_minor_locator( NullLocator() )
            ax.set_xticklabels(['$50$','$100$','$250$', '$500$', '$1000$', '$3000$'])
            ax.text( 1500,0.91,r'$f=%.1f$' % f, fontsize=15 )
        else:
            # ax_x2.set_xlim(-50, max(p.times))
            # lnNs_ticks = list( lnNs_ticks_full[start::interval] )
            ax.set_xlim( -75, 3100 )
            ax.text( 2450,0.91,r'$f=%.1f$' % f, fontsize=15 ) # 2950

        # v2l = {}
        # for i,tick in enumerate( lnNs_ticks_full ): v2l[tick] = r'$%i$' % ( i+1 )
        # ax_x2.xaxis.set_minor_locator( NullLocator() )
        # ax_x2.set_xticks(lnNs_ticks)
        # ax_x2.set_xticklabels([v2l[tick] for tick in lnNs_ticks], fontsize=12.5)
        for tick in ax.xaxis.get_major_ticks(): tick.label.set_fontsize(12.5)
        for tick in ax.yaxis.get_major_ticks(): tick.label.set_fontsize(12.5)

        # plot lines
        for stat in stats_2_plot:
            # make x,y
            x, y = [], [] 
            for t in p.times:
                if( ( f,s,t,stat ) in power_dict ):
                   x.append(t)
                   y.append( power_dict[f,s,t,stat] )

            line, = ax.plot( x, y, linestyle=get_style(stat), linewidth=stat_2_graphics[stat][2], # 1.75 for slides 
                           color=get_color(stat), marker=get_marker(stat), label=get_label(stat) )

        
            if( done_so_far == 4 ):
                lines.append( line )
                labels.append( get_label(stat) )

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
        ax.xaxis.set_ticks_position('none') # remove x ticks
        # ax.yaxis.set_ticks_position('none') # remove y ticks

        # axis labels
        if(subf == 221 or subf == 223):
            ax.set_ylabel(r'$\mathbf{Power}$', fontsize=13)
            
        if(subf == 223 or subf == 224):
            ax.set_xlabel(r'$\mathbf{Time\,\,(generations)}$', fontsize=13)
            
        # if(subf == 221 or subf == 222):
            # ax_x2.set_xlabel(r'$\mathbf{Time \,\, (ln(2Ns)/s \,\, generations)}$', fontsize=13)

        # show times to fixation (horizontal 1d scatter, vertical dashed line, countur)
        locs = ax.yaxis.get_majorticklocs()
        bp = ax.boxplot(fix_time_dict[f], vert=False, positions=[1.01], sym='+', widths=0.08)
        fix_boxplot_colors( bp )
        ax.set_ylim(0, 1.06)
        ax.yaxis.set_ticks(locs)
        ax.axvline(x=np.median(fix_time_dict[f]), ymin=0, ymax=0.91, linewidth=1.65, color='0.5', linestyle='dotted')

    # legend
    legend = fig.legend(lines, labels, 'lower center', bbox_to_anchor=[0.5, -0.005], prop={'size':12}, handlelength=3.5, 
                        ncol=len(stats_2_plot), # ncol=4,
                        columnspacing=1, labelspacing=0.1, handletextpad=0.25, fancybox=True, shadow=False)
    rect = legend.get_frame()
    rect.set_facecolor( np.array([float(247)/float(255)]*3) )
    rect.set_linewidth(0.0) # remove edge
    texts = legend.texts
    for t in texts: t.set_color('#262626')

    # save file
    plt.savefig( plot_fname, dpi=550)
    plt.show()

###############################################################################
def fix_boxplot_colors( bp ):
    ''' pretify box plot '''

    # line colors
    bp_line_names = ['medians', 'fliers', 'boxes', 'whiskers', 'caps']
    for element in bp_line_names:
        matplotlib.artist.setp( bp[element], color='0.5') 
            
    # other
    matplotlib.artist.setp( bp['medians'], linestyle='-') # median line style
    matplotlib.artist.setp( bp['whiskers'], linestyle='-') # whiskers line style
    # matplotlib.artist.setp( bp['fliers'], markeredgecolor='none') # outlier line color
    matplotlib.artist.setp( bp['fliers'], markersize=3) # outlier marker size

###############################################################################
def get_label( stat ):
    # stat = stat.replace("HFselect", "HFs")
    # stat = stat.replace("SFselect", "SFs")
    stat = stat.replace("x+", ",")
    stat = stat.replace("-", "\mbox{-}")

    return "$\mathbf{%s}$" % stat

###############################################################################
def get_marker( stat ):
    return None

###############################################################################
def get_style( stat ):
    return stat_2_graphics[stat][1]

###############################################################################
def get_color( stat ):
    return colors1[ stat_2_graphics[stat][0] ]

###############################################################################
def read_power_from_file( fname ):
    ''' read power (at FPR=0.05) frmo given file into power_dict[f,s,t,stat_name] '''

    global power_dict
    PF = open( fname, 'r' )
    
    for line in PF:
        if( line.startswith('#') ):
            stat_name = line.rstrip().split()[3]
            continue
        else:
            f, s, t, power = [ float(x) for x in line.rstrip().split() ]
            power_dict[f,s,t,stat_name] = power

    PF.close()

###############################################################################
def read_fixation_times( pref, suff ):
    ''' for relevant f (starting freq. or 'softness'), read the
        times to fixation into an np.array in fix_time_dict[f]
    '''
    global fix_time_dict

    for f in f_2_subf.keys():
        times = []
        fname = pref + "%.1f" % f + suff
        
        F = open( fname, 'r')
        for line in F:
            sim, t = [ float(x) for x in line.rstrip().split() ]
            times.append(t)

        fix_time_dict[f] = np.array( times )

###############################################################################
################################## Main #######################################
###############################################################################
if __name__ == '__main__':
    
    if len(sys.argv) > 1:
        plot_power( True )
    else:
        plot_power( False )

