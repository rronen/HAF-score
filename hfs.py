''' main haplotype frequency spectrum '''

import sys
import numpy as np

''' local imports '''
import hap_reader_ms as hread
import cfp_score as cfp
import hfs_utils as hfs
import params as p

import plots.plot_cfp_dist as pltcfp
import plots.plot_hfs as plthfs
import plots.plot_mut_scatter as mut_scatt

###############################################################################
################################# Settings ####################################
###############################################################################

times = [0, 25,  50, 100, 200, 400, 600, 800, 1000] # for 2D freq,CFP plots

clust_str = "<clust> is one of: 'exact', 'flt-f', 'flt-f-cfp', or 'unj'"

# lists for HFS vectors
HFS_s, HFS_n1, HFS_n2 = [],[],[] 

# arrays for HFS (per bin) mean & std-dev vectors
asize = p.nbins if(p.scale_counts) else p.sample_size
HFS_s_mean,  HFS_s_std  = np.empty(asize), np.empty(asize)
HFS_n1_mean, HFS_n1_std = np.empty(asize), np.empty(asize)
HFS_n2_mean, HFS_n2_std = np.empty(asize), np.empty(asize)

# lists for CFP scores
scores_s, scores_n, scores_s_yb, scores_s_nb = [], [], [], []

# frequency categories for CFP score dist
freq_bounds = [ 0., .15, .3, .6, .9, 1. ]

# lists for CFP scores vectors per, frequency category
s_freq_cat_vlists  = [ [] for startf in freq_bounds[:-1] ]
n_freq_cat_vlists  = [ [] for startf in freq_bounds[:-1] ]
sy_freq_cat_vlists = [ [] for startf in freq_bounds[:-1] ]
sn_freq_cat_vlists = [ [] for startf in freq_bounds[:-1] ]

# skip sim instance if selected allele fixed, asserts strictly ongoing sweeps
skip_fixed, skip_count = False, 0

# plot CFP scores
plot_cfp_scores = False
plot_mut_scatter = False
plot_mean_hfs = False

###############################################################################
################################### Go ########################################
###############################################################################
def go( f, s, t, group ):
    ''' Generate mean haplotype frequency spectra for data in given (s,t).
        Clustering methods are 'exact' (e.g. none), 'flt-f', 'flt-f-cfp', or 'unj'.
    '''

    # global parameters
    global HFS_s, HFS_n1, HFS_n2, skip_count
    global HFS_s_mean, HFS_s_std, HFS_n1_mean, HFS_n1_std, HFS_n2_mean, HFS_n2_std
    
    for sim in range( p.last_sim ):

        for t in times:
            
            # read popuation sample files
            hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
            hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )
            hap_mat_n2, col_freqs_n2, mut_pos_n2,   _   = hread.ms_hap_mat( f, s, t, sim, "n2" )
            
            # sanity check
            assert len(hap_mat_s) == len(hap_mat_n1) == len(hap_mat_n2) == p.sample_size
            
            # polymorphic positions of init-pop (SUPERVISED)
            init_mut_pos = None
            if(plot_mut_scatter): init_mut_pos = hread.ms_mut_pos( f, s, sim )

            # compute haplotype frequency spectrum
            HFS_s.append(  hfs.get_hfs( hap_mat_s,  col_freqs_s,  group) )
            HFS_n1.append( hfs.get_hfs( hap_mat_n1, col_freqs_n1, group) )
            HFS_n2.append( hfs.get_hfs( hap_mat_n2, col_freqs_n2, group) )
                
            if(plot_mut_scatter and group == 'flt-f-cfp'):
                # mutation scatter
                mut_scatt.compute_scatter( f, s, t, hap_mat_n1, col_freqs_n1, mut_pos_n1, init_mut_pos )
                mut_scatt.compute_scatter( f, s, t, hap_mat_s,  col_freqs_s,  mut_pos_s,  init_mut_pos, bacol )
            
            if(plot_cfp_scores and group == 'flt-f-cfp'):
                # accumulate CFP scores for plots
                if( skip_fixed and subset_haps_on_mut(hap_mat_s, bacol, present=False).size == 0 ):
                    skip_count += 1
                else:
                    accumulate_cfp_scores( hap_mat_s, col_freqs_s, hap_mat_n1, col_freqs_n1, hap_mat_n2, col_freqs_n2, bacol )
                    accumulate_cfp_scores_fbins( hap_mat_s, col_freqs_s, hap_mat_n1, col_freqs_n1, hap_mat_n2, col_freqs_n2, bacol )

        # plot mutation scatter
        if(plot_mut_scatter): mut_scatt.plot_scatter( sim )

    # reports
    print "\nskipped %i simulations with fixed beneficial allele.\n" % skip_count
    
    # plot CFP scores
    if(group == 'flt-f-cfp' and plot_cfp_scores): cfp_score_plots()
    
    # stack HFS arrays
    HFS_s  = np.vstack(HFS_s)
    HFS_n1 = np.vstack(HFS_n1)
    HFS_n2 = np.vstack(HFS_n2)
    
    # plot mean spectra
    if( plot_mean_hfs ): plthfs.plot_mean_hfs(HFS_s, HFS_n1, HFS_n2, p.bins, p.bin_edges, p.scale_counts, group, f, s, t)

###############################################################################
def cfp_score_plots():
    ''' '''
    
    # report score counts & means
    print "Score counts:"
    print "\t" + "neutral     : %i (avg=%.2f)" % ( len( scores_n )   , len( scores_n )   /float(p.last_sim) )
    print "\t" + "sweep       : %i (avg=%.2f)" % ( len( scores_s )   , len( scores_s )   /float(p.last_sim) )
    print "\t" + "sweep BA yes: %i (avg=%.2f)" % ( len( scores_s_yb ), len( scores_s_yb )/float(p.last_sim) )
    print "\t" + "sweep BA  no: %i (avg=%.2f)" % ( len( scores_s_nb ), len( scores_s_nb )/float(p.last_sim) )
    print "\n"
    
    # make histograms
    #pltcfp.plot_score_hists( scores_s_yb, scores_s_nb, scores_n, s, t )
    
    # make violins
    datasets, positions, ticks = [scores_s_yb, scores_s_nb, scores_n], [1,2,3], [1,2,3]
    tick_labs = [r"$\mathbf{Sweep[1]}$", r"$\mathbf{Sweep[0]}$", r"$\mathbf{Neutral}$"]
    
    pltcfp.violin_plot( datasets, positions, ticks, tick_labs, s, t, boxp=True )
    
    # make violins per frequency bins
    pos, datasets, positions, ticks, tick_labs = 2, [], [], [], []
    for i, from_f in enumerate(freq_bounds[:-1]):
        to_f = freq_bounds[i+1]
        
        # sweep carriers
        datasets.append( sy_freq_cat_vlists[i] )
        positions.append(pos)
        pos += 1
        
        # mid-category tick
        ticks.append(pos)
        tick_labs.append( r"$[%g, %g]$" % (from_f, to_f) )
        
        # sweep non carriers
        datasets.append( sn_freq_cat_vlists[i] )
        positions.append(pos)
        pos += 1
        
        # neutral
        datasets.append(  n_freq_cat_vlists[i] )
        positions.append(pos)
        pos += 2
    
    pltcfp.violin_plot(datasets, positions, ticks, tick_labs, s, t, boxp=True)

###############################################################################
def accumulate_cfp_scores( hap_mat_s, col_freqs_s, hap_mat_n1, col_freqs_n1, hap_mat_n2, col_freqs_n2, bacol ):
    ''' Accumulates CFP scores. '''
    
    global scores_s, scores_n, scores_s_yb, scores_s_nb
    
    # compute CFP scores for general categories
    s_scores  = cfp.CFP_scores( hap_mat_s , col_freqs_s )
    n1_scores = cfp.CFP_scores( hap_mat_n1, col_freqs_n1 )
    n2_scores = cfp.CFP_scores( hap_mat_n2, col_freqs_n2 )
    
    # compute CFP scores for selected allele categories & remove np.nan
    s_yb_scores = cfp.CFP_scores( hap_mat_s, col_freqs_s, subset_col=bacol, present=True  ) # includes np.nan
    s_nb_scores = cfp.CFP_scores( hap_mat_s, col_freqs_s, subset_col=bacol, present=False ) # includes np.nan
    s_yb_scores = s_yb_scores[ np.logical_not( np.isnan(s_yb_scores) ) ] 
    s_nb_scores = s_nb_scores[ np.logical_not( np.isnan(s_nb_scores) ) ]
    
    # accumulate scores
    scores_s.extend( s_scores  )
    scores_n.extend( n1_scores )
    scores_n.extend( n2_scores )

    scores_s_yb.extend( s_yb_scores )
    scores_s_nb.extend( s_nb_scores )
    
###############################################################################
def accumulate_cfp_scores_fbins( hap_mat_s, col_freqs_s, hap_mat_n1, col_freqs_n1, hap_mat_n2, col_freqs_n2, bacol ):
    ''' Accumulates CFP scores for frequency bins. '''
    
    global n_freq_cat_vlists, s_freq_cat_vlists
    global sy_freq_cat_vlists, sn_freq_cat_vlists
    
    # iterate frequency categories
    for i, from_f in enumerate(freq_bounds[:-1]):
        to_f = freq_bounds[i+1]

        # compute CFP scores for general categories
        s_scores  = cfp.CFP_scores_freq_bin( hap_mat_s , col_freqs_s , from_f, to_f )
        n1_scores = cfp.CFP_scores_freq_bin( hap_mat_n1, col_freqs_n1, from_f, to_f )
        n2_scores = cfp.CFP_scores_freq_bin( hap_mat_n2, col_freqs_n2, from_f, to_f )
        
        # compute CFP scores for selected allele categories & remove np.nan
        s_yb_scores = cfp.CFP_scores_freq_bin( hap_mat_s, col_freqs_s, from_f, to_f, subset_col=bacol, present=True  ) # includes np.nan
        s_nb_scores = cfp.CFP_scores_freq_bin( hap_mat_s, col_freqs_s, from_f, to_f, subset_col=bacol, present=False ) # includes np.nan
        s_yb_scores = s_yb_scores[ np.logical_not( np.isnan(s_yb_scores) ) ]
        s_nb_scores = s_nb_scores[ np.logical_not( np.isnan(s_nb_scores) ) ]
                
        # accumulate scores
        n_freq_cat_vlists[i].extend( n1_scores )
        n_freq_cat_vlists[i].extend( n2_scores )
        s_freq_cat_vlists[i].extend( s_scores  )
        
        sy_freq_cat_vlists[i].extend( s_yb_scores )
        sn_freq_cat_vlists[i].extend( s_nb_scores )

###############################################################################
def subset_haps_on_mut( haps, col, present=True ):
    ''' Subset a haplotypes from a matrix, on the state of a given mutation (column). 
        If present=True, returns haplotypes with the mutation. Otherwise, returns 
        haplotypes without. Returns np.array.
    '''
    
    if( present ): 
        return haps[ haps[:, col] == 1 ]
    else:
        return haps[ haps[:, col] == 0 ]

###############################################################################
if __name__ == '__main__':
    
    # get command line
    if len(sys.argv) == 5:
       f, s, t, group = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    else: 
        print "\n" + "    usage: %s <f> <s> <t> <clust>" % sys.argv[0]
        print "\n" + "where %s" % clust_str
        print ""
        sys.exit(1)
    
    # validate clustering method
    if ( not group in ["exact", "flt-f", "flt-f-cfp", "unj"] ):
        print "\n" + "Please make sure " + clust_str + "\n"
        sys.exit(1)
    
    go( f, s,t, group )
