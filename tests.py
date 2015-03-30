#!/usr/bin/env python

import sys, os
from collections import defaultdict
import numpy as np
from scipy.stats import ks_2samp

''' local imports '''
import hap_reader_ms as hread
import cfp_score as cfp
import hfs_utils as hfs
import params as p

###############################################################################
clust_str = "<clust> is one of: 'exact', 'flt-f', 'flt-f-cfp', or 'unj'"
s = p.selection

# EHH and iHS parameters
max_iHH_dist = 0.3
min_EHH_prob = 0.05

# general test parameters
tests = ["CFP-ks"] # ["iHS", "H1", "H12", "CFP-ks"]
FPR = 0.05

###############################################################################
def run_tests():
    ''' compute haplotype-based tests of selection '''

    print "#" + "f\ts\tt\t" + "\t".join( tests )

    for f,t in [(f,t) for f in p.start_f for t in p.times]:

        test_stats = {}

        # all simulations of current (f,t)
        for sim in range( p.last_sim ):

            # read popuation sample files
            hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
            hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )
            hap_mat_n2, col_freqs_n2, mut_pos_n2,   _   = hread.ms_hap_mat( f, s, t, sim, "n2" )
            
            # sanity check
            assert p.sample_size == len(hap_mat_s) == len(hap_mat_n1) == len(hap_mat_n2)
            
            if( "H1" in tests or "H12" in tests ):
                # Petrov's H12
                s_H1 , s_H12  = petrov_stats( hap_mat_s  )
                n1_H1, n1_H12 = petrov_stats( hap_mat_n1 )
                n2_H1, n2_H12 = petrov_stats( hap_mat_n2 )

                # save results
                test_stats[ "H1" , sim, "s"  ] = s_H1
                test_stats[ "H12", sim, "s"  ] = s_H12
                test_stats[ "H1" , sim, "n1" ] = n1_H1
                test_stats[ "H12", sim, "n1" ] = n1_H12
                test_stats[ "H1" , sim, "n2" ] = n2_H1
                test_stats[ "H12", sim, "n2" ] = n2_H12

            if( "iHS" in tests ):
                # iHS & save
                test_stats[ "iHS" , sim, "s"  ] = iHS( hap_mat_s , mut_pos_s , col_freqs_s , bacol )
                test_stats[ "iHS" , sim, "n1" ] = iHS( hap_mat_n1, mut_pos_n1, col_freqs_n1, None  )
                test_stats[ "iHS" , sim, "n2" ] = iHS( hap_mat_n2, mut_pos_n2, col_freqs_n2, None  )

        # power for current (f,s,t) over all stats
        pows = power( test_stats )
        print "%.1f\t%.2f\t%i\t" % (f,s,t) + "\t".join( ["%g" % x for x in pows] )

###############################################################################
def run_CFP_ks_cont_heldout():
    ''' Generate large sample of neutral CFP scores, then compare (small) 
        sweep & neutral samples to former using two sample KS test. 
        Use P-value as test statistic and compute power at 5pc FPR.
    '''

    # setup
    norm = 1
    neut_samp = []

    # build large neutral sample
    for sim in range( p.last_sim ):
        hap_mat_n2, col_freqs_n2, mut_pos_n2, _  = hread.ms_hap_mat( 0.0, s, 0, sim, "n2" )
        # neut_samp.extend( cfp.haplotype_CFP_scores( hap_mat_n2, col_freqs_n2, norm=norm ) ) # RETURN THIS!!!!!
        neut_samp.extend( col_freqs_n2[ np.where(col_freqs_n2 < 1.0 ) ] ) # REMOVE THIS !!!!!
    neut_samp = np.array( neut_samp )
    print "\nSize of neutral sample: %i haplotype CFP scores\n" % len(neut_samp)

    for f,t in [(f,t) for f in p.start_f for t in p.times]:
        test_stats = {}

        # test sweep & neutral against existing sample
        for sim in range( p.last_sim ):
            # read popuation sample filesx
            hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
            hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )

            # s_CFP_ks , _ = ks_2samp( cfp.haplotype_CFP_scores( hap_mat_s , col_freqs_s , norm=norm ), neut_samp ) # RETURN THIS!!!!!
            # n1_CFP_ks, _ = ks_2samp( cfp.haplotype_CFP_scores( hap_mat_n1, col_freqs_n1, norm=norm ), neut_samp ) # RETURN THIS!!!!!
            s_CFP_ks , _ = ks_2samp( col_freqs_s[ np.where(col_freqs_s < 1.0 ) ]  , neut_samp ) # REMOVE THIS !!!!!
            n1_CFP_ks, _ = ks_2samp( col_freqs_n1[ np.where(col_freqs_n1 < 1.0 ) ], neut_samp ) # REMOVE THIS !!!!!

            # save results
            test_stats[ "CFP-ks" , sim, "s"  ] = s_CFP_ks 
            test_stats[ "CFP-ks" , sim, "n1" ] = n1_CFP_ks

            # print "%g\t%g" % (s_CFP_ks, n1_CFP_ks)

        pows = power( test_stats )
        print "%.1f\t%.2f\t%i\t" % (f,s,t) + "\t".join( ["%g" % x for x in pows] )

###############################################################################
def SpiLR( freqs_case, freqs_cont, ss ):
    ''' Compute the S_{\pi} statistic.
        S_{\pi} is the log ratio of Tajima's theta in control vs. case populations
    '''

    contTT = tajimas_theta( freqs_cont, ss )
    caseTT = tajimas_theta( freqs_case, ss )
    
    return np.log( (contTT + 0.1) / (caseTT + 0.1) )

###############################################################################
def tajimas_theta( freqs, numSamp ):
    ''' Compute average heterozygosity from SNP frequencies 
    '''

    numSamp = float(numSamp)

    avg_heterozigosity = 0.0
    for f in freqs:
        # only non-fixed SNPs 
        if(f < 1.0): avg_heterozigosity += f * (1.0 - f)

    scaling_factor = 2.0 * ( numSamp / ( numSamp - 1.0 ) )
    return scaling_factor * avg_heterozigosity

###############################################################################
def power( test_stats ):
    ''' Compute power of tests from 'test_stats' dict, for given (f,t) '''

    pows = []

    for stat in tests:
        s_vals, n1_vals, n2_vals = [], [], []
        
        s_vals  = [ test_stats[ stat,i, "s"  ] for i in range( p.last_sim ) ]
        n1_vals = [ test_stats[ stat,i, "n1" ] for i in range( p.last_sim ) ]
        if(stat in ['iHS', 'H1', 'H12'] ):
            # more controls to determine cutoff
            n2_vals = [ test_stats[ stat,i, "n2" ] for i in range( p.last_sim ) ]

        n_vals = n1_vals + n2_vals

        # get FRP threshold
        thresh  = sorted( n_vals, reverse=True )[ int( round(FPR*len(n_vals)) ) ]

        # power
        power = sum( 1 for x in s_vals if x >= thresh ) / float( p.last_sim )
        pows.append( power )

    return pows

###############################################################################
def petrov_stats( hap_mat ):
    ''' Computes Petrov's H1 or H12 statistics on given haplotype matrix
    '''

    # haplotype counts
    counts = defaultdict( int )
    for h in hap_mat: counts[ h.tostring() ] += 1
    srt_h_freqs = sorted( counts.values(), reverse=True )

    # normalization by sample size (makes no actual difference)
    srt_h_freqs = np.array( srt_h_freqs ) / float( p.sample_size )

    # statistics
    petrov_H1  = sum( [ x**2 for x in srt_h_freqs ] )
    petrov_H12 = ( ( srt_h_freqs[0] + srt_h_freqs[1] ) ** 2 ) + sum( [ x**2 for x in srt_h_freqs[2:] ] )
    
    return petrov_H1, petrov_H12

###############################################################################
def iHS( hap_mat, mut_pos, col_freqs, bacol=None ):
    ''' Integrated Haplotype Score, iHH (as per Voight et al., Genome Biology 2006).
        Computed for max. distance 0.3 in either direction, or until EHH < 0.05. 
        Core allele is,
            if selection: the beneficial allele.
            if neutral  : allele near center (0.5) of interval
    '''
    
    # haplotype core
    core_col = find_core_neut( hap_mat, mut_pos, col_freqs )
    core_pos = mut_pos[core_col]
    # print "pos: %g" % core_pos + "\t" + "freq: %.2f" % col_freqs[core_col] # REMOVE
    
    # integration boundries
    rm_pos = core_pos + max_iHH_dist
    lm_pos = core_pos - max_iHH_dist
    
    # rows for iHS (bool mask, all)
    rows = hap_mat[:, core_col] > -1
    assert( len(rows) == len(hap_mat) )
    
    # iHS 
    iHH = .0
    iHH += iHH_left(  hap_mat, mut_pos, core_col, lm_pos, rows )
    iHH += iHH_right( hap_mat, mut_pos, core_col, rm_pos, rows )
    return iHH

    ##################
    # iHH per allele #
    ##################
    # # determine core allele
    # if(bacol is None):
    #     core_col = find_core_neut( hap_mat, mut_pos, col_freqs )
    # else:
    #     core_col = bacol

    # core_pos = mut_pos[core_col]

    # # determine integration boundries
    # rm_pos = core_pos + max_iHH_dist
    # lm_pos = core_pos - max_iHH_dist

    # # subset rows on allele at core
    # ma_hap_rows = hap_mat[:, core_col] == 1
    # Ma_hap_rows = hap_mat[:, core_col] == 0

    # iHH_ma, iHH_Ma = .0, .0

    # # iHH for minor allele (at core) haplotypes
    # iHH_ma += iHH_left(  hap_mat, mut_pos, core_col, lm_pos, ma_hap_rows )
    # iHH_ma += iHH_right( hap_mat, mut_pos, core_col, rm_pos, ma_hap_rows )
    
    # # iHH for major allele (at core) haplotypes
    # iHH_Ma += iHH_left(  hap_mat, mut_pos, core_col, lm_pos, Ma_hap_rows )
    # iHH_Ma += iHH_right( hap_mat, mut_pos, core_col, rm_pos, Ma_hap_rows )

    # return -np.log( iHH_Ma / iHH_ma )

###############################################################################
def iHH_left( hap_mat, mut_pos, core_col, lm_pos, rows ):
    ''' Integarate EHH to the left '''    
    iHH = .0
    curr_lm_col = core_col

    while( curr_lm_col >= 0 and mut_pos[curr_lm_col] >= lm_pos ):
        # print curr_lm_col
        # print hap_mat[ rows , curr_lm_col:core_col+1 ]
        curr_EHH = EHH( hap_mat[ rows , curr_lm_col:core_col+1 ] )
        # print curr_EHH
        if(curr_EHH < min_EHH_prob): break # EHH too low, stop
        iHH += curr_EHH
        curr_lm_col -= 1

    return iHH

###############################################################################
def iHH_right( hap_mat, mut_pos, core_col, rm_pos, rows ):
    ''' Integarate EHH to the right '''
    iHH = .0
    curr_rm_col = core_col

    while(curr_rm_col <= len(mut_pos)-1 and mut_pos[curr_rm_col] <= rm_pos ):
        # print curr_rm_col
        # print hap_mat[ rows , core_col:curr_rm_col+1 ]
        curr_EHH = EHH( hap_mat[ rows , core_col:curr_rm_col+1 ] )
        # print curr_EHH
        if(curr_EHH < min_EHH_prob): break # EHH too low, stop
        iHH += curr_EHH
        curr_rm_col += 1

    return iHH

###############################################################################
def EHH( hap_mat ):
    ''' Probability that any two haplotypes are identical, at the given loci '''
    
    n = len(hap_mat)

    # haplotype frequencies
    counts = defaultdict( int )
    for h in hap_mat: counts[ h.tostring() ] += 1
    hfs = np.array( counts.values() )

    # compute EHH in region
    return np.sum( hfs*(hfs-1)/2.0 ) / ( n*(n-1)/2.0 )

###############################################################################
def find_core_neut( hap_mat, mut_pos, col_freqs ):
    ''' find the column of an allele positioned near center of interval (0.5), 
        and at a frequency near 0.5.
    '''

    # index of allele positioned closest to 0.5
    nearest_center = (np.abs(mut_pos-0.5)).argmin()
    
    # return variant closest to center (beneficial allele, if selection)
    return nearest_center
    
    #######################################
    # select an allele based on frequency #
    #######################################
    tried_dict, radios = {}, 0

    while(True):
        # stop, too far from center
        if( mut_pos[nearest_center+radios] >= 0.7 or 
            mut_pos[nearest_center-radios] <= 0.3 ): break

        # stop, found it
        if( 0.45 <= col_freqs[nearest_center+radios] <= 0.55 ):
            return nearest_center+radios
        if( 0.45 <= col_freqs[nearest_center-radios] <= 0.55 ):
            return nearest_center-radios

        tried_dict[ nearest_center+radios ] = ( mut_pos[nearest_center+radios], col_freqs[nearest_center+radios] )
        tried_dict[ nearest_center-radios ] = ( mut_pos[nearest_center-radios], col_freqs[nearest_center-radios] )

        radios += 1
    
    print "tests::find_core_neut::cannot find SNP near center [0.3,0.7] at freq. near 0.5 [0.45,0.55]"
    sys.exit(1)

###############################################################################
################################## MAIN #######################################
###############################################################################
if __name__ == '__main__':
    
    run_CFP_ks_cont_heldout()
    #run_tests()
    #run_tests_samp_size()

##############################################################################
################ NEUTRALITY TESTS UNDER VARIABLE SAMPLE SIZE #################
##############################################################################
#
# The functions below used to compute the power of neutrality tests (EHH, S_pi)
# under varying input sample size. Percent of 'optimal' power is computed as the
# percent of power obtained with a 'large' sample of 200 haplotypes.
# To re-use: 
#     1) move functions above __main__, and uncomment
#     2) replace the commented function in __main__
#
##############################################################################
# def run_tests_samp_size():
    
#     ''' MOVE ME TO BOTTOM WHEN I'VE SURVED MY PURPOSE!
#         compute power of tests of selection, varying the sample size 
#     '''

#     # setup
#     burn_in = 100
#     f,s,t, = 0.0, 0.02, 1500 # 450,700,1000,1500
#     s_sizes = [p.sample_size] + range(2,81,2)
#     # s_sizes = [p.sample_size] + [2,3,4,5]
#     print "\nComputing power for s-sizes:", s_sizes, "\n"
#     print "f=%g, s=%g, t=%g" % (f,s,t), "\n"
#     tests = ["Spi"] # Spi, iHS  one test at a time
#     test_stats = {}
#     sim_n = p.last_sim

#     for sim in range( sim_n ):
        
#         # message
#         print "Working on sim %i" % sim

#         # read popuation sample files
#         hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
#         hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )
#         hap_mat_n2, col_freqs_n2, mut_pos_n2,   _   = hread.ms_hap_mat( f, s, t, sim, "n2" )
        
#         # sanity check
#         assert p.sample_size == len(hap_mat_s) == len(hap_mat_n1) == len(hap_mat_n2)

#         # try desired sample sizes
#         for ss in s_sizes:

#             for b in range(burn_in):

#                 # down sample
#                 hap_mat_s_ds , col_freqs_s_ds , mut_pos_s_ds , bcol_ds = down_sample_to(ss, hap_mat_s , col_freqs_s , mut_pos_s , bacol)
#                 hap_mat_n1_ds, col_freqs_n1_ds, mut_pos_n1_ds,   _     = down_sample_to(ss, hap_mat_n1, col_freqs_n1, mut_pos_n1, None )
#                 hap_mat_n2_ds, col_freqs_n2_ds, mut_pos_n2_ds,   _     = down_sample_to(ss, hap_mat_n2, col_freqs_n2, mut_pos_n2, None )

#                 # compute test
#                 # test_stats[ tests[0] , sim, b, "s" , ss ] = iHS( hap_mat_s_ds , mut_pos_s_ds , col_freqs_s_ds , bcol_ds )
#                 # test_stats[ tests[0] , sim, b, "n1", ss ] = iHS( hap_mat_n1_ds, mut_pos_n1_ds, col_freqs_n1_ds, None  )
#                 # test_stats[ tests[0] , sim, b, "n2", ss ] = iHS( hap_mat_n2_ds, mut_pos_n2_ds, col_freqs_n2_ds, None  )
#                 test_stats[ tests[0] , sim, b, "s" , ss ] = SpiLR( col_freqs_s_ds , col_freqs_n1_ds, ss )
#                 test_stats[ tests[0] , sim, b, "n1", ss ] = SpiLR( col_freqs_n1_ds, col_freqs_n2_ds, ss )

#     # compute power across all sample sizes
#     pows = power_samp_size( test_stats, s_sizes, tests[0], sim_n, burn_in )

#     # write output
#     print "\n#s\tt\tsamp-size\tpower(%s)\tpc-optimal" % tests[0]
#     for ss in s_sizes:
#         print "%g\t%i\t%i\t%g\t%g" % (s,t,ss, pows[ss], 100.0*(pows[ss]/pows[p.sample_size]) )

# ###############################################################################
# def down_sample_to( size, hap_mat, col_freqs, mut_pos, bacol=None ):
#     ''' MOVE ME TO BOTTOM WHEN I'VE SURVED MY PURPOSE!
#         Down-samples the given haplotype matrix down to 'size' rows.
#         Returns the downsampled matrix & recomputed frequencies.
#         all-0 columns (post-downsampling) are also removed.
#     '''

#     # (1) copy haplotype matrix
#     new_hap_mat = np.copy(hap_mat)
 
#     # (2) shuffle rows
#     np.random.shuffle( new_hap_mat )

#     # (3) keep first 'size' rows
#     new_hap_mat = new_hap_mat[ 0:size, : ]

#     # recompute allele frequencies
#     new_col_freqs = np.sum( new_hap_mat, axis=0 ) / float( len(new_hap_mat) )

#     # remove all-0 columns, if any
#     new_hap_mat, new_col_freqs, new_ba_col, new_mut_pos = hread.remove_zero_cols( new_hap_mat, new_col_freqs, bacol, mut_pos )
    
#     return new_hap_mat, new_col_freqs, new_mut_pos, new_ba_col

# ###############################################################################
# def power_samp_size( test_stats, s_sizes, stat, last_sim, burn_in ):
    
#     ''' MOVE ME TO BOTTOM WHEN I'VE SURVED MY PURPOSE!
#         Compute power of test from 'test_stats' dict, for all sample size
#     '''
    
#     # Q: control distribution determined with full sample (p.sample_size)? Don't think so.

#     pows = {}

#     for ss in s_sizes:

#         s_vals, n1_vals, n2_vals = [], [], []

#         for sim in range(last_sim):
#             for b in range(burn_in):
#                 s_vals.append(  test_stats[ stat,sim, b, "s" , ss ] ) # case
#                 n1_vals.append( test_stats[ stat,sim, b, "n1", ss ] ) # control 1
                
#                 # single population test, can use more controls
#                 if(stat == "iHS"): 
#                     n2_vals.append( test_stats[ stat,sim, b, "n2", ss ] ) # control 2
        
#         if(stat == "iHS"): 
#             n_vals = n1_vals + n2_vals
#         else:
#             n_vals = n1_vals

#         # get FRP threshold
#         thresh  = sorted( n_vals, reverse=True )[ int( round(FPR*len(n_vals)) ) ]

#         # power
#         pows[ss] = sum( 1 for x in s_vals if x >= thresh ) / float( last_sim*burn_in )
        
#     return pows

