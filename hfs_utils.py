''' utility for computing the haplotype frequency spectrum (HFS) and meta-HFS '''

import sys
from collections import defaultdict
import numpy as np
from sklearn import preprocessing

import scipy.cluster.hierarchy as hier

''' internal imports '''
import cfp_score as cfp
import params as p

###############################################################################
def get_hfs( hap_mat, col_freqs, type, clf=None ):
    ''' Compute the meta-haplotype frequency spectrum using specified method '''

    if(type == 'exact'):
        return exact_hfs( hap_mat )

    elif( type == 'flt-f' ):
        return mhfs_flt_f( hap_mat, col_freqs )

    elif( type == 'flt-f-cfp' ):
        if( clf is None ):
            return mhfs_flt_f_cfp( hap_mat, col_freqs )
        else:
            return mhfs_flt_f_cfp_clf( hap_mat, col_freqs, clf )
    elif( type == 'clust-hier' ):
        return mhfs_clust_hier( hap_mat, col_freqs )

    else:
        print "[get_hfs]:error, unrecognized meta-haplotype method: %s" % type
        sys.exit(1)

###############################################################################
def exact_hfs( hap_mat ):
    ''' Computes the exact HFS, given haplotype matrix.
        Returns np.array historgram of haplotype frequencies.
    '''

    # haplotype counts
    counts = defaultdict( int )
    for h in hap_mat: counts[ h.tostring() ] += 1
    
    # haplotype frequency spectrum
    return counts_to_hist( counts )
    
###############################################################################
def mhfs_flt_f( hap_mat, col_freqs, bacol=None ):
    ''' Computes meta-haplotype frequency spectrum, given haplotype matrix.
        Clusters by filtering low frequency alleles.
        Returns histogram (np.array) of meta-haplotype frequencies.
    '''
    
    # filter low frequency alleles
    flt_hap_mat = hap_mat[:, col_freqs > p.flt_freq ]
    
    # meta-haplotype counts
    counts = defaultdict(int)
    for h in flt_hap_mat: counts[ h.tostring() ] += 1
    
    # meta-haplotype frequency spectrum
    return counts_to_hist( counts )

###############################################################################
def mhfs_flt_f_cfp( hap_mat, col_freqs ):
    ''' Computes meta-haplotype frequency spectrum, given haplotype matrix.
        Clusters by filtering low frequency mutations inriched in individuals with high Clade Fitness Proxy (CFP) scores.
        Returns histogram (np.array) of meta-haplotype frequencies.
    '''
    
    keep = np.ones( len(col_freqs) ) # 1=keep, initially keep all

    # mutation CFP scores
    mutation_cfp_scores = cfp.mutation_CFP_scores( hap_mat, col_freqs )

    # if unfolded, determine regime
    if( not p.fold_freq ):
        # mean CFP of non fixed mutations
        if( np.mean( mutation_cfp_scores[ col_freqs < 1.0 ] ) > 3.5 ):
            high = True # high mean, pre fixation
        else:
            high = False # low mean, post fixation
    
    # mark columns (mutations) for removal
    for i in range( len(col_freqs) ):
        
        f_i, cfp_i = col_freqs[i], mutation_cfp_scores[i]
        
        # filter fixed, uninformative for clustering
        if( f_i == 0.0 ): keep[i] = 0 # only b-allele (kept even if 0-column, has cfp_i=np.nan)
        # if( f_i == 1.0 ): keep[i] = 0 # only unfolded (fixed alleles kept)

        # apply filtering rule
        if( p.fold_freq ):
            # just frequency
            # if(f_i < 0.25): keep[i] = 0

            # linear constraint
            if( cfp_i < -5.8333*f_i+1.75 ): keep[i] = 0 # (0.30,0) m=-5.833=-1.75/0.3
            # if( cfp_i < -4*f_i+2.2 ): keep[i] = 0
            # if( cfp_i < -10.0*f_i + 2.00 ): keep[i] = 0 # (0.20,0) m=-10
            # if( cfp_i < -7.00*f_i + 1.75 ): keep[i] = 0 # (0.25,0) m=-7
            
            # bottom left rectangle
            # if( cfp_i < 1.75 and f_i < 0.25 ): keep[i] = 0

        else:
            # just frequency
            # if(f_i < 0.35): keep[i] = 0

            # linear constraint
            # if( cfp_i < -5.1*f_i+4 ): keep[i] = 0 # (0.0,4.0),(1.275,0.0)
                        
            # both rectangles
            # if( cfp_i > p.flt_cfp_h and f_i < p.flt_freq_h ): 
            #     keep[i] = 0
            # elif( cfp_i < p.flt_cfp_l and f_i < p.flt_freq_l ):
            #     keep[i] = 0

            # conditional rectangles
            if(high):
                if( cfp_i > 3.75 and f_i < 0.2 ): keep[i] = 0
            else:
                if( cfp_i < -5.1*f_i+4.0 ): keep[i] = 0 # (0.0,4.0),(1.275,0.0)

    # filtered meta-haplotype matrix
    flt_hap_mat = hap_mat[:, keep > 0 ]
    
    # cluster remaining columns -- REMOVE?
    # col_freqs = col_freqs[ keep > 0 ]
    # return mhfs_clust_hier( flt_hap_mat, col_freqs )
    
    # meta-haplotype counts
    counts = defaultdict(int)
    for h in flt_hap_mat: counts[ h.tostring() ] += 1
    
    # meta-haplotype frequency spectrum
    return counts_to_hist( counts )

###############################################################################
def mhfs_flt_f_cfp_clf( hap_mat, col_freqs, clf ):
    ''' Computes meta-haplotype frequency spectrum, given haplotype matrix and a 2D classifier of freq/CFP.
        Clusters by filtering mutations deemed sweep class by the classifier.
        Returns histogram (np.array) of meta-haplotype frequencies.
    '''

    # remove columns with freq=0 (cannot be classified, as they have CFP=np.npn)
    hap_mat = hap_mat[:, col_freqs > 0 ]
    col_freqs = col_freqs[ col_freqs > 0 ]

    # mutation CFP scores
    mutation_cfp_scores = cfp.mutation_CFP_scores( hap_mat, col_freqs )

    # data points for classification
    A = np.c_[ col_freqs, mutation_cfp_scores ]

    # predict
    # A = preprocessing.normalize( A )
    # A = preprocessing.scale( A )

    pred = clf.predict( A ) # 0 neutral, 1 sweep
    # pred = clf.predict_proba( A )

    # print "%i mutations" % len( pred )
    # print "%i sweep1, %i neutr0" % ( len( pred[ pred == 1 ] ), len( pred[ pred == 0 ] ) )

    # filtered meta-haplotype matrix using predicted labels
    flt_hap_mat = hap_mat[:, pred == 0 ]
    # flt_hap_mat = hap_mat[:, pred[:,1] > 0.45 ]

    # cluster reduced matrix
    col_freqs = col_freqs[ pred == 0 ]
    return mhfs_clust_hier( flt_hap_mat, col_freqs )

    # meta-haplotype counts
    counts = defaultdict(int)
    for h in flt_hap_mat: counts[ h.tostring() ] += 1

    # print "mH counts: ", sorted( counts.values(), reverse=True )
    # raw_input("Press Enter to continue...")
    # print

    # meta-haplotype frequency spectrum
    return counts_to_hist( counts )

###############################################################################
def mhfs_clust_hier( hap_mat, col_freqs ):

    # filter low frequency alleles
    # hap_mat = hap_mat[:, (col_freqs > 0.3)]

    # Z = hier.linkage( hap_mat, method='ward', metric='euclidean' )
    Z = hier.linkage( hap_mat, method='average', metric='hamming' )
    clusters = hier.fcluster( Z, 1.0 )
    counts = defaultdict(int)
    for m in clusters: counts[m] += 1
    
    # meta-haplotype frequency spectrum
    return counts_to_hist( counts )

###############################################################################
def counts_to_hist( counts_dict ):
    ''' Computes a scaled histogram from the values of given dict.
        Returns hist (np.array) scaled by bin centers.
    '''
    
    # convert counts to frequencies
    if(p.scale_counts):
        # re-scale counts to [0,1]
        freqs =  np.array( counts_dict.values() ) / float( p.sample_size )
    else:
        # raw counts
        freqs = counts_dict.values()
    
    # make bins for histogram, if not made already
    if(p.bins is None and not p.scale_counts): p.bins = np.arange(0, p.sample_size+1, 1)
    if(p.bins is None and     p.scale_counts): p.bins = make_bins()
    
    # histogram
    hist,  p.bin_edges = np.histogram(freqs , p.bins)
    hist = hist.astype( np.float64 )
    
    # histogram scaling
    if( p.scale_counts ):
        # sclaed bin centers
        centers = 0.5 * ( p.bin_edges[:-1] + p.bin_edges[1:] )
        hist *= centers**2 # >1 to scale by polynomial of bin-centers
    else:
        # actual descrete bins
        hist *= np.arange(0, p.sample_size, 1)

    return hist 

###############################################################################
def make_bins():
    ''' Makes bins for haplotype frequency spectra. 
        Returns np.array of bin boundaries, in [0,1] or [0, sample-size].
    '''
    bin_dist = 'const' # 'low-high' 'exp-inc'

    if( p.scale_counts ):
        
        if( bin_dist == 'const' ):
            # constant bin sizes
            r = np.arange( 0, 1.00001, 1.0/p.nbins )
        
        elif(bin_dist == 'low-high'):
            # low/high bin sizes
            r1 = np.arange(0.0, 0.201, 0.04) # low  frequencies, high resolution
            r2 = np.arange(0.3, 1.001, 0.20) # high frequencies, low resolution
            r = np.concatenate( (r1, r2) )
        
        elif( bin_dist == 'exp-inc'):
            # exponentially growing bin sizes
            r, f = [0.0], 0.02
            while f < 1.0:
               r.append(f)
               f *= 1.75
            r.append(1.0)
            r = np.array(r)

    else:
        # counts, no binning
        r = np.arange( 0, sample_size+1, 1 )
    
    # write bins screen
    print "\n" + "HFS bins:", r, "\n"

    return r

###############################################################################
################################# DEPRACATED ##################################
###############################################################################
def hist_hapgroup_unj_freqs(clust_file):
    ''' Computes haplotype-cluster frequency spectrum, given a file with clustering results.
        Clustering done via max-diameter subtrees in the unrooted neighbor joining tree.
        Returns histogram (np.array) of halohroup frequnecies.
    '''

    counts = {}
    f = open(clust_file)
    for i,line in enumerate(f):
        if('Number of labels' in line and i > 1): # skips 1st (tree total) 'Number of labels'
            counts[i] = int(line.split()[4])
    
    return counts_to_hist(counts)
