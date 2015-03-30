''' haplotype reader utility '''
import sys
import cPickle as pck
import numpy as np
import params as p

# ms simulation file dirs, require ('%.3f' % s) and ('%.2f' % f)
p1_hs, p2_hs, p3_hs = "/home/rronen/Documents/selection/data/sim_500_2.4e-07_s", "_f", "/sim"            # hard sweep
p1_ss, p2_ss, p3_ss = "/home/rronen/Documents/selection/data/sim_soft_500_2.4e-07_s", "_f", "/sim"       # soft sweep
p1_nj, p2_nj, p3_nj = "/home/rronen/Documents/selection/data/sim_500_2.4e-07_s", "_f", "/unj_clust_md6"  # neighbor join

# cahce files
cached_p1, cached_p2, cached_p3 = "/home/rronen/Documents/selection/data/sim_soft_CACHE/f", "_s", "_t"

use_cache = False # turns out it's too big
cache     = None  # dict for current cached data

###############################################################################    
def get_file_name( sim, sim_type, t=None ):
    ''' returns file name of given simulation.
        sim_type can be: s, n1, n2, init
    '''

    # init file, no time (t)
    if( sim_type == "init"):
        return "%i.pop.case.start" % (sim)
    elif( t is None ):
        print "[get_file_name]::error, no time (t) specified."
        sys.exit(1)

    # simulation file
    if(sim_type in ["s", "s-noR"]):
        return "%i.pop.case.samp.t%i" % (sim, t)
    elif(sim_type == "n1"):
        return "%i.pop.cont.samp1.t%i" % (sim, t)

    elif(sim_type == "n2"):
        return "%i.pop.cont.samp2.t%i" % (sim, t)


###############################################################################    
def get_dir_name(f, s, sim_type):
    ''' returns directory of (f,s) simulation files '''

    if sim_type == "s-noR":
        # soft sweep WITHOUT recombination
        return p1_ss + "%.3f" % s + p2_ss + "%.2f_rho0" % f
    else:
        # soft/hard sweep WITH recombination
        if s == 0.05:
            # soft sweep directories
            return p1_ss + "%.3f" % s + p2_ss + "%.2f" % f + p3_ss
        else:
            # hard sweep directories
            return p1_hs + "%.3f" % s + p2_hs + "%.2f" % f + p3_hs


###############################################################################
def get_cache_filename( f, s, t, sim_type ):
    ''' returns name of (f,s,t,sim_type) cahce file '''

    return cached_p1 + "%.1f" % f + cached_p2 + "%.2f" % s + cached_p3 + "%i" % t + "_%s" % sim_type

###############################################################################
def ms_mut_pos( f, s, sim ):
    ''' returns np.array of mutation positions for given simulation instance.
    '''
    
    msfile = open( get_dir_name( f, s, None ) + "/" + get_file_name( sim, "init" ) )

    positions = []
    for line in msfile:
        if( line.startswith("positions") ):
            positions = line.rstrip().split()[1:]
            break
            
    return np.array( positions, dtype='float64' )


###############################################################################
def ms_neut_mat(sim, dname):
    ''' function to read ms neutral file '''

    fpath = "/home/rronen/Documents/selection/data/" + dname + "/%i.ms" % sim
    
    hap_mat, col_freqs, positions = read_from_ms_file( fpath )

    # note: no need to exclude all-0/1 columns, as this is purely ms output

    if p.fold_freq:
        # fold haplotype matrix
        for j in range(len(col_freqs)):
            if col_freqs[j] > 0.5: 
                # fold column & frequency
                hap_mat[:,j] = np.array(np.logical_not(hap_mat[:,j]))
                col_freqs[j] = 1.0 - col_freqs[j]

    return hap_mat, col_freqs, positions


###############################################################################
def ms_hap_mat(f, s, t, sim, sim_type):
    ''' reads simulated population file in ms format.
        returns haplotype matrix, variant frequencies, variant positions, column of adaptive allele.
    '''

    if use_cache:
        # read matrix & other info from cache
        hap_mat, col_freqs, ba_col, positions = read_from_cache(f, s, t, sim_type, sim)
    else:
        # get file path, read matrix & other info from ms file
        fpath = get_dir_name(f, s, sim_type) + "/" + get_file_name(sim, sim_type, t)
        hap_mat, col_freqs, ba_col, positions = read_from_mpop_file(fpath)

    if p.fold_freq:

        # fold haplotype matrix
        for j in range(len(col_freqs)):
            if(col_freqs[j] > 0.5): 
                # fold column & frequency
                hap_mat[:,j] = np.array(np.logical_not(hap_mat[:, j]))
                col_freqs[j] = 1.0 - col_freqs[j]

        # again, exclude all-0 columns (previously all-1 columns) & re-compute necessary info
        hap_mat, col_freqs, ba_col, positions = remove_zero_cols(hap_mat, col_freqs, ba_col, positions)

    # sanity check
    zero_f = np.where( col_freqs == 0 )[0]  # indices of 0-columns
    if ba_col is None:
        # neutral, no 0-columns
        assert len(zero_f) == 0
    else:
        # sweep, only ba_col allowed as all-0 column
        assert (len(zero_f) == 0) or (len(zero_f) == 1 and zero_f[0] == ba_col)


    return hap_mat, col_freqs, positions, ba_col

###############################################################################
def remove_zero_cols( hap_mat, col_freqs, ba_col, positions ):
    ''' remove all-zero columns from given haplotype matrix '''

    cols_2_keep, new_cols_count, new_ba_col  = [], 0, None
    
    for j in range( len( col_freqs ) ):
        
        if(j == ba_col): # beneficial allele, keep either way
            cols_2_keep.append( j )
            new_ba_col = new_cols_count
            new_cols_count += 1
        else:
            if( col_freqs[j] > 0 ): # keep column
                cols_2_keep.append( j )
                new_cols_count += 1
    
    new_hap_mat   = hap_mat[ :, cols_2_keep ]
    new_positions = np.array( [ positions[c] for c in cols_2_keep ], 'float64')

    # sanity check
    assert len( new_positions ) == new_hap_mat.shape[1]
    
    # new column frequencies
    new_col_freqs = np.sum( new_hap_mat, axis=0 ) / float( len(new_hap_mat) )

    return new_hap_mat, new_col_freqs, new_ba_col, new_positions

###############################################################################
def read_from_mpop_file(f):
    ''' reads haplotype matrix from given mpop output file.
        returns haplotype matrix, column frequencies, column of beneficial allele (None if neutral).
    '''

    # setup
    ba_pos, ba_col = None, None  # neutral -> ba_col=None
    haps = []

    # open ms file
    mpopfile = open(f)

    # read file    
    for (i,line) in enumerate(mpopfile):
        if i == 0:
            # beneficial allele position
            ba_pos = line.rstrip().split()[2]
            
            # sanity check
            assert ba_pos == '0' or 0.3 <= float(ba_pos) <= 0.7
            
        elif line.startswith("positions"):
            # get mutation positions
            positions = line.rstrip().split()[1:]
            
            # column of beneficial allele
            if ba_pos != '0':
                ba_col = positions.index(ba_pos)
            
        elif i > 5:
            # haplotype string to float vector
            hap = np.array(list(line.rstrip()), dtype='float64')
            haps.append(hap)

    mpopfile.close()  

    # float matrix
    hap_mat = np.vstack(haps)

    # sanity check
    assert len( positions ) == hap_mat.shape[1]

    # column frequencies
    col_freqs = np.sum(hap_mat, axis=0) / float(len(hap_mat))


    # exclude all-0 columns & re-compute necessary info
    return remove_zero_cols(hap_mat, col_freqs, ba_col, positions)


###############################################################################
def read_from_ms_file(f):
    ''' reads haplotype matrix from given ms output file
        returns haplotype matrix, column frequencies, and derived allele positions
    '''

    # setup
    ba_col = None  # neutral -> ba_col=None (not used for now)
    haps = []
    
    # open ms file
    msfile = open(f)
    
    # read file    
    for (i,line) in enumerate(msfile):
            
        if( line.startswith("positions") ):
            # get mutation positions
            positions = line.rstrip().split()[1:]
            
        elif(i > 5):
            # haplotype string to float vector
            hap = np.array( list( line.rstrip() ), dtype='float64' )  # CHECK 'float64' 'int8' (443M)
            haps.append( hap )

    msfile.close()  
    
    # float matrix
    hap_mat = np.vstack( haps )
    
    # sanity check
    assert len( positions ) == hap_mat.shape[1]
    
    # column frequencies
    col_freqs = np.sum( hap_mat, axis=0 ) / float( len( hap_mat ) )

    # Note:
    # no need to exclude all-0 columns as this is purely ms output

    return hap_mat, col_freqs, positions

###############################################################################
def read_and_cache_hap_matrices( f, s, t, sim_type ):
    ''' read all simulations under given parameters, and cache as a cPickle dict '''

    print "reading & caching data for f=%.1f, s=%.2f, t=%i, pop=%s..." % ( f, s, t, sim_type )

    cache_dict = {}

    for sim in range( p.last_sim ):
        # file path
        fpath = get_dir_name( f, s, sim_type ) + "/" + get_file_name( sim, sim_type, t )

        # read haplotype matrix & other data from file
        cache_dict[f, s, t, sim_type, sim] = read_from_mpop_file( fpath )

    # save as cPickle
    cache_f = get_cache_filename( f, s, t, sim_type )
    with open( cache_f, mode='wb') as cache_fh:
        pck.dump( cache_dict, cache_fh )

###############################################################################
def read_from_cache( f, s, t, sim_type, sim ):
    ''' '''
    global cache

    if( cache is None or (not (f,s,t,sim_type,0) in cache) ):
        # load cache
        print "loading cahce"
        cache_f = get_cache_filename( f, s, t, sim_type )
        with open( cache_f, mode='rb' ) as cache_fh:
            cache = pck.load( cache_fh )

    return cache[f, s, t, sim_type, sim]

###############################################################################
################################## MAIN #######################################
###############################################################################
if __name__ == '__main__':
    
    print "\n" + "Starting operation cahce" + "\n"

    s = 0.05
    for f in p.start_f:
        for t in p.times:
            read_and_cache_hap_matrices( f, s, t, "s" )
            read_and_cache_hap_matrices( f, s, t, "n1" )
            read_and_cache_hap_matrices( f, s, t, "n2" )
