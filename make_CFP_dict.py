#!/usr/bin/env python

import sys, os
from collections import defaultdict
import numpy as np

''' local imports '''
import hap_reader_ms as hread
import cfp_score as cfp
import params as p

###############################################################################
################################ SETTINGS #####################################
###############################################################################
s = 0.05
outfile = "/home/rronen/Desktop/CFP_scores_s%g.pck" % s
norm = 1

# data dictionary pickle
# keys: 
# [CFP scores, column_freqs, ba_col_num/None, ba_col/None]
#  OR
# ["metadata"]
CFP_dict = {} 

###############################################################################
def make_pck_dict():
    ''' Read population samples, compute CFP scores & Pickle '''

    for f,t in [(f,t) for f in p.start_f for t in p.times]:
        
        print "Working on f=%g, t=%g" % (f,t)

        # all simulations of current (f,t)
        for sim in range( p.last_sim ):

            # read popuation sample files
            hap_mat_s , col_freqs_s , mut_pos_s , bacol_num = hread.ms_hap_mat( f, s, t, sim, "s"  )
            hap_mat_n1, col_freqs_n1, mut_pos_n1,     _     = hread.ms_hap_mat( f, s, t, sim, "n1" )
            hap_mat_n2, col_freqs_n2, mut_pos_n2,     _     = hread.ms_hap_mat( f, s, t, sim, "n2" )

            # compute CFP scores
            s_CFPs  = cfp.haplotype_CFP_scores( hap_mat_s , col_freqs_s , norm=norm )
            n1_CFPs = cfp.haplotype_CFP_scores( hap_mat_n1, col_freqs_n1, norm=norm )
            n2_CFPs = cfp.haplotype_CFP_scores( hap_mat_n2, col_freqs_n2, norm=norm )

            # save
            CFP_dict[ f, s, t, sim, "s" ]  = [ s_CFPs , col_freqs_s , bacol_num, hap_mat_s[:,bacol_num] ]
            CFP_dict[ f, s, t, sim, "n1" ] = [ n1_CFPs, col_freqs_n1,    None  ,         None           ]
            CFP_dict[ f, s, t, sim, "n2" ] = [ n2_CFPs, col_freqs_n2,    None  ,         None           ]

    # metadata
    CFP_dict["metadata"] = "s=%g, CFP-norm=%g" % (s, norm)

    # pickle everything
    with open(outfile, mode='wb') as out_fh: pck.dump(CFP_dict, out_fh)

###############################################################################
################################## MAIN #######################################
###############################################################################
if __name__ == '__main__':
    
    make_pck_dict()
