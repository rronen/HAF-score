#!/usr/bin/env python

''' plotting utility for haplotype CFP peak and trough durring a selective sweep '''

import sys
import os
import numpy as np

from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform

''' internal imports, parent dir '''
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import params as p
import cfp_score as cfp
import hap_reader_ms as hread

mpop_file = "pop.mpop"
mpop_cmd = "/home/rronen/bin/mpop -i pop.ms -o %s -N 2000 -m 0.012 -g 100" % mpop_file

ms_file = "pop.ms"
ms_cmd = "/home/rronen/bin/ms 2000 1 -t 48.0 > %s" % ms_file

###############################################################################
def go_ms():
	
	mean_ds = []
	for i in range(100):
		os.system(ms_cmd)
		mean_d = mean_dist(ms_file, "ms")
		mean_ds.append(mean_d)

	print
	print "mean pairwise dist using cmd: %s was %g" % (ms_cmd, np.mean(mean_ds))
	print


###############################################################################
def go_mpop():
	
	mean_ds = []
	for i in range(100):
		os.system(mpop_cmd)
		mean_d = mean_dist(mpop_file, "mpop")
		mean_ds.append(mean_d)

	print
	print "mean pairwise dist using cmd: %s was %g" % (mpop_cmd, np.mean(mean_ds))
	print


###############################################################################
def mean_dist(sim_file, format):

	if format == 'ms':
		hap_mat, col_freqs, positions = hread.read_from_ms_file(sim_file)
	elif format == 'mpop':
		hap_mat, col_freqs, ba_col, positions = hread.read_from_mpop_file(sim_file)
	else:
		print "\n\t'format' must be one of 'ms' or 'mpop'. Quitting...\n\n"
		sys.exit(1)

	n = len(hap_mat)
	dist_mat = pdist(hap_mat, 'hamming')
	dist_mat = squareform(dist_mat)

	mean_diff = 0
	for i in range(n):
		for j in range(i+1, n):
			mean_diff = mean_diff + dist_mat[i,j]*len(hap_mat[0])

	mean_diff = mean_diff/(n*(n-1)/2.0)
	print mean_diff
	return mean_diff

###############################################################################
if __name__ == '__main__':

    # if len(sys.argv) != 3:
    #     print "\n\tusage: %s '<ms-like-file.ms> <'ms'/'mpop'>'\n" % sys.argv[0]
    # else:
    # 	go(sys.argv[1], sys.argv[2])

   	go_ms()  # mean pairwise dist using cmd: ms 2000 1 -t 48.0 > pop.init == 25.8554
   	go_mpop()  # mean pairwise dist using cmd: mpop -i pop.ms -o pop.mpop -N 2000 -m 0.012 -g 100 == 19.369463199103

