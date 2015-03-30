#!/home/rronen/bin/python
# -*- coding: utf-8 -*-
import sys, matplotlib, os
import numpy as np
import matplotlib.pyplot as plt

tmp_file = "tmp.txt"
path = "/home/rronen/Documents/selection/data/sim_soft_500_2.4e-07_s0.050_f"
start_f = [0.00, 0.10, 0.20, 0.30, 0.40, 0.50]

###############################################################################
def go():
    
    for f in start_f:
        # get dir name for starting frequency f
        dir_name = path + "%.2f" % f
        
        # get positions of all beneficial alleles in this dir
        cmd = "head -1 %s/sim/*.pop.case.samp.t0 | grep -v \'^$\' | grep -v \'^==\' | cut -f 3 --delim \' \' > %s" % (dir_name, tmp_file)
        os.system(cmd)
        positions = np.loadtxt(tmp_file)
        os.system("rm %s" % tmp_file)
        
        # plot as histogram
        make_hist(positions, f)
        
###############################################################################
def make_hist(positions, f):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(r"starting frequency $f=%.2f$ (total %g instances)" % (f, len(positions)) )
    n, bins, patches = ax.hist(positions, 50, normed=0, facecolor='g', alpha=0.65)
    top = max( max(n) + 0.1*max(n) , 25)
    ax.set_ylim([0, top])
    ax.set_xlim([0, 1])
    plt.savefig("pos_startf_%g.png" % f, dpi=200)
    plt.show()
    plt.close(fig)
    
###############################################################################
if __name__ == '__main__':	
    go()