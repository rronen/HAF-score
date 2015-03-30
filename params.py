''' HFselect Parameters '''

# HFS & mHFS
nbins        = 10
scale_counts = True
bins         = None
bin_edges    = None

# CFP score, has effect only when not folded
cfp_exclude_fixed = True

fold_freq = False
# fold_freq = True

# HFS clustering
flt_freq = 0.15
flt_freq_h, flt_cfp_h = 0.2, 3.75
flt_freq_l, flt_cfp_l = 0.2, 2.50

# simulation
last_sim    = 500
sample_size = 200

# times points post selection
times  = [0, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
times += [i for i in range(600,4001,100)]

# selection coefficients
selection = 0.05

# starting frequency
start_f = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

# error constant for learning
c_grid = [0.1, 1.0, 10.0]
