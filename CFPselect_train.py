#!/usr/bin/env python

import sys
import numpy as np

#from sklearn import linear_model # LogisticRegression
from sklearn import svm
from multiprocessing import Pool
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.cross_validation import StratifiedKFold
from sklearn import preprocessing

''' local imports '''
import hap_reader_ms as hread
import cfp_score as cfp
import hfs_utils as hfs
import params as p

###############################################################################
norm = 1
s = 0.05
K = 10
kernel = 'rbf' # 'linear' 

# TODO: try
# 1) RBF kernel
# 2) different fixed range transform (softmax with 0.5 or 1.5 SD)
# 3) diff normalization

# bins for CFP spectra
# min_cfp, max_cfp, increment = 0, 15000, 1500
# bins = np.arange( min_cfp, max_cfp+0.0001, increment )
bins = np.arange(0, 1.001, 0.1)

###############################################################################
def train_and_test_specific():
    ''' train & test CFPselect specific '''

    ###################################################
    ##### estimate mu & sigma from (neutral) data #####
    ###################################################
    mu, sigma, neutral_CFPs = None, None, []

    # some neutral data
    for f,t in [(f,t) for f in p.start_f for t in [0]]:
        for sim in range( p.last_sim ):
            hap_mat_n2, col_freqs_n2, _ , _ = hread.ms_hap_mat( f, s, t, sim, "n2" )
            cfps_n2  = cfp.haplotype_CFP_scores( hap_mat_n2, col_freqs_n2, norm=norm )
            neutral_CFPs.extend( cfps_n2 )

    mu, sigma = np.mean( neutral_CFPs ), np.std( neutral_CFPs )
    print "mu: %g, sigma: %g (etimated from neutral data)" % (mu, sigma)

    ##########################################
    ##### train & classify on CFP scores #####
    ##########################################
    process_pool = Pool( processes=min(len(p.c_grid), 5) )

    for f,t in [(f,t) for f in p.start_f for t in p.times]:

        ########################################
        #### estimate mu & sigma from sweep ####
        ########################################
        # sweep_CFPs, mu, sigma = [], None, None
        # for sim in range( p.last_sim ):
        #     hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
        #     cfps_s  = cfp.haplotype_CFP_scores( hap_mat_s , col_freqs_s , norm=norm )
        #     sweep_CFPs.extend( cfps_s )

        # mu, sigma = np.mean( sweep_CFPs ), np.std( sweep_CFPs )
        # print "f=%g, t=%i --> mu: %g, sigma: %g" % (f, t, mu, sigma)

        #########################################
        #### learn model of CFP and classify ####
        #########################################
        A, y = [], []
        for sim in range( p.last_sim ):

            # haplotype matrices
            hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
            hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )
            
            # CFP scores
            cfps_s  = cfp.haplotype_CFP_scores( hap_mat_s , col_freqs_s , norm=norm )
            cfps_n1 = cfp.haplotype_CFP_scores( hap_mat_n1, col_freqs_n1, norm=norm )
            
            # CFP spectra
            CFP_spect_s,  _ = np.histogram( softmax_transform( cfps_s,  mu, sigma ), bins )
            CFP_spect_n1, _ = np.histogram( softmax_transform( cfps_n1, mu, sigma ), bins )
            # CFP_spect_s,  _ = np.histogram( np.clip( cfps_s,  min_cfp, max_cfp ), bins ) # use for fixed binning
            # CFP_spect_n1, _ = np.histogram( np.clip( cfps_n1, min_cfp, max_cfp ), bins ) # use for fixed binning
            CFP_spect_s  = np.array( CFP_spect_s , dtype=float )
            CFP_spect_n1 = np.array( CFP_spect_n1, dtype=float )

            # accumulate data for learning
            A.append( CFP_spect_s  )
            y.append( +1 )
            A.append( CFP_spect_n1 )
            y.append( -1 )

        # data for learning
        A, y = np.array(A), np.array(y)
        # pow5fpr = train_and_test_data(A,y,1.0)
        # print "\n" + "%.1f\t%.2f\t%i\t%g" % (f,s,t, pow5fpr)
        
        # spawn processes, try multiple error-constants (c)
        results,pow_best = [], -1.0
        for c in p.c_grid:
            # train specific model, estimate power
            result = process_pool.apply_async( train_and_test_data, (A,y,c) )
            results.append( (c,result) )

        # await processes
        for i, (c, result) in enumerate(results):
            pow5fpr = result.get()
            sys.stdout.write( "(c=%g, p=%g) " % (c, pow5fpr) )

            # save best
            if( pow5fpr > pow_best ): 
                pow_best = pow5fpr

            # clean up
            results[i] = None
        
        print "\n" + "%.1f\t%.2f\t%i\t%g" % (f,s,t, pow_best)
    
    process_pool.terminate()

###############################################################################
def softmax_transform( x, mu, sigma ):
    ''' given a single value or np.array, transform to the range [0,1] using a
        modified hyperbolic tangent function
    '''

    return 0.5 * ( 1 + 
                    ( 1.0 - np.exp(-(x-mu)/sigma) ) / ( 1.0 + np.exp(-(x-mu)/sigma) ) 
                 )

###############################################################################
def train_and_test_data( A, y, c ):
    ''' train model from given data (A) and labels (y)
        use cross validation to estimate power of model at FPR=0.05
     '''

    # normalize each sample A[i] s.t. it has unit norm
    A_norm = preprocessing.normalize( A )

    # classifier
    clf = svm.SVC( kernel=kernel, probability=True, C=c, cache_size=500 ) 

    # prep
    mean_tpr = 0.0
    mean_fpr = np.linspace( 0, 1, 100 )
    cv = StratifiedKFold( y, indices=False, n_folds=K ) # c.v. partition
    
    # mean ROC
    for i, (train, test) in enumerate(cv):

        # train
        clf.fit( A_norm[train], y[train] )

        # classify, class probabilities
        probs = clf.predict_proba( A_norm[test] )
        
        # ROC for current c.v. partition 
        fpr, tpr, thresholds = roc_curve( y[test], probs[:,1] )
        mean_tpr += interp( mean_fpr, fpr, tpr )
        mean_tpr[0] = 0.0
        
    # finalize ROC
    mean_tpr /= len(cv)
    mean_tpr[-1] = 1.0

    # model trained on complete data
    clf.fit( A_norm, y )

    # w, b = clf.coef_[0,:], clf.intercept_[0]

    # power at 0.05 FPR
    return mean_tpr[5]

###############################################################################
################################## MAIN #######################################
###############################################################################
if __name__ == '__main__':
    
    train_and_test_specific()

