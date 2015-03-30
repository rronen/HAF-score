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
import learn
import params as p

###############################################################################
s = 0.05
K = 25
kernel = 'linear' # rbf
learn_2d_clf = False

###############################################################################
def train_and_test_specific():
    ''' train & test HFselect specific '''
    
    process_pool = Pool( processes=min(len(p.c_grid), 5) )

    for f,t in [(f,t) for f in p.start_f for t in p.times]:
        
        clf = None

        if( learn_2d_clf ):
            ###############################
            ##### learn 2d classifier #####
            ###############################
            all_freqs_s, all_cfps_s = [], []
            all_freqs_n, all_cfps_n = [], []

            # all simulations of current (f,s,t)
            for sim in range( p.last_sim ):
            
                # neutral: haplotype matrices with frequencies and CFP scores
                hap_mat_n1, col_freqs_n1, _ , _ = hread.ms_hap_mat( f, s, t, sim, "n1" )
                cfps_n1  = cfp.mutation_CFP_scores( hap_mat_n1 , col_freqs_n1 )
                all_freqs_n.extend( col_freqs_n1 )
                all_cfps_n.extend( cfps_n1 )

                # sweep: haplotype matrices with frequencies and CFP scores
                hap_mat_s, col_freqs_s, _ , _ = hread.ms_hap_mat( f, s, t, sim, "s" )
                cfps_s = cfp.mutation_CFP_scores( hap_mat_s , col_freqs_s  )
                all_freqs_s.extend( col_freqs_s )
                all_cfps_s.extend( cfps_s )

            clf = learn.learn_2d_freq_cfp_classifier( all_freqs_n, all_cfps_n, all_freqs_s, all_cfps_s )
        
        #################################################
        ##### learn from mHFS (using 2d classifier) #####
        #################################################
        A, y = [], []

        # all simulations of current (f,s,t)
        for sim in range( p.last_sim ):

            # haplotype matrices
            hap_mat_s , col_freqs_s , mut_pos_s , bacol = hread.ms_hap_mat( f, s, t, sim, "s"  )
            hap_mat_n1, col_freqs_n1, mut_pos_n1,   _   = hread.ms_hap_mat( f, s, t, sim, "n1" )
            
            # mHFS
            # A.append( hfs.get_hfs( hap_mat_s,  col_freqs_s,  "clust-hier", clf ) )
            A.append( hfs.get_hfs( hap_mat_s,  col_freqs_s,  "flt-f-cfp", clf ) )
            y.append(  1 )
            # A.append( hfs.get_hfs( hap_mat_n1, col_freqs_n1, "clust-hier", clf ) )
            A.append( hfs.get_hfs( hap_mat_n1, col_freqs_n1, "flt-f-cfp", clf ) )
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
def train_and_test_data( A, y, c ):
    ''' train model from given data (A) and labels (y)
        use cross validation to estimate power of model at FPR=0.05
     '''

    # normalize each sample (A[i]) s.t. it has unit norm
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

