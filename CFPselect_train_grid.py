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

from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report

''' local imports '''
import hap_reader_ms as hread
import cfp_score as cfp
import hfs_utils as hfs
import params as p

###############################################################################
norm = 1
s = 0.05
K = 20
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
    # for f,t in [(f,t) for f in p.start_f for t in [0]]:
    #     for sim in range( p.last_sim ):
    #         hap_mat_n2, col_freqs_n2, _ , _ = hread.ms_hap_mat( f, s, t, sim, "n2" )
    #         cfps_n2  = cfp.haplotype_CFP_scores( hap_mat_n2, col_freqs_n2, norm=norm )
    #         neutral_CFPs.extend( cfps_n2 )

    # mu, sigma = np.mean( neutral_CFPs ), np.std( neutral_CFPs )
    # print "mu: %g, sigma: %g (etimated from neutral data)" % (mu, sigma)

    mu, sigma = 4777.48, 2430.13 # SHORT CUT!! REMOVE LATER

    ##########################################
    ##### train & classify on CFP scores #####
    ##########################################

    for f,t in [(f,t) for f in p.start_f for t in p.times]:
    # for f,t in [(f,t) for f in [0.0] for t in [1000] ]:

        #### learn model of CFP and classify ####
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
            CFP_spect_s  = np.array( CFP_spect_s , dtype=float )
            CFP_spect_n1 = np.array( CFP_spect_n1, dtype=float )

            # accumulate data for learning
            A.append( CFP_spect_s  )
            y.append( +1 )
            A.append( CFP_spect_n1 )
            y.append( -1 )

        # data & labels
        X, y = np.array(A), np.array(y)
        
        # fit feature-wise standardization coefficients (for mean=0 & std=1)
        scaler = preprocessing.StandardScaler( with_mean=True, with_std=True ).fit( X )
                
        # standardise features (also used to transform new data points)
        X = scaler.transform( X )

        # normalize each vector lie on the unit sphere
        X = preprocessing.normalize( A )

        # train & estimate performance (power @ FPR=0.05)
        best_clf, pow_best_clf = grid_search_performance_cv( X, y )

        # report
        print best_clf
        print "%.1f\t%.2f\t%i\t%g" % (f,s,t, pow_best_clf)

###############################################################################
def power_at_5pc_FPR(estimator, X, y):
    ''' Given an estimator that implements 'predict_proba' method, data and labels,
        returns the True Positive Rate (power) at 5 percent False Positive Rate.

        IMPORTANT: assumes data in X has already been normalized/transformed properly.
    '''

    # predict class probabilities
    probs = estimator.predict_proba( X )

    # make ROC
    fpr, tpr, thresholds = roc_curve( y, probs[:,1] )

    # partition ROC to 100 FPR bins
    mean_tpr, mean_fpr = 0.0,  np.linspace( 0, 1, 100 )
    mean_tpr += interp( mean_fpr, fpr, tpr )
    mean_tpr[0], mean_tpr[-1]  = 0.0, 1.0

    # report TPR when FPR=0.05
    return mean_tpr[5]

###############################################################################
def grid_search_performance_cv( X, y, verbose=False, nested_cv=False ):
    ''' 
        If nested_cv: separate data into development set and test set.
            1. Grid search parameters on development set by cross validation.
            2. Train final model with best hyperparameters on the entire development set. 
            2. Report performance on test set.
        
        If not nested_cv:
            1. Grid search parameters on development set by cross validation.
            2. Report cross validation performance.

        Similar to: http://scikit-learn.org/stable/auto_examples/grid_search_digits.html
    '''

    # split the dataset in two equal parts
    if( nested_cv ):
        X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.35, random_state=0 )
    else:
        X_train, X_test, y_train, y_test = X, None, y, None

    # set the parameters by cross-validation
    # tuned_parameters = [ {'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4], 'C': [1, 10, 100, 1000] },
    #                      {'kernel': ['linear'],                           'C': [1, 10, 100, 1000] } ]

    tuned_parameters = [ {'kernel': ['linear'], 'C': [0.1,1,10,100,1000,10000] } ]

    # tune parameters
    clf = GridSearchCV( svm.SVC( probability=True ), tuned_parameters, cv=K, scoring=power_at_5pc_FPR, n_jobs=6 )
    clf.fit( X_train, y_train )

    if( verbose ):
        print "\nBest parameters found on development set:"
        print clf.best_estimator_
        print
        print "Grid scores on development set:\n"
        for params, mean_score, cv_scores in clf.grid_scores_:
            print "%0.6f (+/-%0.03f) for %r" % (mean_score, cv_scores.std() / 2, params)
        if( nested_cv ):
            print "\nDetailed classification report:"
            print "Model trained on full devel set & scores computed on eval set."
            y_true, y_pred = y_test, clf.predict( X_test )
            print classification_report( y_true, y_pred )
    
    if( nested_cv ):
        # report held out TPR when FPR=0.05
        return clf.best_estimator_, power_at_5pc_FPR( clf, X_test, y_test )  
    else:
        # report cross validated TPR when FPR=0.05
        return clf.best_estimator_, clf.best_score_ 

###############################################################################
def softmax_transform( x, mu, sigma ):
    ''' given a single value or np.array, transform to the range [0,1] using a
        modified hyperbolic tangent function
    '''

    return 0.5 * ( 1 + 
                    ( 1.0 - np.exp(-(x-mu)/sigma) ) / ( 1.0 + np.exp(-(x-mu)/sigma) ) 
                 )

###############################################################################
################################## MAIN #######################################
###############################################################################
if __name__ == '__main__':
    
    train_and_test_specific()

