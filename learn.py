''' learning & classification of the meta HFS '''

import numpy as np
from sklearn import svm
from sklearn import preprocessing
from sklearn.ensemble import GradientBoostingClassifier


###############################################################################
def learn_2d_freq_cfp_classifier(freqs_n, cfps_n, freqs_s, cfps_s):

    # convert to np.array
    freqs_n, cfps_n = np.array(freqs_n), np.array(cfps_n)
    freqs_s, cfps_s = np.array(freqs_s), np.array(cfps_s)

    # remove data points where CFP=np.nan (only beneficial allele in sweep)
    cfps_s = cfps_s[~np.isnan(cfps_s)]
    freqs_s = freqs_s[~np.isnan(cfps_s)]

    # neutral class for learning ("0-label")
    A_n = np.c_[freqs_n, cfps_n]
    Y_n = np.zeros(len(A_n))

    # sweep class for learning ("1-label")
    A_s = np.c_[freqs_s, cfps_s]
    Y_s = np.ones(len(A_s))

    # combined dataset
    A = np.vstack((A_n, A_s))
    Y = np.r_[Y_n, Y_s]

    # normalize data
    # A = preprocessing.normalize( A )
    # A = preprocessing.scale( A )

    # fit classifier
    # Note: in LinearSVC use dual=False when n_samples > n_features
    #       'l2' for squared hinge, 'l1' for regular hinge
    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3)
    # clf = svm.SVC(kernel='rbf', cache_size=500)  # rbf
    # clf = svm.LinearSVC(dual=False, loss='l2')

    clf.fit(A, Y)

    return clf
