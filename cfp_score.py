
''' Module for computating clade fitness proxy (CFP) scores '''

import sys
import numpy as np
import numpy.linalg as linalg
from sklearn.cluster import MeanShift, estimate_bandwidth, KMeans
from sklearn import mixture
from sklearn.metrics import silhouette_score

import params as p


###############################################################################
def haplotype_CFP_scores(hap_mat, freqs, subset_col=None, present=None, norm=2):
    ''' Computes CFP scores for all haplotypes (rows) in given matrix
        Returns np.array of scores (possibly containing np.nan)
        'hap_mat'   : haplotype matrix, numeric (0,1)
        'freqs'     : np.array of frequencies of corresponding matrix columns
        'subset_col': (optional) column based on which to subset results
        'present'   : (optional) boolean, indicates if column '1's should be kept '0''s
    '''

    # initialize CFP scores vector
    scores = np.zeros(len(hap_mat))

    if p.cfp_exclude_fixed:
        # exclude fixed alleles, converting 1.0 to 0.0
        freqs_mod = np.copy(freqs)
        freqs_mod[freqs_mod == 1.0] = 0.0
    else:
        freqs_mod = freqs

    # de normalize, REMOVE??
    freqs_mod = freqs_mod * len(hap_mat)

    # compute CFP-score for each haplotype
    for (i, hap) in enumerate(hap_mat):

        scores[i] = np.nan  # exclude by defualt

        if subset_col is None:
            # no subset, include
            scores[i] = linalg.norm(np.multiply(hap, freqs_mod), norm)
        else:
            # subset, decide inclusion
            if present and hap_mat[i, subset_col] == 1.0:
                # include carrier i
                scores[i] = linalg.norm(np.multiply(hap, freqs_mod), norm)

            elif (not present) and hap_mat[i, subset_col] == 0.0:
                # include non-carrier i
                scores[i] = linalg.norm(np.multiply(hap, freqs_mod), norm)

    return scores


###############################################################################
def haplotype_CFP_scores_ms(hap_mat, freqs, cfp_norm):
    ''' Compute un-normalized CFP scores for all haplotypes (rows) in given matrix.
        Assumes no fixed sites in input (simulated by e.g. 'ms').
    '''

    # get sample size & init CFP-scores vector
    n = len(hap_mat)
    scores = np.zeros(n)

    # No need to remove fixed sites from score (i.e., convert 1.0s to 0.0s)
    # as this is intended strictly for use with ms output (as name implies)

    # compute CFP score for each haplotype
    for (i, hap) in enumerate(hap_mat):
        # normalized freq., normx
        # scores[i] = linalg.norm( np.multiply(hap, freqs), cfp_norm )

        # un-normalized freq., norm
        scores[i] = linalg.norm(np.multiply(hap, freqs*float(n)), cfp_norm)

        # un-normalized freq., power
        # scores[i] = np.sum(np.multiply(hap, freqs*float(n))**cfp_norm)

    return scores


###############################################################################
def cluster_CFP_scores_GMM(scores):

    # prep
    scores = np.array(scores)
    X = np.reshape(np.array(scores), (-1, 1))

    # fit model
    g = mixture.GMM(n_components=2)
    g.fit(X)

    # predict labels
    labels = g.predict(X)
    high_gaussian_ind = np.argmax(g.means_[:, 0])
    low_gaussian_ind = np.argmin(g.means_[:, 0])
    pred_labels = np.array(labels == high_gaussian_ind, dtype=float)

    # class posteriors
    probs = g.predict_proba(X)
    p_carr = probs[:, high_gaussian_ind]
    p_ncarr = probs[:, low_gaussian_ind]

    return pred_labels, p_carr, p_ncarr, g.bic(X), g.aic(X), silhouette_score(X, pred_labels)


###############################################################################
def cluster_CFP_scores(scores):
    ''' Perform clustering of CFP scores, and infer carrier/non-carrier status.
        Clustering algorithm is one of 'Mean Shift' or '2 Means'.
        Returns: inferred binary labels, 0=non-carrier,1=carrier.
    '''

    algo = '2-means'
    # algo = 'mean-shift'

    # prep
    scores = np.array(scores)
    X = np.reshape(np.array(scores), (-1, 1))

    # cluster
    if algo == 'mean-shift':
        bandwidth = estimate_bandwidth(X, quantile=0.3, n_samples=None)
        est = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        est.fit(X)

    elif(algo == '2-means'):
        est = KMeans(n_clusters=2, init='k-means++', n_init=10)
        est.fit(X)

    # extract clustering info
    labels, clust_centers = est.labels_, est.cluster_centers_
    labels_unique, n_clusters = np.unique(labels), len(np.unique(labels))

    pred_labels = np.zeros(len(scores))

    if n_clusters > 2:
        # reduce to 2 clusters accumulating points from the highest CFP cluster
        # stop when >50% of data & report accumulated points as carriers
        for center in sorted(clust_centers, reverse=True):
            to_add = np.array(labels == np.where(clust_centers == center)[0][0], dtype=float)
            pred_labels += to_add
            if np.sum(pred_labels) > 0.5*len(scores):
                break

    else:
        # assert exactly 2 clusters

        if n_clusters == 1:
            print "RE DOING WITH 2 MEANS!!"
            # force second cluster with 2-means
            est2 = KMeans(n_clusters=2, init='k-means++', n_init=10)
            est2.fit(X)
            labels, clust_centers = est2.labels_, est2.cluster_centers_
            labels_unique, n_clusters = np.unique(labels), len(np.unique(labels))

        # report top cluster
        pred_labels = np.array(labels == np.argmax(clust_centers), dtype=float)

    # compute silhouette score
    sil_score = silhouette_score(X, pred_labels)

    # return predicted labels & silhouette score
    return pred_labels, sil_score


###############################################################################
def cluster_CFP_scores_report(scores, carrier_status):
    '''
        1. cluster CFP scores using the 'cluster_CFP_scores' function
        2. generate report using the 'clustering_report' function
    '''

    # call clustering function
    # predicted_status, clustering_score = cluster_CFP_scores(scores)
    predicted_status, clustering_score = cluster_CFP_scores_GMM(scores)

    # return report function
    return clustering_report(predicted_status, carrier_status)


###############################################################################
def clustering_report(predicted_status, carrier_status):
    ''' return a report including: accuracy, tpr, fpr
    '''

    # sanity check
    assert len(predicted_status) == len(carrier_status), "Error: unexpected # of cluster labels\n"

    # prediction stats
    true_pos, false_pos, true_neg, false_neg = .0, .0, .0, .0

    for i, true_state in enumerate(carrier_status):
        pred_state = predicted_status[i]

        if pred_state and true_state:
            true_pos += 1.0

        elif (not pred_state) and true_state:
            false_neg += 1.0

        elif pred_state and (not true_state):
            false_pos += 1.0

        elif (not pred_state) and (not true_state):
            true_neg += 1.0

    # summarize
    if true_pos + false_neg > 0:
        tpr = true_pos / (true_pos + false_neg)
    else:
        tpr = 1.0

    if false_pos + true_neg > 0:
        fpr = false_pos / (false_pos + true_neg)
    else:
        fpr = 0.0

    acc = (true_pos + true_neg) / len(carrier_status)

    b_acc = 0.5*(true_pos/(true_pos + false_neg) + true_neg/(false_pos+true_neg))

    return acc, b_acc, tpr, fpr


###############################################################################
def mutation_CFP_scores(hap_mat, col_freqs):
    ''' Convenience method to compute mutation CFP scores:
        1. Computes haplotype_CFP_scores().
        2. For each mutation (and its carriers), computes mut_score_from_carrier_scores().
    '''

    # initialize mutation scores
    mut_cfp_scores = np.zeros(len(col_freqs))

    # compute CFP score for individuals (rows)
    hap_cfp_scores = haplotype_CFP_scores(hap_mat, col_freqs)

    # compute mutation (column) scores
    for (i, f_i) in enumerate(col_freqs):

        carrier_scores = hap_cfp_scores[hap_mat[:, i] > 0]
        mut_cfp_scores[i] = mut_score_from_carrier_scores(carrier_scores)

    return mut_cfp_scores


###############################################################################
def mut_score_from_carrier_scores(carrier_scores):
    ''' Computes CFP score of mutation given CFP scores of its carriers '''

    if len(carrier_scores) == 0:
        # print "[cfp_score::mut_score_from_carrier_scores]::warning: 0 carriers"
        return np.nan

    # return np.median(carrier_scores)
    return carrier_scores.mean()


###############################################################################
def score_mutations_and_haplotypes(hap_mat, h_score_func, m_score_func, k):
    ''' Compute k-scores of mutations & individuals in given haplotype matrix.
        All scores at iteration k are computed using scores from iteration k-1.
    '''

    mut_scores_dict, hap_scores_dict = {}, {}

    hap_scores = np.ones(hap_mat.shape[0])

    for it in range(k):

        # compuate mutation scores
        mut_scores = []
        for j in range(hap_mat.shape[1]):
            mut_scores.append(m_score_func(hap_mat[:, j], hap_scores))

        mut_scores = np.array(mut_scores)
        mut_scores_dict[it] = mut_scores

        # compute haplotype scores
        hap_scores = []
        for i in range(hap_mat.shape[0]):
            hap_scores.append(h_score_func(hap_mat[i, :], mut_scores))

        hap_scores = np.array(hap_scores)
        hap_scores_dict[it] = hap_scores

    return mut_scores_dict, hap_scores_dict


###############################################################################
def h_score(hap_row, mut_scores, n_order=1):
    ''' Computes the score of a haplotype given hap_row (its row in the haplotype matrix)
        and mut_scores (the scores of all mutations in the haplotype matrix).
    '''

    # vector containing for each mutation in the sample
    # its score if carried by the haplotype, and 0 othewise.
    carried_mut_scores = np.multiply(hap_row, mut_scores)

    # haplotype score is the (normalized) n_order norm of carried mutation scores
    # n_order=1 normalized sum, =2 normzlized 2-norm, etc.
    hap_score = linalg.norm(carried_mut_scores, n_order)

    # hap_score = 1.0 should never happen
    # since no haplotype carries all mutations

    return hap_score


###############################################################################
def m_score(mut_col, hap_scores, n_order=1):
    ''' Computes the score of a mutation given mut_col (its column in the haplotype matrix)
        and hap_scores (the scores of all haplotypes in the matrix).

        when k=0 all haplotype scores are equal, giving the frequency.
        when k>0 haplotype scores differ, so a mutation of freq. f carried mostly by high-scoring
        haplotypes scores higher than a mutation of freq. f carried mostly by low-scoring haplotypes
    '''

    # vector containing for each haplotype in the sample its
    # score if it is a carrier of the mutation and 0 otherwise
    carrier_hap_scores = np.multiply(mut_col, hap_scores)

    # mutation score is the (normalized) norm of carrier haplotype scores
    # n_order=1 normalized sum, =2 normzlized 2-norm, etc.
    mut_score = linalg.norm(carrier_hap_scores, n_order)

    # mut_score = 1.0 can happen if unfolded matrix
    if not p.fold_freq:
        if mut_score == 1.0 or mut_score == linalg.norm(hap_scores, n_order):
            mut_score = 0.0  # TODO think about this...

    return mut_score


###############################################################################
#                               DEPRECATED
###############################################################################
def CFP_scores_freq_bin(hap_mat, col_freqs, min_f, max_f, subset_col=None, present=None):
    ''' Computes CFP scores, aggregated over carriers of mutations of requested frequencies.

        'hap_mat'    : np.array of haplotype vectors (float)
        'min_f'      : include mutations with greater frequency.
        'max_f'      : include mutations with smaller or equal frequency.
        'subset_col' : (optional) integer, index of a mutation (column) to subset scores on.
        'present'    : (optional) boolean, if true carriers of the mutation included,
                        otherwise excluded.
    '''

    agg_scores = []  # aggregate of scores for frequency category

    # sanity check
    if hap_mat.size == 0:
        return np.array([])

    # cmopute CFP score for each individual (row) in matrix
    cfp_scores = CFP_scores(hap_mat, col_freqs, subset_col, present)

    # aggregate scores of appropriate carriers
    for i, f in enumerate(col_freqs):

        if f > min_f and f <= max_f:
            # mutation in freq. bin, find carrier indices
            carriers = np.nonzero(hap_mat[:, i])[0]

            # get scores of those individuals (may contain np.nan)
            agg_scores.extend(cfp_scores[carriers])

    # return score aggregation
    return np.array(agg_scores)
