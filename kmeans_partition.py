#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import csv
from itertools import izip_longest

from builtins import zip

import pickle
import sys
from time import time

import logging
import numpy as np
import plac
import six
from gensim import utils
from multiprocessing import cpu_count

import io

from sklearn.metrics import pairwise_distances_argmin_min
from sklearn import metrics

from corpus.csv import CsvDirSentences
from corpus.jsonl import JsonlDirSentences
from tcluster.cluster.k_means_ import nearestcenters, pairwise_distances_sparse
from tcluster.metrics import jensenshannon_distance
from tcluster.metrics import nkl_transform, nkl_metric, purity_score

from sklearn.utils import as_float_array

logger = logging.getLogger(__name__)


def split_on_space(text):
    return text.split(' ')


def iter_sentences(sentences):
    for tid, raw, sentence in sentences:
        unicode_sentence = []
        for token in sentence:
            if isinstance(token, six.binary_type):
                token = token.decode('utf-8')
            unicode_sentence.append(token)
        yield ' '.join([token for token in unicode_sentence])


def iter_sentences1(sentences):
    for tid, raw, sentence in sentences:
        unicode_sentence = []
        for token in sentence:
            if isinstance(token, six.binary_type):
                token = token.decode('utf-8')
            unicode_sentence.append(token)
        yield tid, raw, ' '.join([token for token in unicode_sentence])


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    qrels=("Qrels location.", "option", "qrels", str),
    n_workers=("Number of workers", "option", "n", int),
    nr_clusters=("Number of clusters", "option", "t", int),
    batch_size=("Batch size", "option", "c", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int),
    fformat=("By default (ff=jsonl), JSONL format is used."
             "Otherwise (ff='csv'), CSV format is used.", "option", "ff", str),
    no_lemmas=("Disable Lemmatization.", "flag", "nl", bool),
    no_minibatch=("Use ordinary k-means algorithm (in batch mode).", "flag", "nm", bool),
    cosine=("Use cosine distances in place of euclidean distances.", "flag", "cos", bool),
    jsd=("Use Jensen-Shannon divergence in place of euclidean distances.", "flag", "jsd", bool),
    nkl=("Use Negative Kullback-Liebler metric in place of euclidean distances.", "flag", "nkl", bool),
    a=("JM smoothing lambda for KLD metric.", "option", "a", float),
    verbose=("Print progress reports inside k-means algorithm.", "flag", "v", bool)
)
def main(in_dir, out_loc, qrels=None, n_workers=cpu_count()-1, nr_clusters=10, batch_size=1000,
         job_size=1, max_docs=None, fformat='jsonl', no_lemmas=False, no_minibatch=False,
         cosine=False, jsd=False, nkl=False, a=.7, verbose=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    lemmatize = not no_lemmas
    minibatch = not no_minibatch
    # Set training parameters.
    num_clusters = nr_clusters
    batchsize = batch_size

    ff = fformat.lower()
    if (ff == 'jsonl'):
        sentences = utils.ClippedCorpus(JsonlDirSentences(in_dir, n_workers, job_size, lemmatize=lemmatize),
                                        max_docs=max_docs)
    elif (ff == 'csv'):
        sentences = utils.ClippedCorpus(CsvDirSentences(in_dir, n_workers, job_size, lemmatize=lemmatize),
                                        max_docs=max_docs)
    else:
        print('Unsupported corpus format specified.')
        sys.exit(1)

    if nkl:
        logger.info("Using Negative Kullback-Liebler metric")
        metric = 'nkl'
        metric_kwargs = {'a': a}
    elif jsd:
        logger.info('Using Jensen-Shannon divergence')
        metric = 'jsd'
    elif cosine:
        logger.info('Using cosine distances')
        metric = 'cosine'
    else:
        logger.info('Using euclidean distances')
        metric = "euclidean"
        metric_kwargs = {'squared': True}

    t0 = time()
    logger.info("Kmeans Partitioning")
    logger.info('CountVectorizer')
    with open(out_loc + '_count_vect.pk', 'rb') as cv:
        count_vect = pickle.load(cv)

    logger.info('TfidfTransformer')
    with open(out_loc + '_tf_transformer.pk', 'rb') as tf:
        tf_transformer = pickle.load(tf)

    terms = count_vect.get_feature_names()
    centers = np.load(out_loc + '_centers.npy')
    # centers = np.loadtxt(out_loc + '_centers.txt')

    X_train_counts = count_vect.fit_transform(iter_sentences(sentences))
    logger.info('TfidfTransformer')
    X = tf_transformer.fit_transform(X_train_counts)

    logger.info('Per-cluster inertia')
    n_samples = X.shape[0]

    # Breakup nearest neighbor distance computation into batches to prevent
    # memory blowup in the case of a large number of samples and clusters.
    # TODO: Once PR #7383 is merged use check_inputs=False in metric_kwargs.
    if metric == 'euclidean':
        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=centers, metric='euclidean', metric_kwargs={'squared': True})
    elif metric in ['jsd', 'jensenshannon']:
        D = pairwise_distances_sparse(
            X=X, Y=centers, metric=jensenshannon_distance)
        labels = D.argmin(axis=1)
        mindist = D[np.arange(n_samples), labels]
    elif metric in ['nkl', 'negative-kullback-leibler']:
        centers_mean = centers.mean(axis=0)
        nkl_kwargs = {'p_B': as_float_array(centers_mean, copy=True)}
        if metric_kwargs is not None:
            nkl_kwargs.update(metric_kwargs)
        D = pairwise_distances_sparse(
            X=X, Y=centers, metric=nkl_metric, metric_kwargs=nkl_kwargs)
        labels = D.argmin(axis=1)
        mindist = D[np.arange(n_samples), labels]
    else:
        if metric == 'cosine':
            metric_kwargs = None
        labels, mindist = pairwise_distances_argmin_min(
            X=X, Y=centers, metric=metric, metric_kwargs=metric_kwargs)
    # cython k-means code assumes int32 inputs
    labels = labels.astype(np.int32)
    inertia = np.abs(mindist).sum()  # abs is needed to select best_centers iteration

    n_centers = centers.shape[0]
    cluster_inertia_ = np.zeros(shape=(n_centers,), dtype=X.dtype)
    for i in range(n_centers):
        cluster_inertia_[i] = np.abs(mindist[np.where(labels == i)]).sum()

    if metric in ['nkl', 'negative-kullback-leibler']:
        cluster_centers_ = nkl_transform(centers, a=a)
    else:
        cluster_centers_ = centers

    order_centroids = cluster_centers_.argsort()[:, ::-1]
    terms = count_vect.get_feature_names()

    # sort by best inertia
    order_cluster = cluster_inertia_.argsort()
    with io.open(out_loc + '_topwords_sorted.txt', 'wt', encoding='utf-8') as f:
        for i in range(num_clusters):
            f.write(u'{:d}'.format(order_cluster[i]))
            for ind in order_centroids[order_cluster[i], :20]:
                f.write(u' {}'.format(terms[ind]))
            f.write(u'\n')

    qid_map = {}
    if qrels is not None:
        with io.open(qrels, 'rt') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                qid_map[int(row[2])] = int(row[0])

    km_labels_, labels_ = [], []
    sents = iter_sentences1(sentences)
    for group in grouper(batchsize * n_workers, sents):
        X = count_vect.transform([sentence[2] for sentence in group if sentence is not None])
        X = tf_transformer.transform(X)
        C = nearestcenters(X, centers, metric=metric, a=a)
        for sentence, c in zip(group, C):
            tid = sentence[0]
            km_labels_.append(c)
            labels_.append(qid_map[tid])
            print(u"{} {}".format(tid, c))


    logger.info("Kmeans Partitioning: %.0f msec" % ((time() - t0) * 1000))

    print()
    print("Cluster sizes: %s" % np.bincount(labels))
    if qrels is not None:
        km_labels_ = np.array(km_labels_)
        labels = np.array(labels_)
        print("Purity: %0.3f" % purity_score(labels, km_labels_))
        homogeneity, completeness, v_measure_score = metrics.homogeneity_completeness_v_measure(labels, km_labels_)
        print("NMI: %0.3f" % v_measure_score)
        print("ARI: %0.3f" % metrics.adjusted_rand_score(labels, km_labels_))
        print("AMI: %0.3f" % metrics.adjusted_mutual_info_score(labels, km_labels_))
        print("Homogeneity: %0.3f" % homogeneity)
        print("Completeness: %0.3f" % completeness)

    print()


def grouper(n, iterable, fillvalue=None):
    "grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return izip_longest(fillvalue=fillvalue, *args)


if __name__ == '__main__':
    plac.call(main)
