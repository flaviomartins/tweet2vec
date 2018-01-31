#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import pickle
import sys
from time import time

import logging
import numpy as np
import plac
import six
from gensim import utils
from multiprocessing import cpu_count
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import io
from corpus.csv import CsvDirSentences
from corpus.jsonl import JsonlDirSentences
from tcluster.cluster import KMeans, SampleKMeans, MiniBatchKMeans

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


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    n_workers=("Number of workers", "option", "n", int),
    nr_clusters=("Number of clusters", "option", "t", int),
    nr_iter=("Number of iterations", "option", "i", int),
    nr_init=("Number of times the k-means algorithm will be run with different centroid seeds.", "option", "r", int),
    batch_size=("Batch size", "option", "c", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int),
    fformat=("By default (ff=jsonl), JSONL format is used. "
             "Otherwise (ff='csv'), CSV format is used.", "option", "ff", str),
    no_lemmas=("Disable Lemmatization.", "flag", "nl", bool),
    no_minibatch=("Use ordinary k-means algorithm (in batch mode).", "flag", "nm", bool),
    max_features=("Maximum number of features (dimensions) to extract from text.", "option", "D", int),
    delta=("Improvement threshold for stopping.", "option", "d", float),
    binary_tf=("Make tf term in tf-idf binary.", "flag", "b", bool),
    sublinear_tf=("Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).", "flag", "l", bool),
    no_idf=("Disable Inverse Document Frequency feature weighting.", "flag", "ni", bool),
    cosine=("Use cosine distances in place of euclidean distances.", "flag", "cos", bool),
    jsd=("Use Jensen-Shannon divergence in place of euclidean distances.", "flag", "jsd", bool),
    nkl=("Use Negative Kullback-Liebler metric in place of euclidean distances.", "flag", "nkl", bool),
    a=("JM smoothing lambda for KLD metric.", "option", "a", float),
    verbose=("Print progress reports inside k-means algorithm.", "flag", "v", bool)
)
def main(in_dir, out_loc, n_workers=cpu_count()-1, nr_clusters=10, batch_size=1000, nr_iter=100, nr_init=1,
         job_size=1, max_docs=None, fformat='jsonl', no_lemmas=False, max_features=10000, delta=.001, no_minibatch=False,
         binary_tf=False, sublinear_tf=False, no_idf=False, cosine=False, jsd=False, nkl=False, a=.7, verbose=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    lemmatize = not no_lemmas
    minibatch = not no_minibatch
    use_idf = not no_idf
    # Set training parameters.
    num_clusters = nr_clusters
    batchsize = batch_size
    iterations = nr_iter

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

    logger.info('CountVectorizer')
    count_vect = CountVectorizer(input='content', encoding='utf-8',
                                 decode_error='strict', strip_accents=None, lowercase=False,
                                 preprocessor=None, tokenizer=split_on_space, analyzer='word',
                                 stop_words=None, token_pattern=None,
                                 max_df=0.5, min_df=5,
                                 max_features=max_features, vocabulary=None, binary=binary_tf)
    X_train_counts = count_vect.fit_transform(iter_sentences(sentences))

    if nkl:
        logger.info("Using Negative Kullback-Liebler metric")
        logger.info('TfidfTransformer')
        tf_transformer = TfidfTransformer(norm='l1', use_idf=use_idf, smooth_idf=True,
                                          sublinear_tf=sublinear_tf)  # sublinear_tf -> tf = 1 + log(tf)
        X_train_tf = tf_transformer.fit_transform(X_train_counts)
        metric = 'nkl'
    elif jsd:
        logger.info('Using Jensen-Shannon divergence')
        logger.info('TfidfTransformer')
        tf_transformer = TfidfTransformer(norm='l1', use_idf=use_idf, smooth_idf=True,
                                          sublinear_tf=sublinear_tf)  # sublinear_tf -> tf = 1 + log(tf)
        X_train_tf = tf_transformer.fit_transform(X_train_counts)
        metric = 'jsd'
    elif cosine:
        logger.info('Using cosine distances')
        logger.info('TfidfTransformer')
        tf_transformer = TfidfTransformer(norm='l2', use_idf=use_idf, smooth_idf=True,
                                          sublinear_tf=sublinear_tf)  # sublinear_tf -> tf = 1 + log(tf)
        X_train_tf = tf_transformer.fit_transform(X_train_counts)
        metric = 'cosine'
    else:
        logger.info('Using euclidean distances')
        logger.info('TfidfTransformer')
        tf_transformer = TfidfTransformer(norm='l2', use_idf=use_idf, smooth_idf=True,
                                          sublinear_tf=sublinear_tf)  # sublinear_tf -> tf = 1 + log(tf)
        X_train_tf = tf_transformer.fit_transform(X_train_counts)
        metric = "euclidean"

    t0 = time()
    if no_minibatch:
        km = KMeans(n_clusters=num_clusters, init='random', max_iter=iterations, n_init=nr_init,
                    max_no_improvement=iterations / 10,
                    metric=metric, metric_kwargs={'a': a},
                    tol=delta, verbose=True)
    else:
        km = MiniBatchKMeans(n_clusters=num_clusters, init='random', max_iter=iterations, n_init=nr_init,
                             max_no_improvement=iterations / 10, compute_labels=True,
                             metric=metric, metric_kwargs={'a': a},
                             init_size=None, batch_size=batchsize, tol=delta, verbose=True)

    logger.info("Clustering sparse data with %s" % km)
    t0 = time()
    km.fit(X_train_tf)
    logger.info("Kmeans: %.0f msec" % ((time() - t0) * 1000))

    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = count_vect.get_feature_names()

    with io.open(out_loc, 'wt', encoding='utf-8') as f:
        for i in range(num_clusters):
            f.write(u'{:d}'.format(i))
            for ind in order_centroids[i, :20]:
                f.write(u' {}'.format(terms[ind]))
            f.write(u'\n')

    np.save(out_loc + '_centres.npy', km.cluster_centers_)
    np.savetxt(out_loc + '_centres.txt', km.cluster_centers_)

    with open(out_loc + '_count_vect.pk', 'wb') as cv:
        pickle.dump(count_vect, cv)

    with open(out_loc + '_tf_transformer.pk', 'wb') as tf:
        pickle.dump(tf_transformer, tf)


if __name__ == '__main__':
    plac.call(main)
