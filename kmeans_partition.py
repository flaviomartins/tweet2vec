#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
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

from corpus.csv import CsvDirSentences
from corpus.jsonl import JsonlDirSentences
from tcluster.cluster.kmeans import nearestcentres

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
        yield tid, raw, ' '.join([token for token in unicode_sentence])


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    n_workers=("Number of workers", "option", "n", int),
    nr_clusters=("Number of clusters", "option", "t", int),
    nr_iter=("Number of iterations", "option", "i", int),
    batch_size=("Batch size", "option", "c", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int),
    fformat=("By default (ff=jsonl), JSONL format is used."
             "Otherwise (ff='csv'), CSV format is used.", "option", "ff", str),
    no_lemmas=("Disable Lemmatization.", "flag", "nl", bool),
    no_minibatch=("Use ordinary k-means algorithm (in batch mode).", "flag", "nm", bool),
    max_features=("Maximum number of features (dimensions) to extract from text.", "option", "D", int),
    binary_tf=("Make tf term in tf-idf binary.", "flag", "b", bool),
    sublinear_tf=("Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).", "flag", "l", bool),
    no_idf=("Disable Inverse Document Frequency feature weighting.", "flag", "ni", bool),
    cosine=("Use cosine distances in place of euclidean distances.", "flag", "cos", bool),
    jsd=("Use Jensen-Shannon divergence in place of euclidean distances.", "flag", "jsd", bool),
    kld=("Use Kulkarni's Negative Kullback-Liebler metric in place of euclidean distances.", "flag", "kld", bool),
    verbose=("Print progress reports inside k-means algorithm.", "flag", "v", bool)
)
def main(in_dir, out_loc, n_workers=cpu_count()-1, nr_clusters=10, batch_size=1000, nr_iter=100,
         job_size=1, max_docs=None, fformat='jsonl', no_lemmas=False, max_features=10000, no_minibatch=False,
         binary_tf=False, sublinear_tf=False, no_idf=False, cosine=False, jsd=False, kld=False, verbose=False):
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

    if kld:
        logger.info("Using Kulkarni's Negative Kullback-Liebler metric")
        metric = 'kld'
    elif jsd:
        logger.info('Using Jensen-Shannon divergence')
        metric = 'jsd'
    elif cosine:
        logger.info('Using cosine distances')
        metric = 'cosine'
    else:
        logger.info('Using euclidean distances')
        metric = "euclidean"

    t0 = time()
    logger.info("Kmeans Partitioning")
    logger.info('CountVectorizer')
    with open(out_loc + '_count_vect.pk', 'rb') as cv:
        count_vect = pickle.load(cv)

    logger.info('TfidfTransformer')
    with open(out_loc + '_tf_transformer.pk', 'rb') as tf:
        tf_transformer = pickle.load(tf)

    terms = count_vect.get_feature_names()
    centres = np.load(out_loc + '_centres.npy')
    # centres = np.loadtxt(out_loc + '_centres.txt')
    centres_mean = centres.mean(axis=0)

    sents = iter_sentences(sentences)
    for group in grouper(job_size * n_workers, sents):
        X = count_vect.transform([sentence[2] for sentence in group])
        X = tf_transformer.transform(X)
        C = nearestcentres(X, centres, metric=metric, precomputed_centres_mean=centres_mean)
        for sentence, c in zip(group, C):
            tid = sentence[0]
            print(u"{} {}".format(tid, c))

    logger.info("Kmeans Partitioning: %.0f msec" % ((time() - t0) * 1000))


def grouper(n, iterable):
    """grouper(3, 'ABCDEFG') --> ABC DEF G"""
    args = [iter(iterable)] * n
    return zip(*args)


if __name__ == '__main__':
    plac.call(main)
