#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division

import logging
import plac

from multiprocessing import cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans
from gensim import utils
from corpus.jsonl import JsonlDirSentences

logger = logging.getLogger(__name__)


def split_on_space(text):
    return text.split(' ')


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    n_workers=("Number of workers", "option", "n", int),
    nr_clusters=("Number of clusters", "option", "t", int),
    nr_iter=("Number of iterations", "option", "i", int),
    batch_size=("Batch size", "option", "c", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int),
    max_features=("Maximum number of features (dimensions) to extract from text.", "option", "-max-features", int),
    no_minibatch=("Use ordinary k-means algorithm (in batch mode).", "flag", "-no_minibatch", bool),
    no_idf=("Disable Inverse Document Frequency feature weighting.", "flag", "-no_idf", bool),
    verbose=("Print progress reports inside k-means algorithm.", "flag", "verbose", bool)
)
def main(in_dir, out_loc, n_workers=cpu_count()-1, nr_clusters=10, batch_size=1000, nr_iter=100,
         job_size=1, max_docs=None, max_features=10000, no_minibatch=False, no_idf=False, verbose=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    minibatch = not no_minibatch
    use_idf = not no_idf
    # Set training parameters.
    num_clusters = nr_clusters
    batchsize = batch_size
    iterations = nr_iter
    sentences = utils.ClippedCorpus(JsonlDirSentences(in_dir, n_workers, job_size, lemmatize=False),
                                    max_docs=max_docs)

    logger.info('KMeans')
    vectorizer = TfidfVectorizer(input='content', encoding='utf-8',
                                 decode_error='strict', strip_accents=None, lowercase=False,
                                 preprocessor=None, tokenizer=split_on_space, analyzer='word',
                                 stop_words=None, token_pattern=None,
                                 max_df=0.5, min_df=5,
                                 max_features=max_features, vocabulary=None, binary=True,  # binary=True -> tf=1 cap
                                 norm='l2', use_idf=use_idf, smooth_idf=True,
                                 sublinear_tf=False)  # sublinear_tf=True -> tf = 1 + log(tf)

    X = vectorizer.fit_transform(' '.join(sentence) for sentence in sentences)

    if minibatch:
        km = MiniBatchKMeans(n_clusters=num_clusters, init='k-means++', n_init=1,
                             init_size=batchsize, batch_size=batchsize, verbose=verbose)
    else:
        km = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=iterations, n_init=1,
                    verbose=verbose)

    km.fit(X)

    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]

    terms = vectorizer.get_feature_names()
    for i in range(num_clusters):
        print("%d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % repr(terms[ind]), end='')
        print()


if __name__ == '__main__':
    plac.call(main)
