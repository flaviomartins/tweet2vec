#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
from time import time

import io
import six
import logging
import plac

from multiprocessing import cpu_count
from gensim import utils
from corpus.jsonl import JsonlDirSentences
from corpus.csv import CsvDirSentences

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

    t0 = time()
    print("Kmeans Split")

    cass = {}
    with io.open(out_loc, 'rt') as f:
        for line in f:
            tid, C = line.split()
            cass[int(tid)] = C

    fds = []
    for C in range(num_clusters):
        try:
            os.makedirs('splits/' + str(C))
        except:
            pass
        fds.append(io.open('splits/' + str(C) + '/' + str(C) + '.jsonl', 'wt', encoding='utf-8'))

    for tid, raw, sentence in iter_sentences(sentences):
        if tid in cass:
            fds[C].write(raw)

    for fd in fds:
        fd.close()

    print("Kmeans Split: %.0f msec" % ((time() - t0) * 1000))


if __name__ == '__main__':
    plac.call(main)
