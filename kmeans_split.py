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
from os import path
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
    in_file=("Location of input file"),
    out_dir=("Location of output directory"),
    n_workers=("Number of workers", "option", "n", int),
    nr_clusters=("Number of clusters", "option", "t", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int),
    fformat=("By default (ff=jsonl), JSONL format is used."
             "Otherwise (ff='csv'), CSV format is used.", "option", "ff", str),
    verbose=("Print progress reports inside k-means algorithm.", "flag", "v", bool)
)
def main(in_dir, in_file, out_dir, n_workers=cpu_count()-1, nr_clusters=10, job_size=1, max_docs=None,
         fformat='jsonl', verbose=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Set training parameters.
    num_clusters = nr_clusters

    ff = fformat.lower()
    if (ff == 'jsonl'):
        sentences = utils.ClippedCorpus(JsonlDirSentences(in_dir, n_workers, job_size, lemmatize=False),
                                        max_docs=max_docs)
    elif (ff == 'csv'):
        sentences = utils.ClippedCorpus(CsvDirSentences(in_dir, n_workers, job_size, lemmatize=False),
                                        max_docs=max_docs)
    else:
        print('Unsupported corpus format specified.')
        sys.exit(1)

    t0 = time()
    logger.info("Kmeans Split")

    cass = {}
    with io.open(in_file, 'rt') as f:
        for line in f:
            tid, C = line.split()
            cass[int(tid)] = C

    fds = []
    for C in range(num_clusters):
        try:
            os.makedirs(path.join(out_dir, str(C)))
        except OSError:
            pass
        fds.append(io.open(path.join(out_dir, str(C), str(C) + '.jsonl'), 'wt', encoding='utf-8'))

    for tid, raw, sentence in iter_sentences(sentences):
        if tid in cass:
            C = cass[tid]
            fds[int(C)].write(raw)

    for fd in fds:
        fd.close()

    logger.info("Kmeans Split: %.0f msec" % ((time() - t0) * 1000))


if __name__ == '__main__':
    plac.call(main)
