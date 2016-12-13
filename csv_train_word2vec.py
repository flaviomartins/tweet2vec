#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division

import gzip
import logging
from multiprocessing import cpu_count
from os import path

from concurrent.futures import ProcessPoolExecutor, as_completed
from toolz import partition_all

# fails to import scandir < 3.5
try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk
import fnmatch

import plac
import unicodecsv as csv
from gensim.models import Word2Vec
from gensim import utils
from twokenize import twokenize
from preprocessing import process_texts

logger = logging.getLogger(__name__)


class MultipleFileSentences(object):
    def __init__(self, directory, n_workers=cpu_count()-1, job_size=1):
        self.directory = directory
        self.n_workers = n_workers
        self.job_size = job_size

    def __iter__(self):
        jobs = partition_all(self.job_size, iter_jsons(self.directory))
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for j, job in enumerate(jobs):
                futures.append(executor.submit(process_job, job))
                if j % self.n_workers == 0:
                    for future in as_completed(futures):
                        try:
                            results = future.result()
                        except Exception as exc:
                            logger.error('generated an exception: %s', exc)
                        else:
                            logger.debug('job has %d sentences', len(results))
                            for result in results:
                                if result is not None:
                                    yield result
                    futures = []
            for future in as_completed(futures):
                try:
                    results = future.result()
                except Exception as exc:
                    logger.error('generated an exception: %s', exc)
                else:
                    logger.debug('job has %d sentences', len(results))
                    for result in results:
                        if result is not None:
                            yield result


def iter_jsons(directory):
    for root, dirnames, filenames in walk(directory):
        for filename in fnmatch.filter(filenames, '*.csv*'):
            yield path.join(root, filename)


def process_job(job):
    results = []
    for filepath in job:
        result = process_file(filepath)
        if result is not None:
            results += result
    return results


def process_file(filepath):
    if filepath.endswith('.gz'):
        csvfile = gzip.open(filepath, 'rb')
    else:
        csvfile = open(filepath, 'rb')

    # TODO: csv module has problems with null bytes?
    reader = csv.reader(csvfile, encoding='utf-8')
    next(reader, None)  # skip the headers
    result = []
    try:
        for row in reader:
            result.append(twokenize.tokenizeRawTweetText(row[3]))
    except csv.Error as ce:
        logger.warn('DECODE FAIL: %s %s', filepath, ce.message)
        pass
    csvfile.close()
    return process_texts(result)


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    skipgram=("By default (sg=0), CBOW is used. Otherwise (sg=1), skip-gram is employed.", "option", "sg", int),
    n_workers=("Number of workers", "option", "n", int),
    size=("Dimension of the word vectors", "option", "d", int),
    window=("Context window size", "option", "w", int),
    min_count=("Min count", "option", "m", int),
    negative=("Number of negative samples", "option", "g", int),
    nr_iter=("Number of iterations", "option", "i", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int)
)
def main(in_dir, out_loc, skipgram=0, negative=5, n_workers=cpu_count()-1, window=10, size=200, min_count=10, nr_iter=2,
         job_size=1, max_docs=None):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(
        size=size,
        sg=skipgram,
        window=window,
        min_count=min_count,
        workers=n_workers,
        sample=1e-5,
        negative=negative,
        iter=nr_iter
    )
    sentences = utils.ClippedCorpus(MultipleFileSentences(in_dir, n_workers, job_size), max_docs=max_docs)

    model.build_vocab(sentences, progress_per=10000)
    model.train(sentences)

    model.save(out_loc)


if __name__ == '__main__':
    plac.call(main)
