#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
import codecs
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import gzip
import logging
from toolz import partition_all
from os import path
# fails to import scandir < 3.5
try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk
import fnmatch

import plac
try:
    import ujson
except ImportError:
    import json as ujson
import json
import yaml
from gensim.models import Word2Vec
from gensim.utils import ClippedCorpus
from nltk.tokenize import TweetTokenizer

logger = logging.getLogger(__name__)


TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)


class MultipleFileSentences(object):
    def __init__(self, directory, prefixes, n_workers=cpu_count()-1, job_size=1):
        self.directory = directory
        self.prefixes = prefixes
        self.n_workers = n_workers
        self.job_size = job_size

    def __iter__(self):
        jobs = partition_all(self.job_size, iter_jsons(self.directory, self.prefixes))
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for j, job in enumerate(jobs):
                futures.append(executor.submit(process_job, job))
                if j % self.n_workers == 0:
                    for f in as_completed(futures):
                        results = f.result()
                        for result in results:
                            if result is not None:
                                yield result
                    futures = []
            for f in as_completed(futures):
                results = f.result()
                for result in results:
                    if result is not None:
                        yield result


def iter_jsons(directory, prefixes):
    for root, dirnames, filenames in walk(directory):
        for filename in fnmatch.filter(filenames, '*.jsonl.gz'):
            if filename.split('.')[0] in prefixes:
                yield path.join(root, filename)


def process_job(job):
    results = []
    for filepath in job:
        result = process_file(filepath)
        if result is not None:
            results = results + result
    return results


def process_file(filepath):
    result = []
    for line in gzip.open(filepath):
        try:
            data = ujson.loads(line)
        except ValueError:
            try:
                data = json.loads(line)
            except ValueError as ve:
                data = ''
                logger.warn('DECODE FAIL: %s %s', filepath, ve.message)
        if 'text' in data:
            result.append(TOKENIZER.tokenize(data['text']))
    return result


@plac.annotations(
    in_dir=("Location of input directory"),
    out_dir=("Location of output directory"),
    config_file=("YAML config file"),
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
def main(in_dir, out_dir, config, skipgram=0, negative=5, n_workers=cpu_count()-1, window=10, size=200, min_count=10,
         nr_iter=2, job_size=1, max_docs=None):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with open(config, 'r') as cf:
        config = yaml.load(cf)

    for topic, sources in config['selection']['topics'].iteritems():
        logger.info('Topic: %s -> %s', topic, ' '.join(sources))
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
        sentences = ClippedCorpus(MultipleFileSentences(in_dir, sources, n_workers, job_size), max_docs=max_docs)
        model.build_vocab(sentences, progress_per=10000)
        model.train(sentences)

        model.save(path.join(out_dir, topic + '.model'))


if __name__ == '__main__':
    plac.call(main)
