#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import bz2
import io
import logging
import tarfile
from multiprocessing import cpu_count
from os import path
from subprocess import PIPE, Popen

import six
from concurrent.futures import ProcessPoolExecutor, as_completed
from toolz import partition_all

# fails to import scandir < 3.5
try:
    from scandir import scandir, walk
except ImportError:
    from os import scandir, walk
import fnmatch

import plac
try:
    import ujson
except ImportError:
    import json as ujson
import json
from gensim.models import Word2Vec
from gensim import utils
from twokenize import twokenize
from preprocessing import process_texts

logger = logging.getLogger(__name__)

NATIVE_METHOD = 'native'
COMPAT_METHOD = 'compat'


class MultipleFileSentences(object):
    def __init__(self, directory, n_workers=cpu_count()-1, job_size=100000):
        self.directory = directory
        self.n_workers = n_workers
        self.job_size = job_size
        self.method = NATIVE_METHOD
        self.command = 'pbzip2'
        try:
            Popen(['pbzip2', '--version'])
        except OSError:
            try:
                Popen(['bzip2', '--version'])
                self.command = 'bzip2'
            except OSError:
                self.method = COMPAT_METHOD
                self.command = 'python tarfile/bz2'
        print('method: ' + self.method + '\nusing: ' + self.command)

    def __iter__(self):
        for root, dirnames, filenames in walk(self.directory):
            for filename in sorted(fnmatch.filter(filenames, '*.tar')):
                fullfn = path.join(root, filename)
                logger.info("PROGRESS: processing file %s", fullfn)

                if self.method == NATIVE_METHOD:
                    p1 = Popen(['tar', 'xfO', fullfn, '--wildcards', '--no-anchored', '*.bz2'], bufsize=-1, stdout=PIPE)
                    p2 = Popen([self.command, '-dc'], bufsize=-1, stdin=p1.stdout, stdout=PIPE)
                    p1.stdout.close()

                    jobs = partition_all(self.job_size, p2.stdout)
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
                else:
                    with tarfile.open(fullfn, 'r') as tar:
                        for tarinfo in tar:
                            if tarinfo.isfile() and path.splitext(tarinfo.name)[1] == ".bz2":
                                f = tar.extractfile(tarinfo.name)
                                content = io.BytesIO(bz2.decompress(f.read()))
                                job = content.readlines()
                                with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
                                    future = executor.submit(process_job, job)
                                    results = future.result()
                                    for result in results:
                                        if result is not None:
                                            yield result


def process_job(job):
    results = []
    for line in job:
        result = process_line(line)
        if result is not None:
            results.append(result)
    return process_texts(results)


def process_line(line):
    if isinstance(line, six.binary_type):
        try:
            line = line.decode('utf-8')
        except UnicodeDecodeError as ude:
            logger.warn('DECODE FAIL: %s', ude.message)
            return None
    try:
        data = ujson.loads(line)
    except ValueError:
        try:
            data = json.loads(line)
        except ValueError as ve:
            data = ''
            logger.warn('DECODE FAIL: %s', ve.message)
    if 'text' in data:
        return twokenize.tokenizeRawTweetText(data['text'])
    else:
        return None


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
         job_size=1000, max_docs=None):
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
