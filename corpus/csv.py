# -*- coding: utf-8 -*-

import gzip
import unicodecsv as csv

import logging
from multiprocessing import cpu_count
from os import path

# fails to import scandir < 3.5
try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk
import fnmatch

from concurrent.futures import ProcessPoolExecutor, as_completed
from toolz import partition_all

from twokenize import twokenize
from preprocessing import process_texts

logger = logging.getLogger(__name__)


class CsvDirSentences(object):
    def __init__(self, directory, n_workers=cpu_count()-1, job_size=1):
        self.directory = directory
        self.n_workers = n_workers
        self.job_size = job_size

    def __iter__(self):
        jobs = partition_all(self.job_size, iter_files(self.directory))
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


def iter_files(directory):
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
