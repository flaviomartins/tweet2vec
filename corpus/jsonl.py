# -*- coding: utf-8 -*-

import gzip
import io
import six
try:
    import ujson
except ImportError:
    import json as ujson
import json

import logging
import itertools
import multiprocessing
from os import path

# fails to import scandir < 3.5
try:
    from os import scandir, walk
except ImportError:
    from scandir import scandir, walk
import fnmatch

from gensim import utils
from twokenize import twokenize
from preprocessing import process_texts

logger = logging.getLogger(__name__)

# ignore articles shorter than ARTICLE_MIN_WORDS (after full preprocessing)
ARTICLE_MIN_WORDS = 3


class JsonlDirSentences(object):
    def __init__(self, directory, n_workers=multiprocessing.cpu_count()-1, job_size=1, lemmatize=True, prefixes=None):
        self.directory = directory
        self.n_workers = n_workers
        self.job_size = job_size
        self.lemmatize = lemmatize
        self.prefixes = prefixes

    def __iter__(self):
        files = iter_files(self.directory, self.prefixes)
        articles, articles_all = 0, 0
        positions, positions_all = 0, 0
        pool = multiprocessing.Pool(self.n_workers)
        # process the corpus in smaller chunks of docs, because multiprocessing.Pool
        # is dumb and would load the entire input into RAM at once...
        for group in utils.chunkize(files, chunksize=self.job_size * self.n_workers, maxsize=1):
            for texts in pool.imap(process_file, itertools.izip(group, itertools.repeat(self.lemmatize))):
                for tokens in texts:
                    articles_all += 1
                    positions_all += len(tokens)
                    # article redirects and short stubs are pruned here
                    if len(tokens) < ARTICLE_MIN_WORDS:
                        continue
                    articles += 1
                    positions += len(tokens)
                    yield tokens
        pool.terminate()

        logger.info(
            "finished iterating over corpus of %i documents with %i positions"
            " (total %i documents, %i positions before pruning articles shorter than %i words)",
            articles, positions, articles_all, positions_all, ARTICLE_MIN_WORDS)
        self.length = articles  # cache corpus length


def iter_files(directory, prefixes):
    for root, dirnames, filenames in walk(directory):
        for filename in fnmatch.filter(filenames, '*.jsonl*'):
            if prefixes is None or filename.lower().split('.')[0] in prefixes:
                yield path.join(root, filename)


def process_file(args):
    filepath, lemmatize = args
    if filepath.endswith('.gz'):
        f = gzip.open(filepath)
    else:
        f = io.open(filepath, 'rt', encoding='utf-8')

    result = []
    for line in f:
        if isinstance(line, six.binary_type):
            try:
                line = line.decode('utf-8')
            except UnicodeDecodeError as ude:
                logger.warn('DECODE FAIL: %s %s', filepath, ude.message)
                continue
        try:
            data = ujson.loads(line)
        except ValueError:
            try:
                data = json.loads(line)
            except ValueError as ve:
                logger.warn('DECODE FAIL: %s %s', filepath, ve.message)
                continue
        if 'text' in data:
            result.append(twokenize.tokenizeRawTweetText(data['text']))
    f.close()
    return process_texts(result, lemmatize=lemmatize)
