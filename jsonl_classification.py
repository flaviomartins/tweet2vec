#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
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
from gensim.models import Phrases, LdaMulticore
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from gensim.utils import ClippedCorpus, lemmatize
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords


logger = logging.getLogger(__name__)


TOKENIZER = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
stops = set(stopwords.words('english'))  # nltk stopwords list


class MultipleFileSentences(object):
    def __init__(self, directory, n_workers=cpu_count(), job_size=1):
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
        for filename in fnmatch.filter(filenames, '*.jsonl*'):
            yield path.join(root, filename)


def process_job(job):
    results = []
    for filepath in job:
        result = process_file(filepath)
        if result is not None:
            results = results + result
    return results


def process_file(filepath):
    if filepath.endswith('.gz'):
        f = gzip.open(filepath)
    else:
        f = open(filepath)

    result = []
    for line in f:
        try:
            data = ujson.loads(line)
        except ValueError:
            try:
                data = json.loads(line)
            except ValueError as ve:
                data = ''
                logger.warn('DECODE FAIL: %s %s', filepath, ve.message)
        if 'text' in data:
            result.append(TOKENIZER.tokenize(data['text'].encode('unicode-escape')))
    f.close()
    return process_texts(result)


def process_texts(texts):
    """
    Function to process texts. Following are the steps we take:

    1. Stopword Removal.
    1.1 Remove mentions
    2. Collocation detection.
    3. Lemmatization (not stem since stemming can reduce the interpretability).

    Parameters:
    ----------
    texts: Tokenized texts.

    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """
    texts = [[word for word in line if word not in stops] for line in texts]
    texts = [[word for word in line if not word.startswith('@')] for line in texts]
    texts = [[word for word in line if not word.startswith('http')] for line in texts]
    texts = [[word.split('/')[0] for word in lemmatize(' '.join(line), min_length=3)] for line in texts]
    return texts


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
)
def main(in_dir, out_loc, skipgram=0, negative=5, n_workers=cpu_count(), window=10, size=200, min_count=10, nr_iter=2, job_size=1):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    sentences = ClippedCorpus(MultipleFileSentences(in_dir, n_workers, job_size), max_docs=10000)

    logger.info('Bigram phrases')
    bigram_transformer = Phrases(sentences)
    logger.info('Bigram phraser')
    bigram_phraser = Phraser(bigram_transformer)

    logger.info('Trigram phrases')
    trigram_transformer = Phrases(bigram_phraser[sentences])
    logger.info('Trigram phraser')
    trigram_phraser = Phraser(trigram_transformer)

    print(trigram_phraser[bigram_phraser['the', 'best', 'way', 'music', 'video', 'new', 'york', 'city']])

    logger.info('Dictionary')
    dictionary = Dictionary(trigram_phraser[bigram_phraser[sentences]])
    logger.info('Corpus')
    corpus = [dictionary.doc2bow(text) for text in trigram_phraser[bigram_phraser[sentences]]]

    logger.info('LDA')
    ldamodel = LdaMulticore(corpus=corpus, num_topics=10, id2word=dictionary)
    ldamodel.show_topics()

if __name__ == '__main__':
    plac.call(main)
