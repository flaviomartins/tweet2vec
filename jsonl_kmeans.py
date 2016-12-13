#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division

import gzip
import io
import logging
from multiprocessing import cpu_count
from os import path

import six
from concurrent.futures import ProcessPoolExecutor, as_completed
from toolz import partition_all

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
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans

from gensim import utils
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from twokenize import twokenize
from ldig_detector import Detector


logger = logging.getLogger(__name__)
stops = set(stopwords.words('english'))  # nltk stopwords list
stemmer = PorterStemmer()
detector = Detector()


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
        for filename in fnmatch.filter(filenames, '*.jsonl*'):
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
        f = gzip.open(filepath)
    else:
        f = io.open(filepath, 'rt', encoding='utf-8')

    result = []
    count = 0
    for lno, line in enumerate(f):
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
            if detector is not None:
                if lno <= 5:
                    if 'id' in data:
                        long_id = data['id']
                        detected = detector.detect(long_id, data['text'])
                        if detected == 'en':
                            result.append(twokenize.tokenizeRawTweetText(data['text']))
                            count += 1
                else:
                    if count >= 0.5*lno:
                        result.append(twokenize.tokenizeRawTweetText(data['text']))
                        count += 1
                    else:
                        logger.warn('Probably not en : %s : %d < 0.5*%d : %s', detected, count, lno, filepath)
                        result = []
                        break
            else:
                result.append(twokenize.tokenizeRawTweetText(data['text']))
                count += 1
    f.close()
    return process_texts(result)


# Additionally, these things are "filtered", meaning they shouldn't appear on the final token list.
Filtered  = re.compile(
    unicode(twokenize.regex_or(
        twokenize.Hearts,
        twokenize.url,
        twokenize.Email,
        twokenize.timeLike,
        twokenize.numberWithCommas,
        twokenize.numComb,
        twokenize.emoticon,
        twokenize.Arrows,
        twokenize.entity,
        twokenize.punctSeq,
        twokenize.arbitraryAbbrev,
        twokenize.separators,
        twokenize.decorations,
        # twokenize.embeddedApostrophe,
        # twokenize.Hashtag,
        twokenize.AtMention,
        "(?:RT|rt)".encode('utf-8')
    ).decode('utf-8')), re.UNICODE)


def process_texts(texts, lemmatize=True):
    """
    Function to process texts. Following are the steps we take:

    1. Filter mentions, etc.
    1. Lowercasing.
    2. Stopword Removal.
    3. Lemmatization (not stem since stemming can reduce the interpretability).
    OR
    3. Possessive Filtering.

    Parameters:
    ----------
    texts: Tokenized texts.

    Returns:
    -------
    texts: Pre-processed tokenized texts.
    """

    texts = [[token for token in line if not Filtered.match(token)] for line in texts]
    texts = [[token.lower() for token in line] for line in texts]
    texts = [[token for token in line if token not in stops] for line in texts]
    if lemmatize:
        texts = [[
                     re.split('/', token)[0] for token in utils.lemmatize(' '.join(line),
                                                                          allowed_tags=re.compile('(NN)'),
                                                                          min_length=3)
                     ] for line in texts
                 ]
    else:
        texts = [[token.replace("'s", "") for token in line] for line in texts]
        texts = [[stemmer.stem(token) for token in line] for line in texts]
        texts = [[token for token in line if 3 <= len(token)] for line in texts]
    return texts


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
    sentences = utils.ClippedCorpus(MultipleFileSentences(in_dir, n_workers, job_size), max_docs=max_docs)

    logger.info('KMeans')
    vectorizer = TfidfVectorizer(input='content', encoding='utf-8',
                                 decode_error='strict', strip_accents=None, lowercase=False,
                                 preprocessor=None, tokenizer=just_split, analyzer='word',
                                 stop_words=None, token_pattern=r"(?u)\s.*\s",
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
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :20]:
            print(' %s' % terms[ind], end='')
        print()


def just_split(text):
    return text.split(' ')


if __name__ == '__main__':
    plac.call(main)
