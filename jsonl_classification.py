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
import re
from gensim.models import Phrases, LdaModel, LdaMulticore
from gensim.models.phrases import Phraser
from gensim.corpora import Dictionary
from gensim import utils
from nltk.corpus import stopwords
from twokenize import twokenize

logger = logging.getLogger(__name__)
stops = set(stopwords.words('english'))  # nltk stopwords list


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
            result.append(twokenize.tokenizeRawTweetText(data['text'].encode('unicode-escape')))
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

    texts = [[word for word in line if not Filtered.match(word)] for line in texts]
    texts = [[word for word in line if word not in stops] for line in texts]
    if lemmatize:
        texts = [[
                     word.split('/')[0] for word in utils.lemmatize(' '.join(line),
                                                                    allowed_tags=re.compile('(NN)'),
                                                                    min_length=3)
                     ] for line in texts
                 ]
    else:
        texts = [[word.replace("'s", "") for word in line if word not in stops] for line in texts]
        texts = [[token.lower() for token in line if 3 <= len(token)] for line in texts]
    return texts


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    n_workers=("Number of workers", "option", "n", int),
    nr_topics=("Number of topics", "option", "t", int),
    nr_passes=("Number of passes", "option", "p", int),
    nr_iter=("Number of iterations", "option", "i", int),
    chunk_size=("Chunk size", "option", "c", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int)
)
def main(in_dir, out_loc, n_workers=cpu_count()-1, nr_topics=100, chunk_size=2000, nr_passes=20, nr_iter=400,
         job_size=1, max_docs=None):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Set training parameters.
    num_topics = nr_topics
    chunksize = chunk_size
    passes = nr_passes
    iterations = nr_iter
    eval_every = None  # Don't evaluate model perplexity, takes too much time.
    sentences = utils.ClippedCorpus(MultipleFileSentences(in_dir, n_workers, job_size), max_docs=max_docs)

    logger.info('Bigram phrases')
    bigram_transformer = Phrases(sentences, min_count=5, threshold=100)
    logger.info('Bigram phraser')
    bigram_phraser = Phraser(bigram_transformer)

    logger.info('Trigram phrases')
    trigram_transformer = Phrases(bigram_phraser[sentences], min_count=5, threshold=100)
    logger.info('Trigram phraser')
    trigram_phraser = Phraser(trigram_transformer)

    logger.info('Dictionary')
    dictionary = Dictionary(trigram_phraser[bigram_phraser[sentences]])
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    logger.info('Corpus')
    corpus = [dictionary.doc2bow(text) for text in trigram_phraser[bigram_phraser[sentences]]]

    logger.info('id2word')
    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    logger.info('LDA')
    model = LdaMulticore(corpus=corpus, id2word=id2word, chunksize=chunksize, random_state=1, workers=n_workers,
                         # alpha='auto', eta='auto',
                         iterations=iterations, num_topics=num_topics,
                         passes=passes, eval_every=eval_every)

    top_topics = model.top_topics(corpus, num_words=20)

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)

    from pprint import pprint
    pprint(top_topics)

    model.save(out_loc)


if __name__ == '__main__':
    plac.call(main)
