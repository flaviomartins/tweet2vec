#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import logging
import plac
from os import path

from multiprocessing import cpu_count

from gensim.models import LdaModel, LdaMulticore
from gensim.models.wrappers.ldamallet import LdaMallet
from gensim.corpora import Dictionary
from gensim import utils
from corpus.jsonl import JsonlDirSentences
from corpus.csv import CsvDirSentences

logger = logging.getLogger(__name__)


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    n_workers=("Number of workers", "option", "n", int),
    nr_topics=("Number of topics", "option", "t", int),
    nr_passes=("Number of passes", "option", "p", int),
    nr_iter=("Number of iterations", "option", "i", int),
    chunk_size=("Chunk size", "option", "c", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int),
    mallet_path=("Path to mallet", "option", "-mallet_path", str),
)
def main(in_dir, out_loc, n_workers=cpu_count()-1, nr_topics=10, chunk_size=2000, nr_passes=1, nr_iter=400,
         job_size=1, max_docs=None, fformat='jsonl', no_lemmas=False, mallet_path=None):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    lemmatize = not no_lemmas
    # Set training parameters.
    num_topics = nr_topics
    chunksize = chunk_size
    passes = nr_passes
    iterations = nr_iter
    eval_every = None  # Don't evaluate model perplexity, takes too much time.

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

    logger.info('Dictionary')
    dictionary = Dictionary(sentences)
    dictionary.filter_extremes(no_below=20, no_above=0.5)

    logger.info('Corpus')
    corpus = [dictionary.doc2bow(text) for text in sentences]

    logger.info('id2word')
    # Make a index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    logger.info('LDA')

    if mallet_path is None:
        model = LdaMulticore(corpus=corpus, num_topics=num_topics, id2word=id2word, workers=n_workers,
                             chunksize=chunksize, passes=passes, batch=False,
                             # alpha='symmetric', eta=None,
                             decay=0.5, offset=1.0, eval_every=eval_every, iterations=iterations,
                             gamma_threshold=0.001, random_state=1)

        top_topics = model.top_topics(corpus, num_words=20)

        # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
        avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
        print('Average topic coherence: %.4f.' % avg_topic_coherence)

        from pprint import pprint
        pprint(top_topics)

        model.save(out_loc)
    else:
        mallet_model = LdaMallet(mallet_path,
                                 corpus=corpus, num_topics=num_topics, alpha=50, id2word=id2word, workers=n_workers,
                                 prefix=path.dirname(out_loc), optimize_interval=0, iterations=iterations,
                                 topic_threshold=0.0)


if __name__ == '__main__':
    plac.call(main)
