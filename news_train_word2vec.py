#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division

import logging
import plac
import yaml
from os import path

from multiprocessing import cpu_count

from gensim.models import Word2Vec
from gensim import utils
from corpus.jsonl import JsonlDirSentences

logger = logging.getLogger(__name__)


@plac.annotations(
    in_dir=("Location of input directory"),
    out_dir=("Location of output directory"),
    config=("YAML config file"),
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

    for topic, sources in config['selection']['topics'].items():
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
        prefixes = [source.lower() for source in sources]
        sentences = utils.ClippedCorpus(JsonlDirSentences(in_dir, n_workers, job_size, prefixes=prefixes),
                                        max_docs=max_docs)

        model.build_vocab(sentences, progress_per=10000)
        model.train(sentences)

        model.save(path.join(out_dir, topic + '.model'))


if __name__ == '__main__':
    plac.call(main)
