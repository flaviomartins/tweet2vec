#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import io
import six
import logging
import plac

from pkg_resources import parse_version
from multiprocessing import cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import sklearn
from gensim import utils
from corpus.jsonl import JsonlDirSentences
from corpus.csv import CsvDirSentences

logger = logging.getLogger(__name__)


def split_on_space(text):
    return text.split(' ')


def iter_sentences(sentences):
    for tid, raw, sentence in sentences:
        unicode_sentence = []
        for token in sentence:
            if isinstance(token, six.binary_type):
                token = token.decode('utf-8')
            unicode_sentence.append(token)
        yield ' '.join([token for token in unicode_sentence])


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    n_workers=("Number of workers", "option", "n", int),
    nr_clusters=("Number of clusters", "option", "t", int),
    nr_iter=("Number of iterations", "option", "i", int),
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int),
    fformat=("By default (ff=jsonl), JSONL format is used."
             "Otherwise (ff='csv'), CSV format is used.", "option", "ff", str),
    no_lemmas=("Disable Lemmatization.", "flag", "nl", bool),
    max_features=("Maximum number of features (dimensions) to extract from text.", "option", "D", int),
    binary_tf=("Make tf term in tf-idf binary.", "flag", "b", bool),
    sublinear_tf=("Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).", "flag", "l", bool),
    no_idf=("Disable Inverse Document Frequency feature weighting.", "flag", "ni", bool),
    kld=("Use generalized Kullback-Leibler divergence (equivalent to PLSI).", "flag", "kld", bool),
    verbose=("Print progress reports inside k-means algorithm.", "flag", "v", bool)
)
def main(in_dir, out_loc, n_workers=cpu_count()-1, nr_clusters=10, nr_iter=100,
         job_size=1, max_docs=None, fformat='jsonl', no_lemmas=False, max_features=10000,
         binary_tf=False, sublinear_tf=False, no_idf=False, kld=False, verbose=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    lemmatize = not no_lemmas
    use_idf = not no_idf
    # Set training parameters.
    num_clusters = nr_clusters
    iterations = nr_iter

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

    logger.info('TfidfVectorizer')
    vectorizer = TfidfVectorizer(input='content', encoding='utf-8',
                                 decode_error='strict', strip_accents=None, lowercase=False,
                                 preprocessor=None, tokenizer=split_on_space, analyzer='word',
                                 stop_words=None, token_pattern=None,
                                 max_df=0.5, min_df=5,
                                 max_features=max_features, vocabulary=None, binary=binary_tf,  # binary_tf -> tf=1 cap
                                 norm='l2', use_idf=use_idf, smooth_idf=True,
                                 sublinear_tf=sublinear_tf)  # sublinear_tf -> tf = 1 + log(tf)

    tfidf = vectorizer.fit_transform(iter_sentences(sentences))

    if kld and parse_version(sklearn.__version__) >= parse_version('0.19'):
        logger.info('NMF-KL')
        nmf = NMF(n_components=num_clusters, max_iter=iterations, random_state=1, verbose=verbose,
                  beta_loss='kullback-leibler', solver='mu', alpha=.1, l1_ratio=.5)
    else:
        logger.info('NMF')
        nmf = NMF(n_components=num_clusters, max_iter=iterations, random_state=1, verbose=verbose,
                  alpha=.1, l1_ratio=.5)

    W = nmf.fit_transform(tfidf)
    H = nmf.components_

    labels_ = W.argmax(axis=1)

    terms = vectorizer.get_feature_names()

    with io.open(out_loc, 'wt', encoding='utf-8') as f:
        for i, component in enumerate(nmf.components_):
            f.write(u'{:d}'.format(i))
            order_centroids = component.argsort()[::-1]
            for ind in order_centroids[:20]:
                f.write(u' {}'.format(terms[ind]))
            f.write(u'\n')


if __name__ == '__main__':
    plac.call(main)
