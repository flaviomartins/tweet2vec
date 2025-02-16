#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import sys
import io
import six
import logging
import plac

from multiprocessing import cpu_count
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import numpy as np
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
    job_size=("Job size in number of lines", "option", "j", int),
    max_docs=("Limit maximum number of documents", "option", "L", int),
    fformat=("By default (ff=jsonl), JSONL format is used."
             "Otherwise (ff='csv'), CSV format is used.", "option", "ff", str),
    no_lemmas=("Disable Lemmatization.", "flag", "nl", bool),
    max_features=("Maximum number of features (dimensions) to extract from text.", "option", "D", int),
    binary_tf=("Make tf term in tf-idf binary.", "flag", "b", bool),
    sublinear_tf=("Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).", "flag", "l", bool),
    no_idf=("Disable Inverse Document Frequency feature weighting.", "flag", "ni", bool),
    verbose=("Print progress reports inside k-means algorithm.", "flag", "v", bool)
)
def main(in_dir, out_loc, n_workers=cpu_count()-1, nr_clusters=10,
         job_size=1, max_docs=None, fformat='jsonl', no_lemmas=False, max_features=10000,
         binary_tf=False, sublinear_tf=False, no_idf=False, verbose=False):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    lemmatize = not no_lemmas
    use_idf = not no_idf
    # Set training parameters.
    num_clusters = nr_clusters

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

    X = vectorizer.fit_transform(iter_sentences(sentences))
    C = 1 - cosine_similarity(X.T)

    logger.info('AgglomerativeClustering')
    ward = AgglomerativeClustering(n_clusters=num_clusters, linkage='ward')

    ward.fit(C)

    labels = ward.labels_
    terms = vectorizer.get_feature_names()

    with io.open(out_loc, 'wt', encoding='utf-8') as f:
        for i in range(num_clusters):
            f.write(u'{:d}'.format(i))
            where = np.where(labels == i)
            for group in where:
                for ind in group:
                    f.write(u' {}'.format(terms[ind]))
            f.write(u'\n')


if __name__ == '__main__':
    plac.call(main)
