# -*- coding: utf-8 -*-

import re
from twokenize import twokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from gensim import utils


stops = set(stopwords.words('english'))  # nltk stopwords list
stemmer = PorterStemmer()

# Additionally, these things are "filtered", meaning they shouldn't appear on the final token list.
Filtered = re.compile(
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
