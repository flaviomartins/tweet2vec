#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tarfile
import bz2
try:
    from cStringIO import StringIO
except ImportError:
    from StringIO import StringIO
import fnmatch
import os
import sys
import json
import gensim
from nltk.tokenize import TweetTokenizer

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class MultipleFileSentences(object):
    def __init__(self, basedir):
        self.basedir = basedir
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)

    def __iter__(self):
        for root, dirnames, filenames in os.walk(self.basedir):
            for filename in fnmatch.filter(filenames, '*.tar'):
                with tarfile.open(os.path.join(root, filename), 'r') as tar:
                    for tarinfo in tar:
                        print tarinfo
                        if tarinfo.isfile() and os.path.splitext(tarinfo.name)[1] == ".bz2":
                            f = tar.extractfile(tarinfo.name)
                            content = StringIO(bz2.decompress(f.read()))
                            line = content.readline()
                            while line != '':
                                try:
                                    data = json.loads(line)
                                    if 'text' in data:
                                        yield self.tokenizer.tokenize(data['text'])
                                except ValueError:
                                    pass
                                line = content.readline()
                            f.close()


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print "Usage: train_word2vec.py <inputfile> <modelname>"
        sys.exit(0)

    inputfile = sys.argv[1]
    modelname = sys.argv[2]
    sentences = MultipleFileSentences(inputfile)
    model = gensim.models.Word2Vec(sentences, size=200, window=10, min_count=10,
                                   workers=4, iter=1, sorted_vocab=1)
    model.save(modelname)
