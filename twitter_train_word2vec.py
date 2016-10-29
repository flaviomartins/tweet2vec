#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import fnmatch
import json
import gensim
from nltk.tokenize import TweetTokenizer

import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


if sys.version.startswith("3"):
    import io
    io_method = io.BytesIO
else:
    try:
        from cStringIO import StringIO
    except ImportError:
        from StringIO import StringIO
    io_method = StringIO


class MultipleFileSentences(object):
    def __init__(self, basedir):
        self.basedir = basedir
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        self.command = 'pbzip2'
        try:
            subprocess.Popen(['pbzip2', '--version'])
        except OSError:
            subprocess.Popen(['bzip2', '--version'])
            self.command = 'bzip2'
        print 'will use ' + self.command

    def __iter__(self):
        for root, dirnames, filenames in os.walk(self.basedir):
            for filename in fnmatch.filter(filenames, '*.tar'):
                fullfn = os.path.join(root, filename)
                print fullfn
                p1 = subprocess.Popen(['tar', 'xfO', fullfn, '--wildcards', '--no-anchored', '*.bz2'], stdout=subprocess.PIPE)
                p2 = subprocess.Popen([self.command, '-d'], stdin=p1.stdout, stdout=subprocess.PIPE)
                p1.stdout.close()
                fh = io_method(p2.communicate()[0])
                assert p2.returncode == 0

                for line in fh:
                    try:
                        data = json.loads(line)
                        if 'text' in data:
                            yield self.tokenizer.tokenize(data['text'])
                    except ValueError:
                        pass


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
