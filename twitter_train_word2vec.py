#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function, unicode_literals, division
import io
from multiprocessing import cpu_count
import tarfile
import bz2
from subprocess import PIPE, Popen
import logging
from os import path
import os
import fnmatch

import plac
try:
    import ujson as json
except ImportError:
    import json
from gensim.models import Word2Vec
from nltk.tokenize import TweetTokenizer

logger = logging.getLogger(__name__)


NATIVE_METHOD = 'native'
COMPAT_METHOD = 'compat'
BUFSIZE = 64 * 1024**2
PROGRESS_PER = 10000


class MultipleFileSentences(object):
    def __init__(self, directory):
        self.directory = directory
        self.tokenizer = TweetTokenizer(preserve_case=False, reduce_len=True, strip_handles=True)
        self.method = NATIVE_METHOD
        self.command = 'pbzip2'
        try:
            Popen(['pbzip2', '--version'])
        except OSError:
            try:
                Popen(['bzip2', '--version'])
                self.command = 'bzip2'
            except OSError:
                self.method = COMPAT_METHOD
                self.command = 'python tarfile/bz2'
        print('method: ' + self.method + '\nusing: ' + self.command)

    @staticmethod
    def my_json_loads(content):
        try:
            data = json.loads(content)
        except ValueError:
            data = ''
        return data

    def __iter__(self):
        for root, dirnames, filenames in os.walk(self.directory):
            for filename in sorted(fnmatch.filter(filenames, '*.tar')):
                fullfn = path.join(root, filename)
                print(fullfn)

                if self.method == NATIVE_METHOD:
                    p1 = Popen(['tar', 'xfO', fullfn, '--wildcards', '--no-anchored', '*.bz2'], bufsize=BUFSIZE, stdout=PIPE)
                    p2 = Popen([self.command, '-dc'], bufsize=BUFSIZE, stdin=p1.stdout, stdout=PIPE)
                    p1.stdout.close()
                    for line in p2.stdout:
                        data = self.my_json_loads(line)
                        if 'text' in data:
                            yield self.tokenizer.tokenize(data['text'])
                else:
                    with tarfile.open(fullfn, 'r') as tar:
                        for tarinfo in tar:
                            if tarinfo.isfile() and os.path.splitext(tarinfo.name)[1] == ".bz2":
                                f = tar.extractfile(tarinfo.name)
                                content = io.BytesIO(bz2.decompress(f.read()))
                                for line in content:
                                    data = self.my_json_loads(line)
                                    if 'text' in data:
                                        yield self.tokenizer.tokenize(data['text'])


@plac.annotations(
    in_dir=("Location of input directory"),
    out_loc=("Location of output file"),
    n_workers=("Number of workers", "option", "n", int),
    size=("Dimension of the word vectors", "option", "d", int),
    window=("Context window size", "option", "w", int),
    min_count=("Min count", "option", "m", int),
    negative=("Number of negative samples", "option", "g", int),
    nr_iter=("Number of iterations", "option", "i", int),
)
def main(in_dir, out_loc, negative=5, n_workers=cpu_count(), window=5, size=200, min_count=10, nr_iter=2):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(
        size=size,
        window=window,
        min_count=min_count,
        workers=n_workers,
        sample=1e-5,
        negative=negative,
        iter=nr_iter
    )
    sentences = MultipleFileSentences(in_dir)
    model.build_vocab(sentences, progress_per=PROGRESS_PER)
    model.train(sentences)

    model.save(out_loc)


if __name__ == '__main__':
    plac.call(main)
