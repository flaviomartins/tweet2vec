#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from preprocessing import process_texts
from twokenize import twokenize


def test_twokenize_urlnonbreakingspace():
    toks = [twokenize.tokenizeRawTweetText(
        u'@Porsche : 2014 is already here #zebracar #LM24 http://bit.ly/18RUczp\u00a0 pic.twitter.com/cQ7z0c2hMg')]
    toks = process_texts(toks)
    assert toks == [[u'2014', u'#zebracar', u'#lm24']]


def test_twokenize_urlellipsis():
    toks = [twokenize.tokenizeRawTweetText(u'Some cars are in the river #NBC4NY http://t.co/WmK9Hc…')]
    toks = process_texts(toks)
    assert toks == [[u'car', u'river', u'#nbc4ny']]


def test_preprocessing_nospaceurl():
    toks = [twokenize.tokenizeRawTweetText(
        u'Brother of Oscar Pistorius, Carl Pistorius appears in court over road deathhttp://gu.com/p/3em7p/tw')]
    toks = process_texts(toks)
    assert toks == [[u'brother', u'oscar', u'pistoriu', u'carl', u'pistoriu', u'appear', u'court', u'road', u'death']]