#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function

from twokenize import twokenize


def test_twokenize_urlnonbreakingspace():
    toks = twokenize.tokenizeRawTweetText(u'@Porsche : 2014 is already here #zebracar #LM24 http://bit.ly/18RUczp\u00a0 pic.twitter.com/cQ7z0c2hMg')
    assert toks == [u'@Porsche', u':', u'2014', u'is', u'already', u'here', u'#zebracar', u'#LM24', u'http://bit.ly/18RUczp', u'pic.twitter.com/cQ7z0c2hMg']

def test_twokenize_urlellipsis():
    toks = twokenize.tokenizeRawTweetText(u'Some cars are in the river #NBC4NY http://t.co/WmK9Hc…')
    assert toks == [u'Some', u'cars', u'are', u'in', u'the', u'river', u'#NBC4NY', u'http://t.co/WmK9Hc', u'\u2026']


def test_twokenize_nospaceurl():
    toks = twokenize.tokenizeRawTweetText(
        u'Brother of Oscar Pistorius, Carl Pistorius appears in court over road deathhttp://gu.com/p/3em7p/tw')
    assert toks == [u'Brother', u'of', u'Oscar', u'Pistorius', u',', u'Carl', u'Pistorius', u'appears', u'in', u'court',
                    u'over', u'road', u'death', u'http://gu.com/p/3em7p/tw']