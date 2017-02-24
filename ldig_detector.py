# -*- coding: utf-8 -*-

from __future__ import division
from builtins import object

from ldig import ldig
import numpy as np
from os import path


class Detector(object):
    def __init__(self, modeldir=path.join(path.dirname(__file__), 'ldig/models/model.latin.20120315')):
        self.ldig = ldig.ldig(modeldir)
        self.features = self.ldig.load_features()
        self.trie = self.ldig.load_da()
        self.labels = self.ldig.load_labels()
        self.param = np.load(self.ldig.param)
        self.cache = {}

    # prediction probability
    def predict(self, events):
        sum_w = np.dot(self.param[events.keys(), ].T, events.values())
        exp_w = np.exp(sum_w - sum_w.max())
        return exp_w / exp_w.sum()

    def likelihood(self, st):
        label, text, org_text = ldig.normalize_text(st)
        events = self.trie.extract_features(u"\u0001" + text + u"\u0001")
        y = self.predict(events)
        predict_k = y.argmax()

        predict_lang = self.labels[predict_k]
        if y[predict_k] < 0.6:
            predict_lang = ""
        return predict_lang

    def detect(self, long_id, st):
        if long_id in self.cache:
            return self.cache[long_id]
        else:
            predict_lang = self.likelihood(st)

            if id > 0:
                self.cache[long_id] = predict_lang

            return predict_lang
