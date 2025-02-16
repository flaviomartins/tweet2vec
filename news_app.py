from __future__ import print_function
from builtins import object

import io
import logging
from collections import defaultdict
from os import path
from timeit import default_timer as timer

from wsgiref import simple_server
import falcon

import plac
import json
import yaml
import operator
import numpy as np
from numpy import array, abs, dot, sum as np_sum, zeros
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from twokenize import twokenize
from preprocessing import process_texts

logger = logging.getLogger(__name__)


MAX_RESULTS_POOL = 1000
ALLOWED_ORIGINS = ['*']
VECTOR_SIZE = 400
REPRESENTATION_LIMIT = 10000


class CorsMiddleware(object):

    def process_request(self, request, response):
        origin = request.get_header('Origin')
        if '*' in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
            response.set_header('Access-Control-Allow-Origin', origin)


class TagAutocompleteResource(object):

    def __init__(self, global_model, models):
        self.global_model = global_model
        self.models = models
        self.models_vectors, self.models_names = self.init_models_vectors(global_model, models)
        self.models_lens, self.total_lens = self.init_models_lens(models)

    def init_models_vectors(self, global_model, models):
        mv = None
        names = []
        for name, model in self.models.items():
            # ave = model.wv.syn0norm.mean(axis=0)
            ave = get_model_word_vector(global_model, model, VECTOR_SIZE)
            if mv is not None:
                mv = np.vstack([mv, ave])
            else:
                mv = ave
            names.append(name)
        return mv, array(names)

    def init_models_lens(self, models):
        ml = None
        total_vocab_size = 0
        for name, model in models.items():
            size = len(model.wv.vocab)
            if ml is not None:
                ml = np.vstack([ml, size])
            else:
                ml = size
            total_vocab_size += size
        return ml, total_vocab_size

    def tokens(self, q):
        return twokenize.tokenizeRawTweetText(q)

    def most_similar(self, topic, tokens, limit):
        model = self.models[topic]
        # words_in_model = [tok for tok in tokens if tok in model]
        # model_similar = model.most_similar_cosmul(positive=words_in_model, topn=limit)
        # words_in_global_model = [tok[0] for tok in model_similar if tok[0] in self.global_model]
        words_in_global_model = [tok for tok in tokens if tok in self.global_model]
        return self.global_model.most_similar_cosmul(positive=words_in_global_model, topn=limit)

    def most_similar2(self, topic, tokens, limit):
        model = self.models[topic]
        words_in_model = [tok for tok in tokens if tok in model]

        cum = defaultdict(int)
        for word in words_in_model:
            most_similar = model.most_similar_cosmul(positive=word, topn=limit)
            for sim in most_similar:
                cum[sim[0]] += sim[1]
        sorted_cum = sorted(list(cum.items()), key=operator.itemgetter(1), reverse=True)
        return sorted_cum

    def suggestions(self, topic, q, limit):
        tokens = self.tokens(q)
        lemmas = process_texts([tokens], lemmatize=True)[0]
        word = lemmas[-1]
        context = lemmas
        logger.info('word: ' + word + ' context: ' + ' '.join(context))

        start = timer()

        qv = get_query_vector(lemmas, self.global_model, self.models, VECTOR_SIZE).reshape(1, -1)
        mv = self.models_vectors

        sims = abs(cosine_similarity(qv, mv).ravel())
        # norms = np.divide(self.total_lens, np.log(1 + self.models_lens)).ravel()
        dists = sims  # * norms

        ix = np.argsort(dists)[::-1]

        top_topics = self.models_names[ix]
        top_scores = dists[ix]

        end = timer()
        logger.info("time: %4.2fms", (end - start) * 10000)

        for i in range(len(ix)):
            logger.info('%s: %f', top_topics[i], top_scores[i])

        # Selecting the topic
        topic = top_topics[0]
        score = top_scores[0]
        logger.info('topic: %s: %f', topic, score)

        most_similar = self.most_similar(topic, context, limit)
        return topic, most_similar[:limit]

    def on_get(self, req, resp):
        topic = req.get_param('topic') or 'general'
        q = req.get_param('q') or ''
        limit = req.get_param_as_int('limit') or 10

        try:
            topic, suggestions = self.suggestions(topic, q, limit)
            data = {'topic': topic, 'suggestions': [hit for hit in suggestions]}
            result = json.dumps(data)
        except Exception as ex:
            logger.error(ex)

            description = ('Aliens have attacked our base! We will '
                           'be back as soon as we fight them off. '
                           'We appreciate your patience.')

            raise falcon.HTTPServiceUnavailable(
                'Service Outage',
                description,
                30)

        resp.body = result

        resp.set_header('Powered-By', 'Falcon')
        resp.status = falcon.HTTP_200


def get_model_word_vector(global_model, model, vector_size):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = zeros((vector_size,), dtype="float32")
    #
    nwords = 0.
    #
    # Loop over words
    for word in model.wv.vocab:
        if word in global_model.wv.vocab:
            if nwords > REPRESENTATION_LIMIT:
                break
            nwords = nwords + 1.
            featureVec = np.add(featureVec, global_model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


def get_global_word_vector(word, global_model, models, vector_size):
    if word in global_model.wv.vocab:
        return global_model[word]
    return None


def get_query_vector(words, global_model, models, vector_size):
    # Function to average all of the word vectors in a given
    # query
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = zeros((vector_size,), dtype="float32")
    #
    nwords = 0.
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        wv = get_global_word_vector(word, global_model, models, vector_size)
        if np.isnan(np_sum(wv)):
            break
        nwords = nwords + 1.
        featureVec = np.add(featureVec, wv)
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nwords)
    return featureVec


# Useful for debugging problems in your API; works with pdb.set_trace(). You
# can also use Gunicorn to host your app. Gunicorn can be configured to
# auto-restart workers when it detects a code change, and it also works
# with pdb.


@plac.annotations(
    in_global=("Location of global model"),
    in_dir=("Location of input models"),
    config_file=("YAML config file"),
    host=("Bind to host", "option", "b", str),
    port=("Bind to port", "option", "p", int),
)
def main(in_global, in_dir, config_file, host='127.0.0.1', port=8001):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with io.open(config_file, 'rt', encoding='utf-8') as cf:
        config = yaml.load(cf)

    if path.exists(in_global):
        global_model = Word2Vec.load(in_global)
        global_model.init_sims()

    models = {}
    for topic, sources in list(config['selection']['topics'].items()):
        logger.info('Topic: %s -> %s', topic, ' '.join(sources))
        fullfn = path.join(in_dir, topic) + '.model'
        if path.exists(fullfn):
            model = Word2Vec.load(fullfn)
            model.init_sims()
            models[topic] = model
        else:
            logger.error('Missing model: %s', fullfn)

    # Configure your WSGI server to load "quotes.app" (app is a WSGI callable)
    app = falcon.API(middleware=[
        CorsMiddleware()
    ])

    tag_autocomplete = TagAutocompleteResource(global_model, models)
    app.add_route('/', tag_autocomplete)

    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()


if __name__ == '__main__':
    plac.call(main)
