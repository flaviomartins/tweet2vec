from __future__ import print_function
from builtins import object

import io
import logging
from collections import defaultdict
from os import path

from wsgiref import simple_server
import falcon

import plac
import json
import yaml
import operator
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from twokenize import twokenize
from preprocessing import process_texts

logger = logging.getLogger(__name__)


MAX_RESULTS_POOL = 1000
ALLOWED_ORIGINS = ['*']
VECTOR_SIZE = 400


class CorsMiddleware(object):

    def process_request(self, request, response):
        origin = request.get_header('Origin')
        if '*' in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
            response.set_header('Access-Control-Allow-Origin', origin)


class TagAutocompleteResource(object):

    def __init__(self, models):
        self.models = models
        self.models_vectors, self.models_names = self.init_models_vectors(models)
        self.models_lens, self.total_lens = self.init_models_lens(models)

    def init_models_vectors(self, models):
        mv = np.zeros((VECTOR_SIZE,), dtype="float32")
        names = ["None"]
        for name, model in self.models.items():
            ave = np.mean(model.wv.syn0norm, axis=0)
            mv = np.column_stack([mv, ave])
            names.append(name)
        return mv, np.array(names)

    def init_models_lens(self, models):
        ml = np.zeros((1,))
        total_vocab_size = 0
        for name, model in models.items():
            size = len(model.wv.vocab)
            ml = np.column_stack([ml, size])
            total_vocab_size += size
        return ml, total_vocab_size

    def tokens(self, q):
        return twokenize.tokenizeRawTweetText(q)

    def most_similar(self, topic, tokens, limit):
        model = self.models[topic]
        words_in_model = [tok for tok in tokens if tok in model]
        return model.most_similar_cosmul(positive=words_in_model, topn=limit)

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

        qv = get_query_vector(lemmas, self.models, VECTOR_SIZE)

        sims = np.dot(qv, self.models_vectors)
        # sims = self.total_lens / np.log(1 + self.models_lens)
        ix = np.argsort(sims, axis=0)[::-1]

        top_topics = self.models_names[ix]
        top_scores = sims[ix]

        for i in range(len(ix)):
            logger.info('Topic: %s: %f', top_topics[i], top_scores[i])

        # Selecting the topic
        topic = top_topics[0]
        score = top_scores[0]
        logger.info('Selected Topic: %s: %f', topic, score)

        most_similar = self.most_similar(topic, context, limit)
        return most_similar[:limit]

    def on_get(self, req, resp):
        topic = req.get_param('topic') or 'general'
        q = req.get_param('q') or ''
        limit = req.get_param_as_int('limit') or 10

        try:
            suggestions = self.suggestions(topic, q, limit)
            result = json.dumps([hit[0] for hit in suggestions])
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


def get_global_word_vector(word, models, vector_size):
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((vector_size,), dtype="float32")
    #
    nmodels = 0.
    #
    # Loop over each model
    for name, model in models.items():
        if word in model.wv.vocab:
            nmodels = nmodels + 1.
            featureVec = np.add(featureVec, model[word])
    #
    # Divide the result by the number of words to get the average
    featureVec = np.divide(featureVec, nmodels)
    return featureVec


def get_query_vector(words, models, vector_size):
    # Function to average all of the word vectors in a given
    # query
    #
    # Pre-initialize an empty numpy array (for speed)
    featureVec = np.zeros((vector_size,), dtype="float32")
    #
    nwords = 0.
    #
    # Loop over each word in the review and, if it is in the model's
    # vocaublary, add its feature vector to the total
    for word in words:
        wv = get_global_word_vector(word, models, vector_size)
        if np.isnan(np.sum(wv)):
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
    in_dir=("Location of input model"),
    config_file=("YAML config file"),
    host=("Bind to host", "option", "b", str),
    port=("Bind to port", "option", "p", int),
)
def main(in_dir, config_file, host='127.0.0.1', port=8001):
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

    with io.open(config_file, 'rt', encoding='utf-8') as cf:
        config = yaml.load(cf)

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

    tag_autocomplete = TagAutocompleteResource(models)
    app.add_route('/', tag_autocomplete)

    httpd = simple_server.make_server(host, port, app)
    httpd.serve_forever()


if __name__ == '__main__':
    plac.call(main)
