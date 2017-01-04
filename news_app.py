from __future__ import print_function, unicode_literals, division
import logging
from os import path
from collections import OrderedDict

from wsgiref import simple_server
import falcon

import plac
import json
import yaml
import requests
from gensim.models import Word2Vec
from twokenize import twokenize
from preprocessing import process_texts

logger = logging.getLogger(__name__)


MAX_RESULTS_POOL = 1000
ALLOWED_ORIGINS = ['*']
SELECTION_URL = 'http://localhost:8080/taily'


class CorsMiddleware(object):

    def process_request(self, request, response):
        origin = request.get_header('Origin')
        if '*' in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
            response.set_header('Access-Control-Allow-Origin', origin)


class TagAutocompleteResource:

    def __init__(self, models):
        self.models = models

    def tokens(self, q):
        return twokenize.tokenizeRawTweetText(q)

    def most_similar(self, topic, tokens, limit):
        model = self.models[topic]
        return model.most_similar_cosmul(positive=tokens, topn=limit)

    def suggestions(self, topic, q, limit):
        params = {'q': q}
        r = requests.get(SELECTION_URL, params=params)
        if r.status_code == 200:
            data = r.json(object_pairs_hook=OrderedDict)
            if 'response' in data and 'collections' in data['response']:
                cols = data['response']['collections']
                for col, scores in cols.iteritems():
                    topic = col
                    logger.info('Topic: ' + col)
                    break
        tokens = self.tokens(q)
        lemmas = process_texts([tokens])[0]
        word = lemmas[-1]
        context = lemmas
        logger.info('word: ' + word + ' context: ' + ' '.join(context))
        most_similar = self.most_similar(topic, context, MAX_RESULTS_POOL)
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

    with open(config_file, 'r') as cf:
        config = yaml.load(cf)

    models = {}
    for topic, sources in config['selection']['topics'].items():
        logger.info('Topic: %s -> %s', topic, ' '.join(sources))
        fullfn = path.join(in_dir, topic) + '.model'
        if path.exists(fullfn):
            models[topic] = Word2Vec.load(fullfn)
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
