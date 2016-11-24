from __future__ import print_function, unicode_literals, division
import logging
from os import path

from wsgiref import simple_server
import falcon

import plac
import json
import yaml
import re
from gensim.models import Word2Vec
from gensim import utils
from nltk.corpus import stopwords
from twokenize import twokenize


logger = logging.getLogger(__name__)
stops = set(stopwords.words('english'))  # nltk stopwords list


MAX_RESULTS_POOL = 1000
ALLOWED_ORIGINS = ['*']


class CorsMiddleware(object):

    def process_request(self, request, response):
        origin = request.get_header('Origin')
        if '*' in ALLOWED_ORIGINS or origin in ALLOWED_ORIGINS:
            response.set_header('Access-Control-Allow-Origin', origin)


class TagAutocompleteResource:

    def __init__(self, models):
        self.models = models

    def tokens(self, q):
        return process_texts([twokenize.tokenizeRawTweetText(q)])[0]

    def most_similar(self, topic, tokens, limit):
        model = self.models[topic]
        return model.most_similar(positive=tokens, topn=limit)

    def suggestions(self, topic, q, limit):
        tokens = self.tokens(q)
        word = tokens[-1]
        context = tokens[:-1]
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


# Additionally, these things are "filtered", meaning they shouldn't appear on the final token list.
Filtered  = re.compile(
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


def process_texts(texts, lemmatize=False):
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

    texts = [[word for word in line if not Filtered.match(word)] for line in texts]
    texts = [[word for word in line if word not in stops] for line in texts]
    if lemmatize:
        texts = [[
                     word.split('/')[0] for word in utils.lemmatize(' '.join(line),
                                                                    allowed_tags=re.compile('(NN)'),
                                                                    min_length=3)
                     ] for line in texts
                 ]
    else:
        texts = [[word.replace("'s", "") for word in line if word not in stops] for line in texts]
        texts = [[token.lower() for token in line if 3 <= len(token)] for line in texts]
    return texts


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
