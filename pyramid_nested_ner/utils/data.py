import bisect
import json
import string
from collections import defaultdict

from pyramid_nested_ner.data import DataPoint, Entity
from pyramid_nested_ner.vectorizers.text.char import CharVectorizer


def rasa_data_reader(path):
    for example in json.load(open(path, 'r'))['rasa_nlu_data']['common_examples']:
        entities = [
            Entity(
                entity['entity'],
                entity['value'],
                entity['start'],
                entity['end']
            ) for entity in example['entities']
        ]

        yield DataPoint(example['text'], entities)


def jsonline_data_reader(path):
    for example in (json.loads(jsonline) for jsonline in open(path, 'r')):
        text = "".join(c if c in string.printable else CharVectorizer.UNK for c in example['text'])
        entities = [
            Entity(
                entity['category'],
                # entity['title'],
                text[entity['start']:entity['end']],
                entity['start'],
                entity['end']
            ) for entity in example['entities']
        ]

        yield DataPoint(text, entities)


def wikipedia_article_data_reader(path):
    """
    New data reader for automatically annotated Wikipedia articles.
    """
    # for example in json.load(open(path, 'r'))['rasa_nlu_data']['common_examples']:
    for jsonline in open(path, 'r'):
        if not jsonline.strip().startswith("{\"id\":") or jsonline.strip().startswith("]"):
            continue

        article = json.loads(jsonline)

        entities = [
            {
                'title': entity['title'],
                'category': entity['category'],
                'start': entity['start'],
                'end': entity['end']
            }
            for entity in article['entities']
        ]

        sentences = list(sorted(
            [
                {
                    'start': entity['start'],
                    'end': entity['end']
                }
                for entity in article['sentences']
            ],
            key=lambda d: d['start']
        ))

        sentences_starts = [sentence['start'] for sentence in sentences]

        entity_map = defaultdict(list)
        for entity in entities:
            start = entity['start']
            sentence_index = bisect.bisect(sentences_starts, start) - 1
            entity_map[sentence_index].append(entity)

        for idx, sentence in enumerate(sentences):
            sentence_entities = entity_map[idx]

            if not sentence_entities:
                continue

            sentence_start = sentence['start']
            sentence_end = sentence['end']
            sentence_text = article['text'][sentence_start:sentence_end]

            yield DataPoint(sentence_text, entities)
