import bisect
import itertools
import json
import string
from collections import defaultdict
from pathlib import Path

from pyramid_nested_ner.data import DataPoint, Entity
from pyramid_nested_ner.data.contextualized import SentenceWindowDataPoint
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


def jsonline_data_reader(path, limit=-1, drop_entities=None):
    for example in (json.loads(jsonline) for jsonline in itertools.islice(open(path, 'r'), limit)):
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
        if drop_entities:
            entities = list(filter(lambda entity: entity.name not in drop_entities, entities))

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


def wrg_reader(path):
    with open(path, 'r') as fp:
        lines = fp.readlines()

    for idx in range(0, len(lines), 4):
        text, pos, tags = lines[idx:idx + 3]
        text, pos, tags = text.strip(), pos.strip(), tags.strip()

        token_offsets = []
        last_offset = 0
        while (offset := text.find(" ", last_offset)) > -1:
            token_offsets.append((last_offset, offset))
            last_offset = offset + 1
        token_offsets.append((last_offset, len(text)))

        tokens = list(zip(text.split(), token_offsets))

        entities = []
        if len(tags) > 0:
            for tag in tags.split("|"):
                offsets, category = tag.split()
                start, end = offsets.split(",")
                start, end = int(start), int(end)
                covered = tokens[start:end + 1]
                first, last = covered[0][1][0], covered[-1][1][1]
                first, last = int(first), int(last)

                entities.append(
                    Entity(
                        category,
                        text[first:last],
                        first,
                        last
                    )
                )

        yield DataPoint(text, entities)


def wrg_sentence_window_reader(path, window_size=5):
    path = Path(path)
    for file in path.iterdir():
        dataset = list(wrg_reader(str(file.absolute())))

        pre_buffer = []
        current = dataset.pop(0)
        post_buffer = dataset[:window_size]

        yield SentenceWindowDataPoint(current.text, current.entities, pre_buffer, post_buffer)

        while len(dataset) > 0:
            pre_buffer.append(current)
            pre_buffer = pre_buffer[-window_size:]
            current = dataset.pop(0)
            post_buffer = dataset[:window_size]

            yield SentenceWindowDataPoint(current.text, current.entities, pre_buffer, post_buffer)
