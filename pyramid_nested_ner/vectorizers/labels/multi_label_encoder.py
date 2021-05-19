# encoding: utf-8
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence

from pyramid_nested_ner.vectorizers.labels import PyramidLabelEncoder


class SigmoidMultiLabelEncoder(PyramidLabelEncoder):

    def __init__(self):
        super(SigmoidMultiLabelEncoder, self).__init__()
        self.entity_array: np.array = np.empty(0, dtype=str)

    def fit(self, entities):
        self.entities = list(sorted({entity for entity in entities}))
        self.entity_array = np.array(self.entities, dtype=str)
        self.iob2_entities = [
            f'{iob2}-{entity}'
            for entity in self.entities
            for iob2 in 'BI' if entity
        ]

    def _transform_layer(self, data, layer):
        y_layer = []
        for x in (data if isinstance(data, list) else [data]):
            bitmaps = [bit for bit in self._entity_ngram_bitmap(x, layer)]
            if bitmaps:
                y_layer.append(torch.stack(bitmaps))
            else:
                y_layer.append(torch.zeros(0, len(self.entities)))

        # Note the missing .long() type conversion in comparison to the super method.
        # This is necessary as the BCEWithLogitsLoss requires float target tensors.
        return pad_sequence(y_layer, batch_first=True)

    def _entity_ngram_bitmap(self, data_point, order):
        for i, ngram in enumerate(self._ngrams(data_point.text, order)):
            ngram_start = i
            ngram_stop = i + len(ngram)

            bitmap = torch.zeros(len(self.entities))
            for entity in data_point.entities:
                entity_start = len(self.tokenize(data_point.text[:entity.start]))
                entity_stop = entity_start + len(self.tokenize(entity.value))
                if entity_start == ngram_start and entity_stop == ngram_stop:
                    bitmap[self.entities.index(entity.name)] = 1
                    # Continue, as a span can have many annotations

            yield bitmap

    # def _remedy_solution_bitmap(self, data_point, order):
    #     remedy_solution_bitmap = list()
    #     for i, ngram in enumerate(self._ngrams(data_point.text, order)):
    #         ngram_start = i
    #         ngram_stop = i + len(ngram)
    #         # Multi Label Bitmap: num(classes) × num(labels) = num(classes) × 2
    #         ngram_bitmap = torch.zeros((len(self.entities) - 1) * 2)
    #         for entity in data_point.entities:
    #             entity_start = len(self.tokenize(data_point.text[:entity.start]))
    #             entity_stop = entity_start + len(self.tokenize(entity.value))
    #             if ngram_start >= entity_start and ngram_stop <= entity_stop:
    #                 index = self.entities.index(entity.name)
    #                 # current n-gram is inside an entity span:
    #                 if entity_start == ngram_start:
    #                     ngram_bitmap[index * 2] = 1
    #                 elif ngram_stop <= entity_stop:
    #                     ngram_bitmap[index * 2 + 1] = 1
    #                 else:
    #                     raise AssertionError(" ")
    #         remedy_solution_bitmap.append(torch.flatten(ngram_bitmap.clone()))
    #     if remedy_solution_bitmap:
    #         return torch.stack(remedy_solution_bitmap)
    #     return torch.tensor([])
    #
    # def _remedy_encoding_transform(self, data, order):
    #     y_layer = list()
    #     for x in (data if isinstance(data, list) else [data]):
    #         y_layer.append(self._remedy_solution_bitmap(x, order))
    #     try:
    #         return pad_sequence(y_layer, batch_first=True)
    #     except RuntimeError:
    #         # pad_sequence can crash if some sequences in `data` are shorter than the number of lay-
    #         # ers and therefore their encoding yields an empty tensor, while other sequences are tr-
    #         # ansformed into tensors of shape (n, |entities| * 2).
    #         y_layer = [y if y.numel() else torch.zeros(1, (len(self.entities) - 1) * 2) for y in y_layer]
    #         return pad_sequence(y_layer, batch_first=True)

    def _inverse_layer_transform(self, y_layer):
        return [
            [
                self.entity_array[indicators.cpu() == 1].tolist()
                for indicators in sequence
            ]
            for sequence in y_layer
        ]

    def inverse_remedy_transform(self, y_remedy):
        longest_span = 0
        sequences_tags = []
        for sequence in y_remedy:
            # Keep a list for each entity class for both entities that are already finished, and those entities
            # that may be continued with the next token
            sequence_entities = defaultdict(list)
            current_entities = defaultdict(list)
            previous_entities = np.full_like(sequence[0].cpu().numpy().reshape(-1, 2)[:, 0], False, dtype=np.bool)

            # As the modified remedy solution predicts multiple labels per token, sequences can both begin and continue
            # on any given token. As such, all entities, that have begone on a previous token, and have not ended yet,
            # are updated each round. This may lead to misclassifications with partially overlapping entities.
            # However, the NNE dataset does not contain any partially overlapping annotations.
            for offset, logits in enumerate(sequence):
                np_logits = logits.cpu().numpy().reshape(-1, 2)
                begin_entities = np_logits[:, 0] == 1

                # Inside tag calculation
                # An inside tag requires a preceding begin tag -> multiplication with previous_entities.
                inside_entities = (np_logits[:, 1] == 1) * previous_entities

                # Add each beginning entity to its respective current list.
                begin_entities_list = self.entity_array[begin_entities].tolist()
                for begin_entity in begin_entities_list:
                    current_entities[begin_entity].append([offset, offset + 1])
                    longest_span = max(longest_span, 1)

                # Extend all continued entities by increasing their span-end offset.
                inside_entities_list = self.entity_array[inside_entities].tolist()
                for inside_entity in inside_entities_list:
                    for entity in current_entities[inside_entity]:
                        entity[1] = offset + 1
                        longest_span = max(longest_span, abs(entity[1] - entity[0]))

                previous_entities = begin_entities | inside_entities

                # Push all entities, that neither started nor continued on this token, to the sequence_entities lists
                for missing_entity in self.entity_array[np.logical_not(previous_entities)].tolist():
                    sequence_entities[missing_entity].extend(current_entities[missing_entity])
                    current_entities[missing_entity] = []

            # Push remaining entities to sequence_entities lists
            for key, value in current_entities.items():
                sequence_entities[key].extend(value)

            sequence_tags = {}
            for entity_name, entities in sequence_entities.items():
                for start, end in entities:
                    length = end - start
                    if length not in sequence_tags:
                        sequence_tags[length] = [[] for _ in range(len(sequence) - (length - 1))]
                    sequence_tags[length][start].append(entity_name)

            sequences_tags.append(sequence_tags)

        return self._decode_labels(y_remedy, sequences_tags, longest_span)

    @staticmethod
    def _decode_labels(y_remedy, sequences_tags, longest_span):
        decoded_labels = []
        for i in range(1, longest_span + 1):
            decoded_labels_for_order = []
            for sequence, sequence_tags in zip(y_remedy, sequences_tags):
                sequence_length = max(0, len(sequence) - (i - 1))
                if i in sequence_tags:
                    span = [iob2_tag or [] for iob2_tag in sequence_tags[i]]
                else:
                    span = [[] for _ in range(sequence_length)]
                decoded_labels_for_order.append(span)
            decoded_labels.append(decoded_labels_for_order)
        return decoded_labels
