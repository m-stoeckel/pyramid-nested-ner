# encoding: utf-8
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
        self.entity_array = np.array([f'B-{entity}' for entity in self.entities], dtype=str)
        self.iob2_entities = [f'{iob2}-{entity}' for entity in self.entities for iob2 in 'IB' if entity]

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
        sequences_tags = []
        for sequence in y_layer:
            tags = []
            for indicators in sequence:
                arr = self.entity_array.copy()
                arr[indicators.cpu() != 1] = 'O'
                tags.extend(list(arr))
            sequences_tags.append(tags)
        return sequences_tags

    # FIXME: This does not work with multi-label annotations
    def inverse_remedy_transform(self, y_remedy):

        def _recover_span(tensor_slice, entity_name):
            for j, vect in enumerate(tensor_slice[1:]):
                if not vect[self.iob2_entities.index(f'I-{entity_name}')]:
                    return tensor_slice[:j + 1]
            return tensor_slice

        longest_span, sequences_tags = 0, list()

        for sequence in y_remedy:
            sequence_tags = dict()
            for offset, logits in enumerate(sequence):
                for entity in self.entities:
                    if logits[self.iob2_entities.index(f'B-{entity}')]:
                        span = _recover_span(sequence[offset:], entity)
                        if len(span) not in sequence_tags:
                            sequence_tags[len(span)] = ['O' for _ in range(len(sequence) - (len(span) - 1))]
                        if 'O' == sequence_tags[len(span)][offset]:
                            sequence_tags[len(span)][offset] = f'B-{entity}'
                            longest_span = max(len(span), longest_span)
                        else:
                            sequence_tags[len(span)][offset] = None
                            # print(
                            #   f"Tokens {span} have two different annotations: "
                            #   f"{sequence_tags[len(span)][2:]}, and {entity}. "
                            #   f"Due to this conflict, both annotations will be"
                            #   f" discarded."
                            # )
            sequences_tags.append(sequence_tags)

        decoded_labels = list()
        for i in range(1, longest_span + 1):
            decoded_labels_for_order = list()
            for sequence, sequence_tags in zip(y_remedy, sequences_tags):
                sequence_length = max(0, len(sequence) - (i - 1))
                if i in sequence_tags:
                    span = [iob2_tag or 'O' for iob2_tag in sequence_tags[i]]
                else:
                    span = ['O' for _ in range(sequence_length)]
                decoded_labels_for_order.append(span)
            decoded_labels.append(decoded_labels_for_order)

        return decoded_labels
