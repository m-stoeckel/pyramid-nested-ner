from typing import List

from pyramid_nested_ner.data.contextualized import ContextWindowDataPoint
from pyramid_nested_ner.data.dataset import PyramidNerDataset
from pyramid_nested_ner.vectorizers.labels.multi_label_encoder import SigmoidMultiLabelEncoder


class SigmoidMultiLabelNerDataset(PyramidNerDataset):
    def _fit_label_encoder(self, entities_lexicon):
        self.label_encoder = SigmoidMultiLabelEncoder()
        self.label_encoder.set_tokenizer(self.tokenizer)
        if entities_lexicon is not None:
            self.label_encoder.fit(entities_lexicon)
        else:
            self.label_encoder.fit([e.name for x in self.data for e in x.entities])


class SentenceWindowMultiLabelNerDataset(SigmoidMultiLabelNerDataset):
    def _transform_x(self, sample: List[ContextWindowDataPoint]):
        data: dict = super(SentenceWindowMultiLabelNerDataset, self)._transform_x(sample)

        data['pre_word_vectors'] = [
            self.word_vectorizer.transform(data_point.preceding_data)
            for data_point in sample
        ]

        data['post_word_vectors'] = [
            self.word_vectorizer.transform(data_point.subsequent_data)
            for data_point in sample
        ]

        return data


class TokenWindowMultiLabelNerDataset(SigmoidMultiLabelNerDataset):
    def _transform_x(self, sample: List[ContextWindowDataPoint]):
        data: dict = super(TokenWindowMultiLabelNerDataset, self)._transform_x(sample)

        data['pre_word_vectors'], data['pre_word_masks'] = self.word_vectorizer.pad_sequences(
            self.word_vectorizer.transform(
                [data_point.preceding_data for data_point in sample]
            )
        )

        data['post_word_vectors'], data['post_word_masks'] = self.word_vectorizer.pad_sequences(
            self.word_vectorizer.transform(
                [data_point.subsequent_data for data_point in sample]
            )
        )

        return data
