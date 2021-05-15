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
