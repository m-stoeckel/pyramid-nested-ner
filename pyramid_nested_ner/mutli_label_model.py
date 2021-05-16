from pyramid_nested_ner.model import PyramidNer
from pyramid_nested_ner.modules.decoding.linear import SigmoidMultiLabelLinearDecoder
from pyramid_nested_ner.vectorizers.labels.multi_label_encoder import SigmoidMultiLabelEncoder


class SigmoidMultiLabelPyramidNer(PyramidNer):

    def _initialize_label_encoder(self, entities_lexicon):
        self.label_encoder = SigmoidMultiLabelEncoder()
        self.label_encoder.fit(entities_lexicon)

    def _init_linear_decoder(self, decoder_output_size):
        classifier = SigmoidMultiLabelLinearDecoder(
            decoder_output_size,
            classes=len(self.label_encoder.entities)
        )
        classifier.to(self.device)
        return classifier

    def logits_to_classes(self, logits):
        """
        The remedy solution already utilizes a Sigmoid classification layer
        """
        return [self.remedy_to_classes(logit) for logit in logits]
