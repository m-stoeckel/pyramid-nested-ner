import enum
from typing import Optional

from pyramid_nested_ner.model import PyramidNer
from pyramid_nested_ner.modules.decoding.multi_label import ContextualOneVsRestDecoder, ContextualSigmoidLinearDecoder, \
    OneVsRestDecoder, SigmoidLinearDecoder
from pyramid_nested_ner.modules.encoding.contextual_encoder import DocumentRNNEncoder
from pyramid_nested_ner.vectorizers.labels.multi_label_encoder import SigmoidMultiLabelEncoder


class SigmoidMultiLabelPyramid(PyramidNer):
    class ClassifierType(enum.Enum):
        linear = SigmoidLinearDecoder
        one_vs_rest = OneVsRestDecoder

    def __init__(self, *args, classifier_type='linear', **kwargs):
        self.classifier_type = classifier_type
        super(SigmoidMultiLabelPyramid, self).__init__(*args, **kwargs)

    def _initialize_label_encoder(self, entities_lexicon):
        self.label_encoder = SigmoidMultiLabelEncoder()
        self.label_encoder.fit(entities_lexicon)

    def _init_linear_decoder(self, decoder_output_size):
        classifier = self.ClassifierType[self.classifier_type].value(
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


class DocumentRNNSentenceWindowPyramid(SigmoidMultiLabelPyramid):
    class ClassifierType(enum.Enum):
        linear = ContextualSigmoidLinearDecoder
        one_vs_rest = ContextualOneVsRestDecoder

    class _Model(SigmoidMultiLabelPyramid._Model):
        def __init__(self, sentence_encoder, pyramid, context_encoder, classifier):
            super(DocumentRNNSentenceWindowPyramid._Model, self).__init__(
                sentence_encoder,
                pyramid,
                classifier
            )
            self.context_encoder = context_encoder

        def forward(self, *args, **kwargs):
            x, mask = self.encoder(
                kwargs['word_vectors'],
                kwargs['word_mask'],
                kwargs.get('char_vectors'),
                kwargs.get('char_mask')
            )
            h, h_remedy = self.pyramid(x, mask)
            x_context = self.context_encoder(
                kwargs.get('pre_word_vectors'),
                kwargs.get('post_word_vectors')
            )
            return self.classifier(h, x_context, h_remedy)

        def to(self, device, *args, **kwargs):
            self.context_encoder.to(device, *args, **kwargs)
            super(DocumentRNNSentenceWindowPyramid._Model, self).to(device, *args, **kwargs)

    def __init__(
            self,
            word_lexicon,
            word_embeddings,
            entities_lexicon,
            use_pre=True,
            use_post=False,
            padding_idx=0,
            hidden_size: int = 128,
            rnn_layers: int = 1,
            reproject_words: bool = True,
            reproject_words_dimension: Optional[int] = None,
            bidirectional: bool = False,
            dropout: float = 0.5,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.0,
            casing=True,
            **kwargs
    ):
        self._context_model_args = {
            'word_embeddings': [word_embeddings] if isinstance(word_embeddings, str) else word_embeddings,
            'word_lexicon': word_lexicon,
            'use_pre': use_pre,
            'use_post': use_post,
            'padding_idx': padding_idx,
            'hidden_size': hidden_size,
            'rnn_layers': rnn_layers,
            'reproject_words': reproject_words,
            'reproject_words_dimension': reproject_words_dimension,
            'bidirectional': bidirectional,
            'dropout': dropout,
            'word_dropout': word_dropout,
            'locked_dropout': locked_dropout,
            'casing': casing
        }

        super(DocumentRNNSentenceWindowPyramid, self).__init__(
            word_lexicon,
            word_embeddings,
            entities_lexicon,
            **kwargs
        )

    def _init_nnet(self):
        sentence_encoder = self._init_sentence_encoder()
        pyramid_decoder = self._init_pyramid_decoder()

        context_encoder = self._init_context_encoder()

        decoder_output_size = self._model_args['decoder_hidden_size'] * 2 * (
                1 + int(self._model_args['inverse_pyramid']))
        # TODO: account for contextual representation dim
        decoder_output_size += context_encoder.embeddings.embedding_dim * context_encoder.directions

        classifier = self._init_linear_decoder(decoder_output_size)
        model = self._Model(sentence_encoder, pyramid_decoder, context_encoder, classifier)
        model.to(self.device)
        return model

    def _init_context_encoder(self):
        context_encoder = DocumentRNNEncoder(
            self._context_model_args['word_embeddings'],
            self._context_model_args['word_lexicon'],
            padding_idx=self._context_model_args['padding_idx'],
            hidden_size=self._context_model_args['hidden_size'],
            rnn_layers=self._context_model_args['rnn_layers'],
            reproject_words=self._context_model_args['reproject_words'],
            reproject_words_dimension=self._context_model_args['reproject_words_dimension'],
            bidirectional=self._context_model_args['bidirectional'],
            dropout=self._context_model_args['dropout'],
            word_dropout=self._context_model_args['word_dropout'],
            locked_dropout=self._context_model_args['locked_dropout'],
            device=self.device,
            casing=self._context_model_args['casing'],
            use_pre=self._context_model_args['use_pre'],
            use_post=self._context_model_args['use_post']
        )
        context_encoder.to(self.device)
        return context_encoder

    def _init_linear_decoder(self, decoder_output_size):
        classifier = self.ClassifierType[self.classifier_type].value(
            decoder_output_size,
            classes=len(self.label_encoder.entities)
        )
        classifier.to(self.device)
        return classifier
