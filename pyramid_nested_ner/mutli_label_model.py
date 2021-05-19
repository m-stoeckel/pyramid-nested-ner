from enum import Enum
from typing import Optional

from pyramid_nested_ner.model import PyramidNer
from pyramid_nested_ner.modules.decoding.multi_label import ContextualOneVsRestConvDecoder, \
    ContextualOneVsRestMultiHeadDecoder, ContextualSigmoidLinearDecoder, \
    OneVsRestConvDecoder, OneVsRestMultiHeadDecoder, SigmoidLinearDecoder
from pyramid_nested_ner.modules.encoding.contextual_encoder import ContextEncoder, DocumentRNNEncoder, \
    SentenceTransformerEncoder
from pyramid_nested_ner.vectorizers.labels.multi_label_encoder import SigmoidMultiLabelEncoder


class SigmoidMultiLabelPyramid(PyramidNer):
    class ClassifierType(Enum):
        linear = SigmoidLinearDecoder
        ovr_conv = OneVsRestConvDecoder
        ovr_multihead = OneVsRestMultiHeadDecoder

    def __init__(self, *args, classifier_type='linear', **kwargs):
        self.classifier_cls = self.ClassifierType[classifier_type].value
        super(SigmoidMultiLabelPyramid, self).__init__(*args, **kwargs)

    def _initialize_label_encoder(self, entities_lexicon):
        self.label_encoder = SigmoidMultiLabelEncoder()
        self.label_encoder.fit(entities_lexicon)

    def _init_linear_decoder(self, decoder_output_size):
        classifier = self.classifier_cls(
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


class ContextualMultiLabelPyramid(SigmoidMultiLabelPyramid):
    class ClassifierType(Enum):
        linear = ContextualSigmoidLinearDecoder
        ovr_conv = ContextualOneVsRestConvDecoder
        ovr_multihead = ContextualOneVsRestMultiHeadDecoder

    class _Model(SigmoidMultiLabelPyramid._Model):
        def __init__(self, sentence_encoder, pyramid, context_encoder, classifier):
            super(ContextualMultiLabelPyramid._Model, self).__init__(
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
            super(ContextualMultiLabelPyramid._Model, self).to(device, *args, **kwargs)

    def _init_nnet(self):
        sentence_encoder = self._init_sentence_encoder()
        pyramid_decoder = self._init_pyramid_decoder()

        context_encoder = self._init_context_encoder()

        decoder_output_size = self._model_args['decoder_hidden_size'] * 2 * (
                1 + int(self._model_args['inverse_pyramid']))
        decoder_output_size += context_encoder.embedding_dim

        classifier = self._init_linear_decoder(decoder_output_size)
        model = self._Model(sentence_encoder, pyramid_decoder, context_encoder, classifier)
        model.to(self.device)
        return model

    def _init_linear_decoder(self, decoder_output_size):
        classifier = self.classifier_cls(
            decoder_output_size,
            classes=len(self.label_encoder.entities)
        )
        classifier.to(self.device)
        return classifier

    def _init_context_encoder(self) -> ContextEncoder:
        pass


class DocumentRNNSentenceWindowPyramid(ContextualMultiLabelPyramid):

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


class SentenceTransformerPyramid(ContextualMultiLabelPyramid):
    def __init__(
            self,
            word_lexicon,
            word_embeddings,
            entities_lexicon,
            model: str = "paraphrase-distilroberta-base-v1",
            batch_size: int = 1,
            embedding_encoder_type='rnn',
            embedding_encoder_hidden_size=128,
            encoder_type: str = 'identity',
            transformer_encoder_output_size=64,
            padding_idx=0,
            casing=True,
            use_pre=True,
            use_post=False,
            **kwargs
    ):
        self._context_model_args = {
            'word_lexicon': word_lexicon,
            'model': model,
            'batch_size': batch_size,
            'embedding_encoder_type': embedding_encoder_type,
            'embedding_encoder_hidden_size': embedding_encoder_hidden_size,
            'encoder_type': encoder_type,
            'transformer_encoder_output_size': transformer_encoder_output_size,
            'use_pre': use_pre,
            'use_post': use_post,
            'padding_idx': padding_idx,
            'casing': casing
        }
        super(SentenceTransformerPyramid, self).__init__(
            word_lexicon,
            word_embeddings,
            entities_lexicon,
            **kwargs
        )

    def _init_context_encoder(self) -> ContextEncoder:
        return SentenceTransformerEncoder(
            self._context_model_args['word_lexicon'],
            model=self._context_model_args['model'],
            batch_size=self._context_model_args['batch_size'],
            embedding_encoder_type=self._context_model_args['embedding_encoder_type'],
            embedding_encoder_hidden_size=self._context_model_args['embedding_encoder_hidden_size'],
            encoder_type=self._context_model_args['encoder_type'],
            encoder_output_size=self._context_model_args['transformer_encoder_output_size'],
            padding_idx=self._context_model_args['padding_idx'],
            casing=self._context_model_args['casing'],
            use_pre=self._context_model_args['use_pre'],
            use_post=self._context_model_args['use_post'],
            device=self.device
        )
