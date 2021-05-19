from typing import Optional, Union

import torch
from torch import nn as nn

from pyramid_nested_ner.modules.word_embeddings.document_embeddings import DocumentRNNEmbeddings, \
    SentenceTransformerEmbeddings


class IdentityEncoder(nn.Module):
    def forward(self, x):
        return x


class LinearEncoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(LinearEncoder, self).__init__()
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.dense(x)


class ContextEncoder(nn.Module):
    embeddings: Union[DocumentRNNEmbeddings, SentenceTransformerEmbeddings]

    def __init__(
            self,
            use_pre=True,
            use_post=False,
            encoder_type: str = 'identity',
            encoder_output_size=64,
            device='cpu'
    ):
        super(ContextEncoder, self).__init__()

        self.use_pre = use_pre
        self.use_post = use_post

        self.directions = int(self.use_pre) + int(self.use_post)

        assert self.directions, "Neither 'use_pre' nor 'use_post' is set True!"

        self.device = device

        self._init_embeddings()

        self._init_encoder(encoder_type, encoder_output_size)

    def _init_embeddings(self):
        pass

    def _init_encoder(self, encoder_type, encoder_output_size):
        encoder_type = encoder_type.lower()
        assert encoder_type in ('identity', 'linear'), f"Invalid encoder_type '{encoder_type}'! " \
                                                       f"Must be 'linear' or 'identity'."

        if encoder_type == 'identity':
            self.encoder = IdentityEncoder()
            self.embedding_dim = self.embeddings.embedding_dim * self.directions
        else:
            self.encoder_hidden_size = self.embeddings.embedding_dim * self.directions
            self.encoder = LinearEncoder(self.encoder_hidden_size, encoder_output_size)
            self.encoder.to(self.device)
            self.embedding_dim = encoder_output_size

    def to(self, device, *args, **kwargs):
        self.embeddings.to(device, *args, **kwargs)
        self.encoder.to(device, *args, **kwargs)
        return super().to(device, *args, **kwargs)

    def train(self, mode: bool = True):
        self.embeddings.train(mode)
        self.encoder.train(mode)
        return super(ContextEncoder, self).train(mode)

    def eval(self):
        self.embeddings.eval()
        self.encoder.eval()
        return super(ContextEncoder, self).eval()

    def forward(
            self,
            pre_word_vectors,
            post_word_vectors
    ):
        if self.use_pre and self.use_post:
            output = torch.cat((
                self.embeddings(pre_word_vectors),
                self.embeddings(post_word_vectors)
            ), dim=-1)
        elif self.use_post:
            output = self.embeddings(post_word_vectors)
        else:
            output = self.embeddings(pre_word_vectors)

        return self.encoder(output)

class DocumentRNNEncoder(ContextEncoder):

    def __init__(
            self,
            word_embeddings,
            lexicon,
            padding_idx=0,
            hidden_size: int = 128,
            rnn_layers: int = 1,
            reproject_words: bool = True,
            reproject_words_dimension: Optional[int] = None,
            bidirectional: bool = False,
            dropout: float = 0.5,
            word_dropout: float = 0.0,
            locked_dropout: float = 0.0,
            device='cpu',
            casing=True,
            encoder_type: str = 'identity',
            encoder_output_size=64,
            use_pre=True,
            use_post=False
    ):
        self._embeddings_args = {
            'word_embeddings': word_embeddings,
            'lexicon': lexicon,
            'padding_idx': padding_idx,
            'hidden_size': hidden_size,
            'rnn_layers': rnn_layers,
            'reproject_words': reproject_words,
            'reproject_words_dimension': reproject_words_dimension,
            'bidirectional': bidirectional,
            'dropout': dropout,
            'word_dropout': word_dropout,
            'locked_dropout': locked_dropout,
            'casing': casing,
        }
        super(DocumentRNNEncoder, self).__init__(
            use_pre,
            use_post,
            encoder_type,
            encoder_output_size,
            device
        )

    def _init_embeddings(self):
        self.embeddings = DocumentRNNEmbeddings(
            self._embeddings_args['word_embeddings'],
            self._embeddings_args['lexicon'],
            padding_idx=self._embeddings_args['padding_idx'],
            hidden_size=self._embeddings_args['hidden_size'],
            rnn_layers=self._embeddings_args['rnn_layers'],
            reproject_words=self._embeddings_args['reproject_words'],
            reproject_words_dimension=self._embeddings_args['reproject_words_dimension'],
            bidirectional=self._embeddings_args['bidirectional'],
            dropout=self._embeddings_args['dropout'],
            word_dropout=self._embeddings_args['word_dropout'],
            locked_dropout=self._embeddings_args['locked_dropout'],
            casing=self._embeddings_args['casing'],
            device=self.device,
        )
        self.embeddings.to(self.device)


class SentenceTransformerEncoder(ContextEncoder):

    def __init__(
            self,
            lexicon,
            model: str = "paraphrase-distilroberta-base-v1",
            batch_size: int = 1,
            embedding_encoder_type='rnn',
            embedding_encoder_hidden_size=128,
            encoder_type: str = 'identity',
            encoder_output_size=64,
            padding_idx=0,
            casing=True,
            device='cpu',
            use_pre=True,
            use_post=False
    ):
        self._embeddings_args = {
            'lexicon': lexicon,
            'model': model,
            'batch_size': batch_size,
            'embedding_encoder_type': embedding_encoder_type,
            'embedding_encoder_hidden_size': embedding_encoder_hidden_size,
            'padding_idx': padding_idx,
            'casing': casing,
            'device': device,
        }
        super(SentenceTransformerEncoder, self).__init__(
            use_pre,
            use_post,
            encoder_type,
            encoder_output_size
        )

    def _init_embeddings(self):
        self.embeddings = SentenceTransformerEmbeddings(
            self._embeddings_args['lexicon'],
            model=self._embeddings_args['model'],
            batch_size=self._embeddings_args['batch_size'],
            embedding_encoder_type=self._embeddings_args['embedding_encoder_type'],
            padding_idx=self._embeddings_args['padding_idx'],
            casing=self._embeddings_args['casing'],
            device=self._embeddings_args['device'],
        )
        self.embeddings.to(self.device)
