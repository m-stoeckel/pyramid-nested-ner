from typing import Optional, Union

import torch
from torch import nn as nn

from pyramid_nested_ner.modules.word_embeddings.document_embeddings import DocumentRNNEmbeddings, \
    PooledSentenceTransformerEmbeddings


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
    embeddings: Union[DocumentRNNEmbeddings, PooledSentenceTransformerEmbeddings]

    def __init__(
            self,
            use_pre=True,
            use_post=False,
            device='cpu'
    ):
        super(ContextEncoder, self).__init__()

        self.use_pre = use_pre
        self.use_post = use_post

        self.directions = int(self.use_pre) + int(self.use_post)

        assert self.directions, "Neither 'use_pre' nor 'use_post' is set True!"

        self.device = device

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

        self._train = False

    def to(self, device, *args, **kwargs):
        self.embeddings.to(device, *args, **kwargs)
        self.encoder.to(device, *args, **kwargs)
        return super().to(device, *args, **kwargs)

    def train(self, mode: bool = True):
        self.embeddings.train(mode)
        self.encoder.train(mode)
        self._train = mode
        return super(ContextEncoder, self).train(mode)

    def eval(self):
        self.embeddings.eval()
        self.encoder.eval()
        self._train = False
        return super(ContextEncoder, self).eval()

    def forward(
            self,
            pre_word_vectors,
            post_word_vectors,
            pre_word_masks=None,
            post_word_masks=None
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
            embedding_encoder_type='linear',
            embedding_encoder_output_size=128,
            use_pre=True,
            use_post=False
    ):
        super(DocumentRNNEncoder, self).__init__(
            use_pre,
            use_post,
            device
        )

        self.embeddings = DocumentRNNEmbeddings(
            word_embeddings,
            lexicon,
            padding_idx=padding_idx,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            reproject_words=reproject_words,
            reproject_words_dimension=reproject_words_dimension,
            bidirectional=bidirectional,
            dropout=dropout,
            word_dropout=word_dropout,
            locked_dropout=locked_dropout,
            casing=casing,
            device=self.device,
        )
        self.embeddings.to(self.device)

        self._init_encoder(embedding_encoder_type, embedding_encoder_output_size)


class SentenceTransformerEncoder(ContextEncoder):

    def __init__(
            self,
            lexicon,
            model: str = "paraphrase-distilroberta-base-v1",
            batch_size: int = 1,
            embedding_pooling_method='mean',
            embedding_encoder_type='linear',
            embedding_encoder_output_size=128,
            padding_idx=0,
            casing=True,
            device='cpu',
            use_pre=True,
            use_post=False
    ):
        super(SentenceTransformerEncoder, self).__init__(
            use_pre,
            use_post,
            device
        )

        self.embeddings = PooledSentenceTransformerEmbeddings(
            lexicon,
            model=model,
            batch_size=batch_size,
            embedding_pooling_method=embedding_pooling_method,
            padding_idx=padding_idx,
            casing=casing,
            device=self.device,
        )
        self.embeddings.to(self.device)

        self._init_encoder(embedding_encoder_type, embedding_encoder_output_size)
