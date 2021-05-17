from typing import Optional

import torch
from torch import nn as nn

from pyramid_nested_ner.modules.word_embeddings.document_embeddings import DocumentEmbeddings, DocumentRNNEmbeddings


class IdentityEncoder(nn.Module):
    def forward(self, x):
        return x


class LinearEncoder(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(LinearEncoder, self).__init__()
        self.dense = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        return self.dense(x)


class DocumentRNNEncoder(nn.Module):

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
        super(DocumentRNNEncoder, self).__init__()

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
            device=device,
            casing=casing
        )
        self.embeddings.to(device)

        self.use_pre = use_pre
        self.use_post = use_post
        self.directions = int(self.use_pre) + int(self.use_post)

        assert self.directions, "Neither 'use_pre' nor 'use_post' is set True!"

        encoder_type = encoder_type.lower()
        assert encoder_type in ('identity', 'linear'), f"Invalid encoder_type '{encoder_type}'! " \
                                                       f"Must be 'linear' or 'identity'."

        if encoder_type == 'identity':
            self.encoder = IdentityEncoder()
        else:
            encoder_hidden_size = self.embeddings.embedding_dim * self.directions
            self.encoder = LinearEncoder(encoder_hidden_size, encoder_output_size)

    def to(self, device, *args, **kwargs):
        self.embeddings.to(device, *args, **kwargs)
        self.encoder.to(device, *args, **kwargs)
        return super().to(device, *args, **kwargs)

    def train(self, mode: bool = True):
        self.embeddings.train(mode)
        self.encoder.train(mode)
        return super(DocumentRNNEncoder, self).train(mode)

    def eval(self):
        self.embeddings.eval()
        self.encoder.eval()
        return super(DocumentRNNEncoder, self).eval()

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
