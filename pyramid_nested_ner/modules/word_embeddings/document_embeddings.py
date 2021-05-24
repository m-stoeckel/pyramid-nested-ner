from typing import List, Optional, Union

import flair
import flair.embeddings
import torch
import torch.nn as nn
import torch.nn.functional as F
from flair.embeddings import WordEmbeddings
from torch.nn.utils.rnn import pad_sequence

from pyramid_nested_ner.modules.rnn import FastRNN


class DocumentEmbeddings(nn.Module):
    embeddings: Union[
        flair.embeddings.DocumentRNNEmbeddings,
        flair.embeddings.TransformerDocumentEmbeddings,
        flair.embeddings.SentenceTransformerDocumentEmbeddings
    ]

    def __init__(
            self,
            lexicon,
            padding_idx=0,
            casing=True,
            device='cpu',
            requires_grad=False
    ):
        super(DocumentEmbeddings, self).__init__()
        self.lexicon = lexicon
        self.pad_index = padding_idx
        self.casing = casing
        self.device = device
        self.requires_grad = requires_grad

        self.vocab_idx = {i: token.lower() if not self.casing else token for i, token in enumerate(self.lexicon)}

        self._cache = {}
        self._train = False

    def to(self, device, *args, **kwargs):
        self.device = device
        self.embeddings.to(device, *args, **kwargs)
        return self

    def train(self, mode=True):
        self._train = mode
        self.embeddings.train(mode)

    def eval(self):
        self._train = False
        self.embeddings.train(False)

    def _generate_cache_key(self, index_sequence):
        return " ".join(str(index) for index in index_sequence)

    def _generate_index_sequence_embedding(self, index_sequence):
        flair_sentence = flair.data.Sentence()
        for index in index_sequence:
            token = self.vocab_idx.get(index, '[UNK]')
            flair_sentence.add_token(token)

        if not len(flair_sentence.tokens):
            return torch.zeros(self.embedding_dim, requires_grad=self.requires_grad)

        self.embeddings.embed(flair_sentence)
        return flair_sentence.get_embedding()

    def _get_index_sequence_embedding(self, index_sequence):
        cache_key = self._generate_cache_key(index_sequence)
        if self._cache.get(cache_key) is None:
            sentence_embedding = self._generate_index_sequence_embedding(index_sequence)
            self._cache[cache_key] = sentence_embedding
        else:
            sentence_embedding = self._cache.get(cache_key).clone().detach().requires_grad_(self.requires_grad)
        return sentence_embedding

    def _embed_list_of_sequences(self, batch: List[List[torch.Tensor]]):
        embeddings = []
        for sequences in batch:
            index_sequence = [
                index.item()
                for sequence in sequences
                for index in sequence.cpu().clone().detach()
                if index.item() != self.pad_index
            ]

            sentence_embedding = self._get_index_sequence_embedding(index_sequence)

            embeddings.append(sentence_embedding.to(self.device))
        return torch.stack(embeddings).to(self.device)

    def _embed_list_of_tensors(self, batch: List[torch.Tensor]):
        embeddings = []
        for sequence in batch:
            index_sequence = [
                index.item()
                for index in sequence.cpu().clone().detach()
                if index.item() != self.pad_index
            ]
            sentence_embedding = self._get_index_sequence_embedding(index_sequence)

            embeddings.append(sentence_embedding.to(self.device))
        return torch.stack(embeddings).to(self.device)


class DocumentRNNEmbeddings(DocumentEmbeddings):
    """
    Fake-module to build word embeddings from transformers' token embeddings.
    Any model from huggingface transformers library can be loaded (using its
    name or path to model weights), if it's supported by Flair. Keep in mind
    that this class is extremely slow and should only be used for research
    experiments. It uses a cache to speed up all training epochs following
    the first one, but inference time on unseen samples remains prohibitive.
    """  # FIXME: Change docstring

    def __init__(
            self,
            embeddings,
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
    ):
        super(DocumentRNNEmbeddings, self).__init__(
            lexicon=lexicon,
            padding_idx=padding_idx,
            casing=casing,
            device=device,
            requires_grad=True
        )
        embeddings = [embeddings] if isinstance(embeddings, str) else embeddings
        embeddings = [WordEmbeddings(emb) for emb in embeddings]
        for embedding in embeddings:
            embedding.to(device)

        self.embeddings = flair.embeddings.DocumentRNNEmbeddings(
            embeddings,
            hidden_size=hidden_size,
            rnn_layers=rnn_layers,
            reproject_words=reproject_words,
            reproject_words_dimension=reproject_words_dimension,
            bidirectional=bidirectional,
            dropout=dropout,
            word_dropout=word_dropout,
            locked_dropout=locked_dropout
        )
        self.embedding_dim: int = self.embeddings.embedding_length
        self.embeddings.to(device)

    def _get_index_sequence_embedding(self, index_sequence):
        return self._generate_index_sequence_embedding(index_sequence)

    def forward(self, batch: Union[List[torch.Tensor], List[List[torch.Tensor]]]):
        if isinstance(batch[0], list):
            return self._embed_list_of_sequences(batch)
        else:
            return self._embed_list_of_tensors(batch)


class MaxPooler(nn.Module):
    def forward(self, x: torch.Tensor):
        x, _ = torch.max(x, dim=0)
        return x


class MinPooler(nn.Module):
    def forward(self, x: torch.Tensor):
        x, _ = torch.min(x, dim=0)
        return x


class MeanPooler(nn.Module):
    def forward(self, x: torch.Tensor):
        x = torch.mean(x, dim=0)
        return x


class PooledSentenceTransformerEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            lexicon,
            dropout=0.5,
            model: str = "paraphrase-distilroberta-base-v1",
            batch_size: int = 1,
            embedding_pooling_method='mean',
            padding_idx=0,
            casing=True,
            device='cpu',
    ):
        super(PooledSentenceTransformerEmbeddings, self).__init__(
            lexicon=lexicon,
            padding_idx=padding_idx,
            casing=casing,
            device=device
        )
        self.embeddings = flair.embeddings.SentenceTransformerDocumentEmbeddings(
            model=model,
            batch_size=batch_size
        )
        self.dropout = dropout or 0.0

        if embedding_pooling_method == 'max':
            self.encoder = MaxPooler()
        elif embedding_pooling_method == 'min':
            self.encoder = MinPooler()
        else:
            self.encoder = MeanPooler()

        self.embedding_dim = self.embeddings.embedding_length

    def _embed_list_of_sequences(self, batch: List[List[torch.Tensor]]):
        embeddings = []
        for sequences in batch:
            sequence_embeddings = []
            for sequence in sequences:
                index_sequence = [
                    index.item()
                    for index in sequence.cpu().clone().detach()
                    if index.item() != self.pad_index
                ]

                sentence_embedding = self._get_index_sequence_embedding(index_sequence).to(self.device)
                sequence_embeddings.append(sentence_embedding)

            if sequence_embeddings:
                sequence_embeddings = self.encoder(torch.stack(sequence_embeddings, 0))
            else:
                sequence_embeddings = torch.zeros(self.embedding_dim, requires_grad=False, device=self.device)

            embeddings.append(sequence_embeddings)
        return torch.stack(embeddings).to(self.device)

    def forward(self, x: Union[torch.Tensor, List[torch.Tensor], List[List[torch.Tensor]]]):
        if isinstance(x[0], list):
            embeddings = self._embed_list_of_sequences(x)
        else:
            embeddings = self._embed_list_of_tensors(x)

        return F.dropout(embeddings, self.dropout, training=self._train)


class RNNSentenceTransformerEmbeddings(PooledSentenceTransformerEmbeddings):
    def __init__(
            self,
            lexicon,
            dropout=0.5,
            model: str = "paraphrase-distilroberta-base-v1",
            batch_size: int = 1,
            rnn_style='lstm',
            rnn_hidden_size=128,
            embedding_pooling_method='linear',
            embedding_encoder_hidden_size=128,
            padding_idx=0,
            casing=True,
            device='cpu',
    ):
        super(RNNSentenceTransformerEmbeddings, self).__init__(
            lexicon=lexicon,
            dropout=dropout,
            padding_idx=padding_idx,
            casing=casing,
            device=device
        )
        self.embeddings = flair.embeddings.SentenceTransformerDocumentEmbeddings(
            model=model,
            batch_size=batch_size
        )
        self.rnn = FastRNN(
            nn.LSTM if rnn_style == 'lstm' else nn.GRU,
            input_size=self.embeddings.embedding_length,
            hidden_size=rnn_hidden_size,
            batch_first=True
        )
        self.embedding_dim = rnn_hidden_size

        self.embedding_encoder_type = embedding_pooling_method
        self.embedding_encoder_hidden_size = embedding_encoder_hidden_size
        if embedding_pooling_method == 'max':
            self.encoder = MaxPooler()
        elif embedding_pooling_method == 'min':
            self.encoder = MinPooler()
        elif embedding_pooling_method == 'mean':
            self.encoder = MeanPooler()
        elif embedding_pooling_method == 'linear':
            self.encoder = nn.Linear(self.rnn.hidden_size, embedding_encoder_hidden_size)
            self.embedding_dim = embedding_encoder_hidden_size

    def _embed_list_of_sequences(self, batch: List[List[torch.Tensor]]) -> List[torch.Tensor]:
        embeddings = []
        for sequences in batch:
            sequence_embeddings = []
            for sequence in sequences:
                index_sequence = [
                    index.item()
                    for index in sequence.cpu().clone().detach()
                    if index.item() != self.pad_index
                ]

                sentence_embedding = self._get_index_sequence_embedding(index_sequence)
                sequence_embeddings.append(sentence_embedding.to(self.device))
            embeddings.append(torch.stack(sequence_embeddings))
        return embeddings

    def forward(self, x: List[List[torch.Tensor]]):
        # Variable length sentence embeddings as a list of tensors
        embeddings: List[torch.Tensor] = self._embed_list_of_sequences(x)

        mask = pad_sequence(
            [torch.ones(sequence.size(0), dtype=torch.int8) for sequence in embeddings],
            batch_first=True
        )
        embeddings: torch.Tensor = pad_sequence(embeddings, batch_first=True).to(self.device)
        h, (hn, _) = self.rnn(embeddings, mask)

        if self.embedding_encoder_type == 'hidden':
            embeddings = hn.squeeze(0)
        elif self.embedding_encoder_type == 'linear':
            embeddings: torch.Tensor = self.encoder(hn.squeeze(0))
        else:
            embeddings: torch.Tensor = self.encoder(h)

        return F.dropout(embeddings, self.dropout, training=self._train)

    def to(self, device, *args, **kwargs):
        if self.embedding_encoder_type == 'rnn':
            self.encoder.to(device, *args, **kwargs)
        return self
