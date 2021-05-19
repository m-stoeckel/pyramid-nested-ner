from typing import Optional, Union

import flair
import flair.embeddings
import torch
import torch.nn as nn
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

    ):
        super(DocumentEmbeddings, self).__init__()
        self.lexicon = lexicon
        self.pad_index = padding_idx
        self.casing = casing
        self.device = device

        self._cache = {}

    def to(self, device, *args, **kwargs):
        self.device = device
        self.embeddings.to(device, *args, **kwargs)
        return self

    def train(self, mode=True):
        self.embeddings.train(mode)

    def eval(self):
        self.embeddings.train(False)

    @property
    def vocab_idx(self):
        return {i: token.lower() if not self.casing else token for i, token in enumerate(self.lexicon)}

    def _tensor_to_cache_key(self, ltensor):
        cache_key = " ".join(
            [str(value.item()) for tensor in ltensor for value in tensor.cpu().clone().detach()]
        )
        return cache_key

    def _get_from_cache(self, key):
        key = self._tensor_to_cache_key(key)
        return self._cache.get(key)

    def _add_to_cache(self, key, value):
        key = self._tensor_to_cache_key(key)
        self._cache[key] = value.cpu()


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
        self.embedding_dim = self.embeddings.embedding_length
        self.embeddings.to(device)

    def forward(self, x: torch.Tensor):
        vocab_idx = self.vocab_idx
        embeddings = []
        for sequences in x:
            if self._get_from_cache(sequences) is None:
                flair_sentence = flair.data.Sentence()
                for sequence in sequences:
                    for index in sequence:
                        index = index.item()
                        if index == self.pad_index:
                            break  # skip padding
                        token = vocab_idx.get(index, '[UNK]')
                        flair_sentence.add_token(token)
                if len(flair_sentence.tokens):
                    self.embeddings.embed(flair_sentence)
                    sentence_embedding = flair_sentence.get_embedding()
                else:
                    sentence_embedding = torch.zeros(self.embedding_dim, requires_grad=True)
                self._add_to_cache(sequences, sentence_embedding)
            else:
                sentence_embedding = self._get_from_cache(sequences).clone().detach().requires_grad_(True)
            sentence_embedding = sentence_embedding.to(self.device)
            embeddings.append(sentence_embedding)

        return torch.stack(embeddings).to(self.device)


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


class SentenceTransformerEmbeddings(DocumentEmbeddings):
    def __init__(
            self,
            lexicon,
            model: str = "paraphrase-distilroberta-base-v1",
            batch_size: int = 1,
            embedding_encoder_type='mean',
            padding_idx=0,
            casing=True,
            device='cpu',
    ):
        super(SentenceTransformerEmbeddings, self).__init__(
            lexicon=lexicon,
            padding_idx=padding_idx,
            casing=casing,
            device=device
        )
        self.embeddings = flair.embeddings.SentenceTransformerDocumentEmbeddings(
            model=model,
            batch_size=batch_size
        )

        self.embedding_encoder_type = embedding_encoder_type
        if embedding_encoder_type == 'max':
            self.encoder = MaxPooler()
        elif embedding_encoder_type == 'min':
            self.encoder = MinPooler()
        else:
            self.encoder = MeanPooler()

        self.embedding_dim = self.embeddings.embedding_length

    def _tensor_to_cache_key(self, tensor):
        cache_key = " ".join(
            [str(value.item()) for value in tensor.cpu().clone().detach()]
        )
        return cache_key

    def to(self, device, *args, **kwargs):
        if self.embedding_encoder_type == 'rnn':
            self.encoder.to(device, *args, **kwargs)
        return self

    def _embed(self, x):
        vocab_idx = self.vocab_idx
        embeddings = []
        for sequences in x:
            sequence_embeddings = []
            for sequence in sequences:
                if self._get_from_cache(sequence) is None:
                    flair_sentence = flair.data.Sentence()
                    for index in sequence:
                        index = index.item()
                        if index == self.pad_index:
                            break  # skip padding
                        token = vocab_idx.get(index, '[UNK]')
                        flair_sentence.add_token(token)
                    if len(flair_sentence.tokens):
                        self.embeddings.embed(flair_sentence)
                        sentence_embedding = flair_sentence.get_embedding()
                    else:
                        sentence_embedding = torch.zeros(self.embeddings.embedding_length, requires_grad=False)
                    self._add_to_cache(sequence, sentence_embedding)
                else:
                    sentence_embedding = self._get_from_cache(sequence).clone().detach()
                sentence_embedding = sentence_embedding.to(self.device)
                sequence_embeddings.append(sentence_embedding)

            if sequence_embeddings:
                sequence_embeddings = self.encoder(torch.stack(sequence_embeddings, 0))
            else:
                sequence_embeddings = torch.zeros(self.embedding_dim, requires_grad=False)

            embeddings.append(sequence_embeddings)
        return torch.stack(embeddings).to(self.device)

    def forward(self, x: torch.Tensor):
        return self._embed(x)


class SentenceRNNTransformerEmbeddings(SentenceTransformerEmbeddings):
    def __init__(
            self,
            lexicon,
            model: str = "paraphrase-distilroberta-base-v1",
            batch_size: int = 1,
            rnn_style='lstm',
            rnn_hidden_size=128,
            embedding_encoder_type='linear',
            embedding_encoder_hidden_size=128,
            padding_idx=0,
            casing=True,
            device='cpu',
    ):
        super(SentenceRNNTransformerEmbeddings, self).__init__(
            lexicon=lexicon,
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

        self.embedding_encoder_type = embedding_encoder_type
        self.embedding_encoder_hidden_size = embedding_encoder_hidden_size
        if embedding_encoder_type == 'max':
            self.encoder = MaxPooler()
        elif embedding_encoder_type == 'min':
            self.encoder = MinPooler()
        elif embedding_encoder_type == 'mean':
            self.encoder = MeanPooler()
        elif embedding_encoder_type == 'linear':
            self.encoder = nn.Linear(self.rnn.hidden_size, embedding_encoder_hidden_size)
            self.embedding_dim = embedding_encoder_hidden_size

    def _tensor_to_cache_key(self, tensor):
        cache_key = " ".join(
            [str(value.item()) for value in tensor.cpu().clone().detach()]
        )
        return cache_key

    def forward(self, x: torch.Tensor):
        embeddings = self._embed(x)

        mask = pad_sequence([torch.ones(sequence.size(0), dtype=torch.int8) for sequence in embeddings],
                            batch_first=True)
        embeddings = pad_sequence(embeddings, batch_first=True).to(self.device)
        h, (hn, _) = self.rnn(embeddings, mask)

        if self.embedding_encoder_type == 'hidden':
            return hn.squeeze(0)
        elif self.embedding_encoder_type == 'linear':
            return self.encoder(hn.squeeze(0))
        else:
            return self.encoder(h)

    def to(self, device, *args, **kwargs):
        if self.embedding_encoder_type == 'rnn':
            self.encoder.to(device, *args, **kwargs)
        return self
