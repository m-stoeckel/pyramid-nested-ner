from typing import Optional, Union

import flair
import flair.embeddings
import torch
from flair.embeddings import WordEmbeddings


class DocumentEmbeddings(object):
    def __init__(
            self,
            lexicon,
            padding_idx=0,
            device='cpu',
            casing=True,
    ):
        self.pad_index = padding_idx
        self.lexicon = lexicon
        self.casing = casing
        self._train = True
        self._cache = {}
        self.device = device

        self.embeddings: Union[
            flair.embeddings.DocumentRNNEmbeddings,
            flair.embeddings.TransformerDocumentEmbeddings,
            flair.embeddings.SentenceTransformerDocumentEmbeddings
        ] = None

    def to(self, device, *args, **kwargs):
        self.device = device
        self.embeddings.to(device, *args, **kwargs)
        return self

    def train(self, mode=True):
        self.embeddings.train(mode)
        self._train = mode

    def eval(self):
        self.embeddings.train(False)
        self._train = False

    @property
    def vocab_idx(self):
        return {i: token.lower() if not self.casing else token for i, token in enumerate(self.lexicon)}

    @property
    def embedding_dim(self):
        return self.embeddings.embedding_length

    def __tensor_to_cache_key(self, ltensor):
        cache_key = " ".join(
            [str(value.item()) for tensor in ltensor for value in tensor.cpu().clone().detach()]
        )
        return cache_key

    def _get_from_cache(self, key):
        key = self.__tensor_to_cache_key(key)
        return self._cache.get(key)

    def _add_to_cache(self, key, value):
        key = self.__tensor_to_cache_key(key)
        self._cache[key] = value.cpu()

    def __call__(self, x):
        pass


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
            device=device,
            casing=casing,
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

        self.embeddings.to(device)

    def _embed(self, x: torch.Tensor):
        vocab_idx = self.vocab_idx
        embeddings = []
        for sequences in x:
            if self._get_from_cache(sequences) is None:
                flair_sentence = flair.data.Sentence()
                for sequence in sequences:
                    for index in sequence:
                        index = index.item()
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

    def __call__(self, x):
        return self._embed(x)

# prior_sentences=True,
# following_sentences=False,

#     self._init_document_embeddings_encoder(
#         document_embeddings_encoder_type.lower(),
#         document_embeddings_encoder_hidden_size
#     )
#
# def _init_document_embeddings_encoder(
#         self,
#         encoder_type: str,
#         encoder_hidden_size: int
# ):
#     if encoder_type == 'lstm':
#         self.document_embeddings_encoder = FastRNN(
#             nn.LSTM,
#             input_size=self.document_embeddings.embedding_length,
#             hidden_size=encoder_hidden_size,
#             batch_first=True,
#             bidirectional=True
#         )
#     elif encoder_type == 'max':
#         self.document_embeddings_encoder = Pooler(torch.max)
#     elif encoder_type == 'mean':
#         self.document_embeddings_encoder = Pooler(torch.mean)
#     elif encoder_type == 'min':
#         self.document_embeddings_encoder = Pooler(torch.min)
#     else:
#         raise ValueError(f"Invalid document embedding encoder type: '{encoder_type}'")


# class Pooler(nn.Module):
#     def __init__(self, pooling=torch.max):
#         super(Pooler, self).__init__()
#
#         self.pooling = pooling
#
#     def forward(self, input):
#         return self.pooling(input, dim=0)
