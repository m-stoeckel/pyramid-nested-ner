from typing import Union

import torch
from torch import nn as nn

from pyramid_nested_ner.modules.decoding.classifiers import ConvolutionalMultiHeadClassifier, \
    CudaStreamMultiHeadClassifier, LinearMultiHeadClassifier
from pyramid_nested_ner.modules.decoding.linear import LinearDecoder


class SigmoidLinearDecoder(LinearDecoder):
    def _init_remedy_decoder(self, input_size, classes):
        return nn.Linear(input_size, classes * 2)


class OneVsRestDecoder(LinearDecoder):
    def __init__(self, input_size, classes):
        super().__init__(input_size, classes)

    def _init_linear_decoder(self, input_size, classes):
        return ConvolutionalMultiHeadClassifier(input_size, classes)

    def _init_remedy_decoder(self, input_size, classes):
        return ConvolutionalMultiHeadClassifier(input_size, classes * 2)


class ContextualDecoder:
    linear_decoder: Union[
        nn.Linear, ConvolutionalMultiHeadClassifier, CudaStreamMultiHeadClassifier, LinearMultiHeadClassifier
    ]
    remedy_decoder: Union[
        nn.Linear, ConvolutionalMultiHeadClassifier, CudaStreamMultiHeadClassifier, LinearMultiHeadClassifier
    ]

    def forward(self, h_layers, x_context, h_remedy=None):
        # TODO: Make sure, that h_layers[0] is always the layer with the longest sequence
        x_context = x_context.unsqueeze(1).repeat(1, h_layers[0].size(1), 1)
        logits = [
            self.linear_decoder(
                torch.cat((h, x_context[:, : h.size(1), :]), dim=-1)
            )
            for h in h_layers
        ]

        if h_remedy is not None:
            return logits, self.remedy_decoder(torch.cat((h_remedy, x_context[:, :h_remedy.size(1), :]), dim=-1))
        else:
            return logits, None


class ContextualSigmoidLinearDecoder(ContextualDecoder, SigmoidLinearDecoder):
    pass


class ContextualOneVsRestDecoder(ContextualDecoder, OneVsRestDecoder):
    pass
