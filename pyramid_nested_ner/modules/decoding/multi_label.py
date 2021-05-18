from enum import Enum
from typing import Union

import torch
from torch import nn as nn

from pyramid_nested_ner.modules.decoding.classifiers import ConvolutionalMultiHeadClassifier, \
    CudaStreamMultiHeadClassifier, LinearMultiHeadClassifier
from pyramid_nested_ner.modules.decoding.linear import LinearDecoder


class SigmoidLinearDecoder(LinearDecoder):
    def __init__(self, input_size, classes, classifier_type=None):
        super().__init__(input_size, classes)

    def _init_remedy_decoder(self, input_size, classes):
        return nn.Linear(input_size, classes * 2)


class OneVsRestDecoder(LinearDecoder):
    class ClassifierType(Enum):
        convolutional = ConvolutionalMultiHeadClassifier
        multihead = LinearMultiHeadClassifier
        stream = CudaStreamMultiHeadClassifier

    def __init__(self, input_size, classes, classifier_type):
        self.classifier_cls = self.ClassifierType[classifier_type].value
        super().__init__(input_size, classes)

    def _init_linear_decoder(self, input_size, classes):
        return self.classifier_cls(input_size, classes)

    def _init_remedy_decoder(self, input_size, classes):
        return self.classifier_cls(input_size, classes * 2)


class OneVsRestConvDecoder(OneVsRestDecoder):
    def __init__(self, input_size, classes):
        super().__init__(input_size, classes, 'convolutional')


class OneVsRestMultiHeadDecoder(OneVsRestDecoder):
    def __init__(self, input_size, classes):
        super().__init__(input_size, classes, 'multihead')


class ContextualDecoder:
    linear_decoder: Union[
        nn.Linear, ConvolutionalMultiHeadClassifier, CudaStreamMultiHeadClassifier, LinearMultiHeadClassifier
    ]
    remedy_decoder: Union[
        nn.Linear, ConvolutionalMultiHeadClassifier, CudaStreamMultiHeadClassifier, LinearMultiHeadClassifier
    ]

    def forward(self, h_layers, x_context, h_remedy=None):
        logits = [
            self.linear_decoder(
                torch.cat((h, x_context.unsqueeze(1).repeat(1, h.size(1), 1)), dim=-1)
            )
            for h in h_layers
        ]

        logits_remedy = None
        if h_remedy is not None:
            logits_remedy = self.remedy_decoder(
                torch.cat((h_remedy, x_context.unsqueeze(1).repeat(1, h_remedy.size(1), 1)), dim=-1)
            )

        return logits, logits_remedy


class ContextualSigmoidLinearDecoder(ContextualDecoder, SigmoidLinearDecoder):
    pass


class ContextualOneVsRestConvDecoder(ContextualDecoder, OneVsRestDecoder):
    def __init__(self, input_size, classes):
        super().__init__(input_size, classes, 'convolutional')


class ContextualOneVsRestMultiHeadDecoder(ContextualDecoder, OneVsRestDecoder):
    def __init__(self, input_size, classes):
        super().__init__(input_size, classes, 'multihead')
