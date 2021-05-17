import torch
from torch import nn as nn

from pyramid_nested_ner.modules.decoding.linear import LinearDecoder


class SigmoidLinearDecoder(LinearDecoder):
    def _init_remedy_decoder(self, classes, input_size):
        return nn.Linear(input_size, classes * 2)


class ContextualSigmoidLinearDecoder(SigmoidLinearDecoder):
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