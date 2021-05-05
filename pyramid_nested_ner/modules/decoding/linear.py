import torch.nn as nn


class LinearDecoder(nn.Module):

    def __init__(self, input_size, classes):
        super(LinearDecoder, self).__init__()
        self.linear_decoder = nn.Linear(input_size, classes)
        self.remedy_decoder = nn.Linear(input_size, (classes - 1) * 2)

    def forward(self, h_layers, h_remedy=None):
        logits = [self.linear_decoder(h) for h in h_layers]
        if h_remedy is not None:
            return logits, self.remedy_decoder(h_remedy)
        else:
            return logits, None

    def to(self, device, *args, **kwargs):
        self.linear_decoder = self.linear_decoder.to(device, *args, **kwargs)
        self.remedy_decoder = self.remedy_decoder.to(device, *args, **kwargs)
        super(LinearDecoder, self).to(device, *args, **kwargs)
