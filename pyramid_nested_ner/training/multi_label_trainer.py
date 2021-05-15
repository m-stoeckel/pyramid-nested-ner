import torch
import torch.nn as nn

from pyramid_nested_ner.training.trainer import PyramidNerTrainer
from pyramid_nested_ner.utils.metrics import multi_label_span_classification_report


class MultiLabelTrainer(PyramidNerTrainer):

    @staticmethod
    def compute_loss(logits, y, mask, remedy_logits=None, remedy_y=None) -> torch.Tensor:
        assert len(logits) == len(y), 'Predictions and labels are misaligned.'
        if remedy_y is None or remedy_logits is None:
            assert remedy_y is None and remedy_logits is None, 'Predictions and labels are misaligned'
        binary_cross_entropy = nn.BCEWithLogitsLoss(reduction='none')
        loss = 0.0
        for i, (logits_layer, y_layer) in enumerate(zip(logits, y)):
            layer_loss = binary_cross_entropy(logits_layer, y_layer)
            layer_mask = mask[:, i:].unsqueeze(-1).expand_as(layer_loss)
            loss += torch.sum(layer_loss * layer_mask)

        if remedy_y is not None and remedy_logits is not None:
            ml_loss = binary_cross_entropy(remedy_logits, remedy_y)
            ml_mask = mask[:, len(logits):].unsqueeze(-1).expand_as(ml_loss)
            loss += torch.sum(ml_loss * ml_mask)

        # note that we return the sum of the loss of each token, rather than averaging it;
        # average leads to a loss that is too small and generates small gradients that pr-
        # event the model from learning anything due to its depth.

        return loss

    def _classes_to_iob2(self, pred, true, flatten=False, remedy_pred=None, remedy_true=None):
        y_pred = self._model.classes_to_iob2(pred, remedy=remedy_pred)
        y_true = self._model.classes_to_iob2(true, remedy=remedy_true)
        if len(y_pred) > len(y_true):
            y_true.extend([[[[] for _ in y] for y in extra_layer] for extra_layer in y_pred[len(y_true):]])
        if len(y_true) > len(y_pred):
            y_pred.extend([[[[] for _ in y] for y in extra_layer] for extra_layer in y_true[len(y_pred):]])
        if flatten:
            y_pred = [seq for layer in y_pred for seq in layer]
            y_true = [seq for layer in y_true for seq in layer]

        return y_pred, y_true

    def classification_report(self, y_pred, y_true, out_dict=False):
        report = multi_label_span_classification_report(
            y_true,
            y_pred,
            self._model.label_encoder.entities,
            digits=4,
            output_dict=out_dict
        )
        return report
