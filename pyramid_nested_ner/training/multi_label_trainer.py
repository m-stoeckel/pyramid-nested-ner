import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from pyramid_nested_ner.training.trainer import PyramidNerTrainer
from pyramid_nested_ner.utils.metrics import multi_label_span_classification_report


class MultiLabelTrainer(PyramidNerTrainer):

    def test_model(self, test_data, out_dict=False, return_metrics=False):
        loss = []
        pred, true = [], []
        pbar = tqdm(total=len(test_data))

        pyramid_max_depth = self.nnet.pyramid._max_depth
        if pyramid_max_depth is not None:
            pred_per_layer = [[] for _ in range(pyramid_max_depth + 1)]
            true_per_layer = [[] for _ in range(pyramid_max_depth + 1)]

        for batch in test_data:
            # inference
            self.nnet.eval()
            y, remedy_y, ids = batch.pop('y'), batch.pop('y_remedy'), batch.pop('id')
            with torch.no_grad():
                logits, remedy = self.nnet(**batch)
            self.nnet.train(mode=True)
            # loss computation
            batch_loss = self.compute_loss(logits, y, batch['word_mask'], remedy, remedy_y).item()
            loss.append(batch_loss)
            layers_y_hat = self._model.logits_to_classes(logits)
            remedy_y_hat = self._model.remedy_to_classes(remedy)
            pbar.set_description(f'valid loss: {round(np.mean(loss), 3)}')

            y_pred, y_true = self._classes_to_iob2(layers_y_hat, y, False, remedy_y_hat, remedy_y)

            if pyramid_max_depth is not None:
                self.layer_wise_classes(y_pred, pred_per_layer, pyramid_max_depth)
                self.layer_wise_classes(y_true, true_per_layer, pyramid_max_depth)

            # Extend all preds/true with flattened batch results
            pred.extend([seq for layer in y_pred for seq in layer])
            true.extend([seq for layer in y_true for seq in layer])
            pbar.update(1)

        report = self.classification_report(pred, true, out_dict)
        if out_dict:
            report['loss'] = np.mean(loss)
            f1_score = round(report["micro avg"]["f1-score"] * 100, 2)
            pbar.set_description(
                f'valid loss: {round(np.mean(loss), 2)}; micro f1: {f1_score}%'
            )
        elif pyramid_max_depth is not None:
            report += "\n\n"
            for depth in range(pyramid_max_depth):
                report += f"Sequence Length: {depth + 1}\n"
                report += self.classification_report(pred_per_layer[depth], true_per_layer[depth], out_dict)

        pbar.close()

        return report

    def layer_wise_classes(self, y, per_layer, pyramid_max_depth):
        for i in range(min(pyramid_max_depth, len(y))):
            per_layer[i].extend([seq for seq in y[i]])
        if len(y) > pyramid_max_depth:
            per_layer[-1].extend([seq for layer in y[pyramid_max_depth:] for seq in layer])

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
        return multi_label_span_classification_report(
            y_true,
            y_pred,
            self._model.label_encoder.entities,
            digits=2,
            output_dict=out_dict
        )
