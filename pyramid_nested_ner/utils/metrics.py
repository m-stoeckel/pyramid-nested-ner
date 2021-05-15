"""
Given a list of span->class mappings,
transform the spans into BIOES format annotations for each token
and compute Precision, Recall and F1-Score for these tokens.
"""
from collections import defaultdict
from typing import Dict, Iterable, Iterator, List, Set, Tuple, Union

import numpy as np
from seqeval.reporters import DictReporter, StringReporter


def _bioes_encode_impl(start, end, in_ex):
    assert in_ex in (0, 1)
    start, end = sorted((start, end))
    if end - start == in_ex:
        yield start, 'S'
    else:
        yield start, 'B'
        if end - start > 1 + in_ex:
            for idx in range(start + 1, end - in_ex):
                yield idx, 'I'
        yield end - in_ex, 'E'


def bioes_encode(start: int, end: int) -> Iterator[Tuple[int, str]]:
    """
    :param start: Token start index.
    :param end: Token end index (exclusively).
    :returns: A list of BIOES encoded

    Example:

    >>> tokens = "This is Goethe University in Frankfurt / Main .".split()
    >>> annotations = [(2, 4, "ORG"), (2, 3, "PER"), (3, 4, "ORG"), (5, 8, "LOC")]
    >>> for start, end, category in annotations:
    ...     for idx, tag in bioes_encode(start, end):
    ...         print(tokens[idx], f'{tag}-{category}')
    Goethe B-ORG
    University E-ORG
    Goethe S-PER
    University S-ORG
    Frankfurt B-LOC
    / I-LOC
    Main E-LOC
    """
    yield from _bioes_encode_impl(start, end, 1)


bioes_encode_exclusively = bioes_encode


def bioes_encode_inclusively(start: int, end: int) -> Iterator[Tuple[int, str]]:
    """
    :param start: Token start index.
    :param end: Token end index (inclusively).
    :returns: A list of BIOES encoded

    Example:

    >>> tokens = "This is Goethe University in Frankfurt / Main .".split()
    >>> annotations = [(2, 3, "ORG"), (2, 2, "PER"), (3, 3, "ORG"), (5, 7, "LOC")]
    >>> for start, end, category in annotations:
    ...     for idx, tag in bioes_encode_inclusively(start, end):
    ...         print(tokens[idx], f'{tag}-{category}')
    Goethe B-ORG
    University E-ORG
    Goethe S-PER
    University S-ORG
    Frankfurt B-LOC
    / I-LOC
    Main E-LOC
    """
    yield from _bioes_encode_impl(start, end, 0)


def encode_sequence(sequence, largest_offset):
    bioes_sequence = [defaultdict(set) for _ in range(largest_offset)]
    for start, end, category in sequence:
        for idx, tag in bioes_encode(start, end):
            bioes_sequence[idx][category].add(tag)

    return bioes_sequence


def calc_fb_score(precision, recall, beta):
    precision, recall = np.array(precision), np.array(recall)
    f1_score = np.zeros_like(precision)
    p_r_sum = precision + recall
    f1_score[p_r_sum > 0] = (1. + beta ** 2.) * precision[p_r_sum > 0] * recall[p_r_sum > 0] / p_r_sum[p_r_sum > 0]
    return f1_score


def calc_f1_score(precision, recall) -> Union[np.ndarray, float]:
    return calc_fb_score(precision, recall, 1)


def compute_confusion_matrix_values(prediction, ground_truth, category_names):
    num_categories = len(category_names)

    true_positive = np.zeros(num_categories, dtype=np.int)
    false_positive = np.zeros(num_categories, dtype=np.int)
    false_negative = np.zeros(num_categories, dtype=np.int)

    for p_sequence, t_sequence in zip(prediction, ground_truth):
        largest_offset = max(max(tup[1] for tup in p_sequence), max(tup[1] for tup in t_sequence))

        # BIOES encoded sequences, one entry for each token
        bioes_p_sequence = encode_sequence(p_sequence, largest_offset)
        bioes_t_sequence = encode_sequence(t_sequence, largest_offset)

        for preds, truths in zip(bioes_p_sequence, bioes_t_sequence):
            preds: Dict[str, Set[str]]
            truths: Dict[str, Set[str]]

            for idx, category in enumerate(category_names):
                true_positive[idx] += len(preds[category].intersection(truths[category]))
                false_positive[idx] += len(preds[category].difference(truths[category]))
                false_negative[idx] += len(truths[category].difference(preds[category]))

    return true_positive, false_positive, false_negative


def calculate_metrics(predicted_true_positive, predicted_false_positive, predicted_false_negative):
    shape = predicted_true_positive.shape
    precision, recall = np.zeros(shape), np.zeros(shape)

    # Sums
    prd_positive = predicted_true_positive + predicted_false_positive
    act_positive = predicted_true_positive + predicted_false_negative

    # Precision, Recall, F1-Score per category
    precision[prd_positive > 0] = predicted_true_positive[prd_positive > 0] / prd_positive[prd_positive > 0]
    recall[act_positive > 0] = predicted_true_positive[act_positive > 0] / act_positive[act_positive > 0]
    f1_score = calc_f1_score(precision, recall)

    # Macro metrics
    precision_macro = precision.mean()
    recall_macro = recall.mean()
    f1_score_macro = calc_f1_score(precision_macro, recall_macro)

    # Micro metrics
    precision_micro, recall_micro = 0., 0.
    if prd_positive.sum():
        precision_micro = predicted_true_positive.sum() / prd_positive.sum()
    if act_positive.sum():
        recall_micro = predicted_true_positive.sum() / act_positive.sum()
    f1_score_micro = calc_f1_score(precision_micro, recall_micro)

    # Weighted metrics
    precision_weighted = (precision * predicted_true_positive / predicted_true_positive.sum()).sum()
    recall_weighted = (recall * predicted_true_positive / predicted_true_positive.sum()).sum()
    f1_score_weighted = calc_f1_score(precision_weighted, recall_weighted)

    # Packing return values
    macro = (precision_macro, recall_macro, f1_score_macro)
    micro = (precision_micro, recall_micro, f1_score_micro)
    weighted = (precision_weighted, recall_weighted, f1_score_weighted)

    return precision, recall, f1_score, macro, micro, weighted


def multi_label_bioes_classification_report(
        prediction: List[List[Tuple[int, int, str]]],
        ground_truth: List[List[Tuple[int, int, str]]],
        category_names: Iterable[str],
        output_dict=False,
        digits=2
):
    category_names = sorted(category_names)

    true_positive, false_positive, false_negative = compute_confusion_matrix_values(
        prediction, ground_truth, category_names
    )

    precision, recall, f1_score, macro, micro, weighted = calculate_metrics(
        true_positive,
        false_positive,
        false_negative
    )

    # Report building
    if output_dict:
        reporter = DictReporter()
    else:
        name_width = max(map(len, category_names))
        avg_width = len('Weighted Avg')
        width = max(name_width, avg_width, digits)
        reporter = StringReporter(width=width, digits=digits)

    for idx, name in enumerate(category_names):
        reporter.write(name, precision[idx], recall[idx], f1_score[idx], true_positive[idx])
    reporter.write_blank()

    reporter.write('Macro Avg', *macro, true_positive.sum())
    reporter.write('Micro Avg', *micro, true_positive.sum())
    reporter.write('Weighted Avg', *weighted, true_positive.sum())

    return reporter.report()


if __name__ == '__main__':
    categories = ["ORG", "OTH", "LOC", "PER"]
    report = multi_label_bioes_classification_report(
        [[(2, 3, "ORG"), (2, 2, "PER"), (5, 7, "OTH")]],
        [[(2, 3, "ORG"), (2, 2, "PER"), (3, 3, "ORG"), (5, 7, "LOC")]],
        categories
    )
    print(report)
