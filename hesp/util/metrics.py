import logging

import numpy as np
import tensorflow as tf

from hesp.hierarchy.tree import Tree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def npy_metrics(total_cm, tree):
    sum_over_row = np.sum(total_cm, axis=0)  # sh M
    acc = np.sum(np.diag(total_cm)) / np.sum(sum_over_row)

    sum_over_col = np.sum(total_cm, axis=1)
    cm_diag = np.diag(total_cm)
    denominator = sum_over_row + sum_over_col - cm_diag

    valid_entries = denominator != 0
    num_valid_entries = np.sum(valid_entries)

    denominator[denominator == 0] = 1
    sum_over_row[sum_over_row == 0] = 1
    cacc = np.mean(np.diag(total_cm)[valid_entries] / sum_over_row[valid_entries])
    iou = cm_diag / denominator

    iou = np.sum(iou) / num_valid_entries
    return iou, acc, cacc


def hierarchical_cm(total_cm, tree):
    """
    Calculate hierarchical metrics in the following way.
    Assuming we have a set of cousins and siblings,
    total_cm holds pred in cols and labels in rows.
    If a label is cat, and it predicts a dog, it will be counted as correct to cat
    if a label is dog, and it predicts a cat, it will be counted as correct to dog
    e.g.
    False positive is added to True positive, and set to 0 for siblings
    predicted false positive dog where there was a cat? those pixels will be true positives now
    thats it. 
    """

    # sibling cm
    sib_cm = total_cm.copy()
    for i in tree.target_classes:
        sib_idx = tree.get_by_idx(i).metric_siblings
        for sidx in sib_idx:
            if i != sidx:  # for all siblings
                # FP (in row i(= label i) , col sibx (=predict sibling)) add to TP [i,i]
                sib_cm[i, i] += sib_cm[i, sidx]
                # erase that FP from confusion matrix
                sib_cm[i, sidx] = 0

    # cousin cm
    cousin_cm = total_cm.copy()
    for i in tree.target_classes:
        cousin_idx = tree.get_by_idx(i).metric_cousins
        for cidx in cousin_idx:
            if i != cidx:  # for all cousins
                cousin_cm[i, i] += total_cm[i, cidx]
                cousin_cm[i, cidx] = 0

    return sib_cm, cousin_cm


def cls_mean_iou_npy(total_cm, tree):
    sum_over_row = np.sum(total_cm, axis=0)
    sum_over_col = np.sum(total_cm, axis=1)
    cm_diag = np.diag(total_cm)
    denominator = sum_over_row + sum_over_col - cm_diag

    denominator[denominator == 0] = 1
    iou = cm_diag / denominator

    metrics = {}
    for i in tree.target_classes:
        cls_name = tree.i2n[i]
        metrics[cls_name] = iou[i]
    return metrics


def mean_iou_matrix(total_cm):
    sum_over_row = tf.to_float(tf.reduce_sum(total_cm, 0))
    sum_over_col = tf.to_float(tf.reduce_sum(total_cm, 1))
    cm_diag = tf.to_float(tf.diag_part(total_cm))
    denominator = sum_over_row + sum_over_col - cm_diag

    # The mean is only computed over classes that appear in the
    # label or prediction tensor. If the denominator is 0, we need to
    # ignore the class.
    num_valid_entries = tf.reduce_sum(
        tf.cast(tf.not_equal(denominator, 0), dtype=tf.float32)
    )

    # If the value of the denominator is 0, set it to 1 to avoid
    # zero division.
    denominator = tf.where(
        tf.greater(denominator, 0), denominator, tf.ones_like(denominator)
    )
    iou = tf.div(cm_diag, denominator)
    return iou, num_valid_entries


def compute_mean_iou(total_cm, tree: Tree):
    """Compute the mean intersection-over-union via the confusion matrix."""
    iou, num_valid_entries = mean_iou_matrix(total_cm)
    with tf.variable_scope('class_metrics'):
        for i in tree.target_classes:
            cls_name = tree.i2n[i]
            tf.identity(iou[i], name=f"train_iou_{cls_name}")
            tf.summary.scalar(f"train_iou_{cls_name}", iou[i])

    # If the number of valid entries is 0 (no classes) we return 0.
    result = tf.where(
        tf.greater(num_valid_entries, 0),
        tf.reduce_sum(iou, name='mean_iou') / num_valid_entries,
        0,
    )
    return result


def tf_metrics(labels, prediction, num_classes):
    accuracy = tf.metrics.accuracy(labels, prediction)
    mean_iou = tf.metrics.mean_iou(labels, prediction, num_classes=num_classes)
    class_accuracy = tf.metrics.mean_per_class_accuracy(labels, prediction, num_classes=num_classes)
    return {
        "px_accuracy": accuracy,
        "mean_iou": mean_iou,
        "class_accuracy": class_accuracy,
    }
