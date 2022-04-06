import tensorflow as tf

from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import EPS


def CCE_old(cond_probs, labels, tree: Tree):
    """ Categorical cross-entropy loss.
    Suppports both flat and hierarchical classification.
    Calculated as -mean(sum(log(p_correct)))

    Args:
        probs: flattened probabilities over H, NxM
        labels: flattened idx of correct class, N
    Returns:
        loss object
    """
    superhot = tf.gather(tree.hmat, labels)  # N,M
    superhot = tf.gather(superhot, tree.train_classes, axis=-1)

    cond_probs = tf.gather(cond_probs, tree.train_classes, axis=-1)
    posprobs = tf.multiply(tf.log(tf.maximum(cond_probs, EPS)), superhot)
    posprobs = tf.debugging.check_numerics(posprobs, 'loss pos probs nan')

    possum = tf.reduce_sum(posprobs, axis=-1)
    possum = tf.debugging.check_numerics(possum, 'loss possum nan')

    loss = -tf.reduce_mean(possum)
    return loss


def CCE(cond_probs, labels, tree: Tree):
    """ Categorical cross-entropy loss.
    Suppports both flat and hierarchical classification.
    Calculated as -mean(sum(log(p_correct)))

    Args:
        probs: flattened probabilities over H, NxM
        labels: flattened idx of correct class, N
    Returns:
        loss object
    """

    log_probs = tf.log(tf.maximum(cond_probs, EPS))

    log_sum_p = tf.tensordot(log_probs, tree.hmat, axes=[-1, -1])

    pos_logp = tf.gather_nd(
        log_sum_p, labels[:, tf.newaxis], batch_dims=1
    )
    loss = -tf.reduce_mean(pos_logp)
    return loss
