import logging
import os
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf

from hesp.config.config import Config
from hesp.hierarchy.tree import Tree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AbstractEmbeddingSpace(ABC):
    """
    General purpose base class for implementing embedding spaces.
    Softmax function turns into hierarchical softmax only when the 'tree' has an hierarchal structure defined.
    """

    offsets = None
    normals = None
    curvature = None

    def __init__(self, tree: Tree, config: Config, train: bool = True, prototype_path: str = ''):
        self.tree = tree
        self.config = config
        self.dim = config.embedding_space._DIM

        logger.info("Setup embedding space variables..")

        offset_init = tf.constant_initializer(0.)
        normal_init = tf.random_normal_initializer(
            mean=0.0, stddev=0.05
        )
        if not train:
            self.normals_npy = np.load(os.path.join(prototype_path, 'normals.npy'))
            self.offsets_npy = np.load(os.path.join(prototype_path, 'offsets.npy'))
            offset_init = tf.constant_initializer(self.offsets_npy)
            normal_init = tf.constant_initializer(self.normals_npy)

        self.normals = tf.get_variable(
            "normals",
            shape=[self.tree.M, self.dim],
            initializer=normal_init,
            dtype=tf.float32,
            trainable=train,
        )
        # tf.summary.histogram('normals',self.normals)
        self.offsets = tf.get_variable(
            "offsets",
            shape=[self.tree.M, self.dim],
            initializer=offset_init,
            dtype=tf.float32,
            trainable=train,
        )
        # tf.summary.histogram('offsets', self.offsets)

    def softmax(self, logits):
        """ Performs softmax function. Agnostic to hierarchical or flat tree."""
        with tf.variable_scope("softmax"):
            logits -= tf.reduce_max(logits, axis=-1, keepdims=True)
            Z = tf.tensordot(tf.exp(logits), self.tree.sibmat, axes=[-1, -1])
            cond_probs = tf.exp(logits) / tf.maximum(Z, 1e-15)
            return cond_probs

    def decide(self, probs, unseen=[]):
        """ Decide on leaf class from probabilities. """
        # cls_probs = tf.gather(probs, self.tree.target_classes, axis=-1) # only consider target classes in decision
        # choice = tf.argmax(cls_probs, axis=-1) # N sized array of indices into target classes
        # decision = tf.gather(self.tree.target_classes, choice) # revert back to original indices
        # return decision
        # more memory intensive I think ->
        ## in case of discontinuous cls ids (looking at you COCO stuff),
        ## tree.hmat[:K,:K] holds class occurance in the diagonal, dot with that removes 'missing' classes from argmax
        cls_probs = tf.gather(probs, np.arange(self.tree.K), axis=-1)
        cls_probs = tf.tensordot(cls_probs, self.tree.hmat[:self.tree.K, :self.tree.K], axes=[-1, -1])
        if len(unseen):
            cls_gather = tf.gather(cls_probs, unseen, axis=-1)
            predict_ = tf.argmax(cls_gather, axis=-1)
            predictions = tf.gather(unseen, predict_)
        else:
            predictions = tf.argmax(cls_probs, axis=-1)
        return predictions

    def run(self, embeddings, offsets=None, normals=None, curvature=None):
        """ Calculates (joint) probabilities for incoming embeddings. Assumes embeddings are already on manifold. """
        if offsets is None:
            offsets = self.offsets
        if normals is None:
            normals = self.normals
        if curvature is None:
            curvature = self.curvature
        logits = self.logits(embeddings=embeddings, offsets=offsets, normals=normals, curvature=curvature)
        logits = tf.debugging.check_numerics(logits, 'logts nan')
        cond_probs = self.softmax(logits)
        cond_probs = tf.debugging.check_numerics(cond_probs, 'cond_probs nan')
        # tf.summary.histogram('cond_probs',cond_probs)
        joints = self.get_joints(cond_probs)
        joints = tf.debugging.check_numerics(joints, 'joints nan')
        return joints, cond_probs

    def get_joints(self, cond_probs):
        """ Calculates joint probabilities based on conditionals """
        with tf.variable_scope("joints"):
            log_probs = tf.log(tf.maximum(cond_probs, 1e-4))
            log_probs = tf.debugging.check_numerics(log_probs, 'log_probs nan')
            # tf.summary.histogram('log_probs',log_probs)
            log_sum_p = tf.tensordot(log_probs, self.tree.hmat, axes=[-1, -1])
            log_sum_p = tf.debugging.check_numerics(log_sum_p, 'log_sum_p normal_init')
            # tf.summary.histogram('log_sum_probs',log_sum_p)
            joints = tf.exp(log_sum_p)
            return joints

    @abstractmethod
    def logits(self, embeddings, offsets=None, normals=None, curvature=None):
        """ Returns logits to pass to (hierarchical) softmax function."""
        pass
