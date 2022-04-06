import os

import numpy as np
import tensorflow as tf

from hesp.config.config import Config
from hesp.embedding_space.abstract_embedding_space import AbstractEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util.hyperbolic_nn import tf_exp_map_zero
from hesp.util.layers import hyp_mlr


class HyperbolicEmbeddingSpace(AbstractEmbeddingSpace):
    def __init__(self, tree: Tree, config: Config, train: bool = True, prototype_path: str = ''):
        super().__init__(tree=tree, config=config, train=train, prototype_path=prototype_path)
        self.geometry = 'hyperbolic'
        # sanity check
        assert self.geometry == config.embedding_space._GEOMETRY, 'config geometry does not match embedding spaces'

        tf.add_to_collection("hyperbolic_vars", self.offsets)
        curv_init = tf.constant_initializer(self.config.embedding_space._INIT_CURVATURE)
        if not train:
            self.c_npy = np.load(os.path.join(prototype_path, 'c.npy'))
            curv_init = tf.constant_initializer(self.c_npy)

        self.curvature = tf.get_variable(
            "curvature",
            shape=[],
            initializer=curv_init,
            dtype=tf.float32,
            trainable=(self.config.prototyper._TRAIN_CURVATURE and train)
        )

    def project(self, embeddings, curvature=0):
        if not curvature:
            curvature = self.curvature
        return tf_exp_map_zero(embeddings, c=curvature)

    def logits(self, embeddings, offsets, normals, curvature):
        return hyp_mlr(
            embeddings,
            c=curvature,
            P_mlr=offsets,
            A_mlr=normals,
        )
