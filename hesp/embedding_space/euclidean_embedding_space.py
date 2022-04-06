import tensorflow as tf

from hesp.config.config import Config
from hesp.embedding_space.abstract_embedding_space import AbstractEmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.util.layers import euc_mlr


class EuclideanEmbeddingSpace(AbstractEmbeddingSpace):
    def __init__(self, tree: Tree, config: Config, train: bool = True, prototype_path: str = ''):
        super().__init__(tree=tree, config=config, train=train, prototype_path=prototype_path)
        self.geometry = 'euclidean'
        # sanity check
        assert self.geometry == config.embedding_space._GEOMETRY, 'config geometry does not match embedding spaces'

        self.curvature = tf.get_variable(
            "curvature",
            shape=[],
            initializer=tf.constant_initializer(0.),
            dtype=tf.float32,
            trainable=False
        )

    def project(self, embeddings, curvature=0):
        return tf.identity(embeddings)

    def logits(self, embeddings, offsets, normals, curvature=0.):
        return euc_mlr(embeddings, P_mlr=offsets, A_mlr=normals)
