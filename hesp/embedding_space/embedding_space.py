import logging

from hesp.config.config import Config
from hesp.embedding_space.euclidean_embedding_space import EuclideanEmbeddingSpace
from hesp.embedding_space.hyperbolic_embedding_space import HyperbolicEmbeddingSpace
from hesp.hierarchy.tree import Tree

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def EmbeddingSpace(tree: Tree, config: Config, train: bool, prototype_path: str = ''):
    """Returns correct embedding space obj based on config."""
    if config.embedding_space._GEOMETRY == 'hyperbolic':
        base = HyperbolicEmbeddingSpace
    elif config.embedding_space._GEOMETRY == 'euclidean':
        base = EuclideanEmbeddingSpace
    else:
        logger.error(f'Embedding space geometry {config.embedding_space._GEOMETRY} not supported.')
        raise NotImplementedError

    class EmbeddingSpace(base):
        def __init__(self, tree: Tree, config: Config, train: bool = True, prototype_path: str = ''):
            super().__init__(tree=tree, config=config, train=train, prototype_path=prototype_path)

    return EmbeddingSpace(tree=tree, config=config, train=train, prototype_path=prototype_path)
