import json as json_lib
import logging
from pprint import pformat

from hesp.hierarchy.tree import Tree
from hesp.models.segmenter import Segmenter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelFactory:

    def create(mode, config):
        """ Initializes and returns a model based on config and mode."""
        logger.info('CONFIGURATION')
        config.pretty_print()
        model_dict = {
            'segmenter': Segmenter,
        }
        # initialize tree containing target class relationships
        if config.embedding_space._HIERARCHICAL:
            json = json_lib.load(open(config.dataset._JSON_FILE))
        else:
            json = {}
        tree_params = {
            'i2c': config.dataset._I2C,
            'json': json
        }
        class_tree = Tree(**tree_params)
        logger.info("IMPORTANT TRAINING RELATIONSHIP INFORMATION: Acting class structure: ")
        logger.info(pformat(class_tree.json))

        # initialize embedding space.
        train_embedding_space = True
        prototype_path = ''

        # initialize model
        model_params = {
            'tree': class_tree,
            'config': config,
            'train_embedding_space': train_embedding_space,
            'prototype_path': prototype_path
        }
        return model_dict[mode](**model_params)
