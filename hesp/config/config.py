import logging
import os
from pprint import pformat

import numpy as np
import tensorflow as tf

from hesp.config.dataset_config import DATASET_CFG_DICT
from hesp.config.embedding_space_config import EmbeddingSpaceConfig
from hesp.config.prototyper_config import VisualizerConfig
from hesp.config.segmenter_config import SegmenterConfig

CFG_DIR = __file__
BASE_DIR = '/'.join(CFG_DIR.split('/')[:-3])

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    visualizer = None
    segmenter = None
    embedding_space = None
    base_model_name = None

    # tf and numpy random seed
    tf.reset_default_graph()
    _SEED = 1
    tf.random.set_random_seed(_SEED)
    np.random.seed(_SEED)

    _IDENT = ''

    def __init__(self, dataset, mode, base_save_dir="", gpu_idx=0, json_selection=""):
        self._MODE = mode
        if base_save_dir:
            self._BASE_DIR = base_save_dir
        else:  # automatically select base directory of current file.
            self._BASE_DIR = BASE_DIR
        try:
            self.dataset = DATASET_CFG_DICT[dataset]
            if ('ade' in dataset) or ('ADE' in dataset):
                if json_selection != "":
                    self.dataset._JSON_FILE = os.path.join(self.dataset._DATASET_DIR, json_selection)
        except KeyError:
            logger.error(f'Dataset {dataset} not supported yet. Valid entries: {[k for k in DATASET_CFG_DICT.keys()]}')
            raise NotImplementedError
        self._GPU_IDX = gpu_idx
        self.visualizer = VisualizerConfig()
        self.segmenter = SegmenterConfig()
        self.embedding_space = EmbeddingSpaceConfig()

    @property
    def _IDENTIFIER(self, ):
        if self.embedding_space._HIERARCHICAL:
            _ident = 'hierarchical_'
        else:
            _ident = ''
        _ident += f'{self.dataset._NAME}_d{self.embedding_space._DIM}_{self.embedding_space._GEOMETRY}'
        if self.embedding_space._GEOMETRY == 'hyperbolic':
            _ident += f'_c{self.embedding_space._INIT_CURVATURE}'
        return _ident

    @property
    def _SEGMENT_IDENTIFIER(self, ):
        _ident = self._IDENT
        if self.segmenter._ZERO_LABEL:
            _ident += 'ZL_'
        _ident += self._IDENTIFIER
        _ident += f'_os{self.segmenter._OUTPUT_STRIDE}'
        _ident += f'_{self.segmenter._BACKBONE}'
        _ident += f'_bs{self.segmenter._BATCH_SIZE}'
        _ident += f'_lr{self.segmenter._INITIAL_LEARNING_RATE}'
        _ident += f'_fbn{self.segmenter._FREEZE_BN}'
        _ident += f'_fbb{self.segmenter._FREEZE_BACKBONE}'
        if self.segmenter._SEGMENTER_IDENT != "":
            _ident += '_' + self.segmenter._SEGMENTER_IDENT

        if self.embedding_space._DIM != self.segmenter._EFN_OUT_DIM:
            _ident += '_nomatch'
        return _ident

    @property
    def _SEGMENTER_SAVE_DIR(self, ):
        home_dir = os.path.abspath('poincare-hesp/')
        save_dir = os.path.join(home_dir, 'save', self.segmenter._SEGMENTER_DIR, self._SEGMENT_IDENTIFIER)
        try:
            os.makedirs(save_dir)
        except FileExistsError:
            pass
        return save_dir

    def pretty_print(self, ):
        """ pprint dict representation of configuratons """
        d = {
            'dataset': {k: v for k, v in self.dataset.__dict__.items() if
                        k not in ['_JSON', '_INITIAL_LEARNING_RATE', '_NUM_EPOCHS', '_I2C', '__doc__', '__module__']},
            'embedding_space': self.embedding_space.__dict__,
            'save_dirs': {}
        }

        if self._MODE == 'segmenter':
            d['segmenter'] = self.segmenter.__dict__
            d['save_dirs']['segmenter'] = self._SEGMENTER_SAVE_DIR
        logger.info('Configuration:')
        logger.info(pformat(d))
