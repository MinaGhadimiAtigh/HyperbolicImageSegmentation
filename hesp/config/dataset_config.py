import os

CFG_DIR = __file__
BASE_DIR = '/'.join(CFG_DIR.split('/')[:-3])
DSET_DIR = os.path.join(BASE_DIR, 'datasets')


def txt2dict(fn):
    """ Reads txt file and converts to idx2concept dictionary"""
    f = open(fn, 'r')
    ls = f.readlines()
    f.close()
    d = {}
    for l in ls:
        i, c = l.split(':')
        d[int(i)] = c.strip()
    return d


class DatasetConfig:
    # Dataset configuration
    _NUM_CLASSES = 0

    # Segmentation config
    _NUM_TRAIN = 0
    _NUM_VALIDATION = 0
    _NUM_EPOCHS = 0
    _INITIAL_LEARNING_RATE = 0

    _DATASET_DIR = ''
    _JSON_FILE = ''
    _I2C_FILE = ''
    _RGB_MEANS = [0, 0, 0]
    _I2C = {}


class CocoConfig(DatasetConfig):
    # Dataset configuration
    _NAME = 'coco'
    _DATASET_DIR = os.path.join(DSET_DIR, 'coco')
    _DATA_DIR = os.path.join(_DATASET_DIR, 'coco_data')
    _TEST_FILE = os.path.join(_DATA_DIR, 'coco_val.record')
    _NUM_CLASSES = 182

    # Segmentation config
    _NUM_TRAIN = 9000
    _NUM_VALIDATION = 1000
    _NUM_EPOCHS = 70
    _INITIAL_LEARNING_RATE = 1e-3

    _RGB_MEANS = [120.23, 114.25, 104.129]
    _JSON_FILE = os.path.join(_DATASET_DIR, 'COCO10K_hierarchy.json')
    _I2C_FILE = os.path.join(_DATASET_DIR, 'COCO10K_i2c.txt')
    _I2C = txt2dict(_I2C_FILE)

    _UNSEEN = ['frisbee', 'skateboard', 'cardboard', 'carrot', 'scissors',
               'suitcase', 'giraffe', 'cow', 'road', 'wall-concrete', 'tree',
               'grass', 'river', 'clouds', 'playingfield']


class PascalConfig(DatasetConfig):
    # Dataset configuration
    _NAME = 'pascal'
    _DATASET_DIR = os.path.join(DSET_DIR, 'pascal')
    _DATA_DIR = os.path.join(_DATASET_DIR, 'pascal_data')
    _TEST_FILE = os.path.join(_DATA_DIR, 'voc_val.record')
    _NUM_CLASSES = 21

    # Segmentation config
    _NUM_TRAIN = 10582
    _NUM_VALIDATION = 1449
    _NUM_EPOCHS = 40
    _INITIAL_LEARNING_RATE = 1e-2

    _RGB_MEANS = [123.68, 116.78, 103.94]
    _JSON_FILE = os.path.join(_DATASET_DIR, 'PASCAL_hierarchy.json')
    _I2C_FILE = os.path.join(_DATASET_DIR, 'PASCAL_i2c.txt')
    _I2C = txt2dict(_I2C_FILE)

    _UNSEEN = ['pot_plant', 'sheep', 'couch', 'locomotive', 'television']


class ToyConfig(DatasetConfig):
    _NAME = 'toy'
    _DATASET_DIR = os.path.join(DSET_DIR, 'toy')
    _JSON_FILE = os.path.join(_DATASET_DIR, 'toy_hierarchy.json')
    _I2C_FILE = os.path.join(_DATASET_DIR, 'toy_i2c.txt')
    _I2C = txt2dict(_I2C_FILE)


class ADEConfig(DatasetConfig):
    _NAME = 'ADE'
    _DATASET_DIR = os.path.join(DSET_DIR, 'ADE')

    _DATA_DIR = os.path.join(_DATASET_DIR, 'ade_data')
    _TEST_FILE = os.path.join(_DATA_DIR, 'ade_val.record')

    _JSON_FILE = os.path.join(_DATASET_DIR, 'ADE20K_hierarchy_v3.json')
    _I2C_FILE = os.path.join(_DATASET_DIR, 'ADE20K_i2c.txt')
    _I2C = txt2dict(_I2C_FILE)

    _NUM_CLASSES = 150
    _NUM_TRAIN = 20210
    _NUM_VALIDATION = 2000
    _NUM_EPOCHS = 140
    _INITIAL_LEARNING_RATE = 1e-3

    @property
    def _UNSEEN(self, ):
        raise NotImplementedError


DATASET_CFG_DICT = {
    'coco': CocoConfig,
    'pascal': PascalConfig,
    'toy': ToyConfig,
    'ade': ADEConfig,
}
