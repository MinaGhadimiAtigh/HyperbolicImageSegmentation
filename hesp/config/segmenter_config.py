CFG_DIR = __file__
BASE_DIR = '/'.join(CFG_DIR.split('/')[:-3])


class SegmenterConfig:
    _MODEL_DIR = ''
    _GRAD_CLIP = 1.

    # Use embedding space as prototype
    _PROTOTYPE = False

    # Whether the segmenter is doing zero label training
    _ZERO_LABEL = False

    # Whether the segmenter is doing zero label testing
    _TEST_ZERO_LABEL = False

    # Pretrained backbone weights
    _PRETRAINED_MODEL = None

    # Data configuration
    _HEIGHT = 513
    _WIDTH = 513
    _DEPTH = 3
    _MIN_SCALE = 0.5
    _MAX_SCALE = 2.0
    _IGNORE_LABEL = 255

    # Training configuration
    _EPOCHS_PER_EVAL = 1
    _BATCH_SIZE = 0
    _FREEZE_BACKBONE = False
    _FREEZE_BN = True  # Freeze batch norm
    _SEGMENTER_DIR = ""
    _SEGMENTER_IDENT = ""

    # Learning rate
    _END_LEARNING_RATE = 0
    _POWER = 0.9

    # Optimizer
    _MOMENTUM = 0.9
    _BATCH_NORM_DECAY = 0.9997
    _WEIGHT_DECAY = 1e-4

    # Embedding Function configuration
    _OUTPUT_STRIDE = 16
    _DECODER_OUPUT_STRIDE = 4
    _BACKBONE = 'resnet_v2_101'
    _EFN_OUT_DIM = 256  # embedding function output dimension

    # dataset specific
    _NUM_EPOCHS = None
    _INITIAL_LEARNING_RATE = None
    _NUM_TRAIN = None

    @property
    def _MAX_ITER(self, ):
        """ Compute lr max iter based on own params. """
        return self._NUM_TRAIN * self._NUM_EPOCHS / self._BATCH_SIZE
