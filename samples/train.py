import argparse
import logging

from hesp.config.config import Config
from hesp.config.dataset_config import DATASET_CFG_DICT
from hesp.models.model import ModelFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train models."
    )

    parser.add_argument(
        '--mode',
        type=str,
        choices=['segmenter'],
        help="Whether to train a segmenter."
    )

    parser.add_argument(
        '--base_model',
        type=str,
        choices=['deeplab'],
        help="Choose the base model for the embedding function."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        choices=DATASET_CFG_DICT.keys(),
        help="Which dataset to use",
        required=True,
    )

    parser.add_argument(
        "--geometry",
        type=str,
        choices=['euclidean', 'hyperbolic'],
        help="Type of geometry to use",
        default="hyperbolic",
    )

    parser.add_argument(
        "--dim",
        type=int,
        help="Dimensionality of embedding space.",
        default=256,
    )

    parser.add_argument(
        "--c",
        type=float,
        help="Initial curvature of hyperbolic space",
        default=1.,
    )

    parser.add_argument(
        "--flat",
        action='store_true',
        help="Disables hierarchical structure when set.",
    )

    parser.add_argument(
        "--freeze_bb",
        action='store_true',
        help="Freeze backbone.",
    )

    parser.add_argument(
        "--freeze_bn",
        action='store_true',
        help="Freeze batch normalization.",
    )

    parser.add_argument(
        "--batch_size",
        default=5,
        type=int,
        help="Batch size."
    )

    parser.add_argument(
        "--num_epochs",
        default=0,
        type=int,
        help="Number of epochs to train."
    )

    parser.add_argument(
        "--slr",
        default=0,
        type=float,
        help="Initial learning rate."
    )

    parser.add_argument(
        "--backbone",
        choices=['resnet_v2_101', 'resnet_v2_50'],
        default='resnet_v2_101',
        help="Backbone architecture."
    )

    parser.add_argument(
        "--backbone_init",
        default="../ini_checkpoints/resnet_v2_101/resnet_v2_101.ckpt",
        type=str,
        help="Backbone initialization weights."
    )

    parser.add_argument(
        "--output_stride",
        type=int,
        choices=[8, 16],
        default=16,
        help="Backbone output stride."
    )

    parser.add_argument(
        "--gpu",
        type=int,
        default=0,
        help="Which GPU to use, in case of multi-gpu system and parallel training."
    )

    parser.add_argument(
        "--base_save_dir",
        type=str,
        default="",
        help="base dir used for saving models."
    )

    parser.add_argument(
        "--segmenter_dir",
        type=str,
        default="segmenter",
        help="prefix of the directory the experiments are going to be saved."
    )

    parser.add_argument(
        "--segmenter_ident",
        type=str,
        default="",
        help="add a suffix to segmenter ident"
    )

    parser.add_argument(
        "--zero_label",
        action='store_true',
        help='whether do zero label training.'
    )

    parser.add_argument(
        "--test_zero_label",
        action='store_true',
        help='whether to perform zero label testing.'
    )

    parser.add_argument(
        "--train",
        action='store_true',
        help='whether to perform testing after training.'
    )

    parser.add_argument(
        "--test",
        action='store_true',
        help='whether to perform testing after training.'
    )

    parser.add_argument(
        "--nomatch",
        action='store_true',
        help='Segmenter: Dont match embedding function output dim to embedding space. Emb fn. then returns 256 '
             'and an extra 1x1 is added to match dimensions.'
    )

    parser.add_argument(
        "--id",
        type=str,
        default="",
        help="Optional identifier for run"
    )

    parser.add_argument(
        "--json_name",
        type=str,
        default="",
        help="Needed When using ADE20K dataset. Select the hierarchy you want to use."
    )

    args = parser.parse_args()

    logger.info('Initializing configuration.')
    config = Config(dataset=args.dataset, base_save_dir=args.base_save_dir, gpu_idx=args.gpu, mode=args.mode,
                    json_selection=args.json_name)

    if not args.mode:
        logger.error('Need to specify mode!')
        raise ValueError

    config._IDENT = args.id

    config.embedding_space._GEOMETRY = args.geometry
    config.embedding_space._DIM = args.dim
    config.embedding_space._INIT_CURVATURE = args.c
    config.embedding_space._HIERARCHICAL = not args.flat

    config.base_model_name = args.base_model

    if args.mode == 'segmenter':
        if args.backbone_init is None:
            logger.warning('Please submit empty string if you dont want backbone pretrained weights! Save guard.')
            raise ValueError

        config.segmenter._PRETRAINED_MODEL = args.backbone_init
        config.segmenter._OUTPUT_STRIDE = args.output_stride
        config.segmenter._BACKBONE = args.backbone
        config.segmenter._BATCH_SIZE = args.batch_size
        config.segmenter._FREEZE_BACKBONE = args.freeze_bb
        config.segmenter._FREEZE_BN = args.freeze_bn
        config.segmenter._ZERO_LABEL = args.zero_label
        config.segmenter._SEGMENTER_DIR = args.segmenter_dir
        config.segmenter._SEGMENTER_IDENT = args.segmenter_ident

        if not args.num_epochs:
            config.segmenter._NUM_EPOCHS = config.dataset._NUM_EPOCHS
        else:
            config.segmenter._NUM_EPOCHS = args.num_epochs

        if not args.slr:
            config.segmenter._INITIAL_LEARNING_RATE = config.dataset._INITIAL_LEARNING_RATE
        else:
            config.segmenter._INITIAL_LEARNING_RATE = args.slr
        config.segmenter._NUM_TRAIN = config.dataset._NUM_TRAIN

        if not args.nomatch:
            config.segmenter._EFN_OUT_DIM = args.dim

    logger.info('Initializing model.')
    model = ModelFactory.create(mode=args.mode, config=config)
    if args.train:
        logger.info('Starting training.')
        model.train()
        logger.info('Training complete!')
    if args.test:
        logger.info('Starting testing...')
        model.test()
        if args.mode == 'segmenter' and args.test_zero_label:
            model.config.segmenter._TEST_ZERO_LABEL = True
            model.test()
        logger.info('Testing done!')
    logger.info('Exiting.')
