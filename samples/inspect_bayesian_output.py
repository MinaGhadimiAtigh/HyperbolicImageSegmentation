import os

import matplotlib.image as image
import numpy as np
import tensorflow as tf

from hesp.config.config import Config
from hesp.models.model import ModelFactory

# HERE I'm loading the model I have trained.
# To load the model, you need to set the hyperparameters and arguments based on the trained model
# ================================================================================================
dataset = 'coco'
mode = 'segmenter'
config = Config(dataset=dataset, base_save_dir="", gpu_idx=0, mode=mode)

config.embedding_space._GEOMETRY = 'hyperbolic'
config.embedding_space._DIM = 2
config.embedding_space._INIT_CURVATURE = 0.1
config.embedding_space._HIERARCHICAL = True

# Resnet weights
config.segmenter._PRETRAINED_MODEL = "PATH TO THE BASE MODEL PRETRAINED WEIGHTS"

config.segmenter._PROTOTYPE = False
config.segmenter._OUTPUT_STRIDE = 16
config.segmenter._BACKBONE = 'resnet_v2_101'
config.segmenter._BATCH_SIZE = 5
config.segmenter._FREEZE_BACKBONE = False
config.segmenter._FREEZE_BN = True
config.segmenter._NUM_EPOCHS = config.dataset._NUM_EPOCHS
config.segmenter._NUM_TRAIN = config.dataset._NUM_TRAIN
config.visualizer._IPF = -1

config.segmenter._SEGMENTER_IDENT = "SEGMENTER IDENT USED WHEN TRAINING MODEL"

config.segmenter._INITIAL_LEARNING_RATE = config.dataset._INITIAL_LEARNING_RATE
nomatch = False
if not nomatch:
    config.segmenter._EFN_OUT_DIM = config.embedding_space._DIM

# ================================================================================================

model = ModelFactory.create(mode=mode, config=config)
model._init_predict()

if len([f for f in os.listdir(config._SEGMENTER_SAVE_DIR) if
        os.path.isfile(os.path.join(config._SEGMENTER_SAVE_DIR, f))]) == 0:
    print("There is no model.")
    raise Exception('There is no model.')

model_dir = config._SEGMENTER_SAVE_DIR

latest_checkpoint = tf.train.latest_checkpoint(model_dir)
offsets_npy = tf.train.load_variable(latest_checkpoint, 'offsets')
normals_npy = tf.train.load_variable(latest_checkpoint, 'normals')
c_value = tf.train.load_variable(latest_checkpoint, 'curvature')

image_paths = ["PATH TO IMAGES"]

total_count = 1000
for image_path in image_paths:

    for i in range(total_count):
        img = image.imread(image_path)

        model.embedding_space.normals_npy = normals_npy
        model.embedding_space.offsets_npy = offsets_npy
        model.embedding_space.c_npy = np.array(c_value, dtype=np.float32)

        results = model.predict(img.astype(np.float32), np.zeros_like(img))

        predictions_array = results['predictions']
        probabilities_array = results['probabilities']

        if i == 0:
            sum_prob_over_T = np.zeros(probabilities_array.shape)
        sum_prob_over_T += probabilities_array

    sum_prob_over_T = sum_prob_over_T / float(total_count)
    Entropy = np.sum(sum_prob_over_T * np.log(sum_prob_over_T), axis=-1)

    np.save(image_path.split('/')[-1].split('.')[0] + '_Sum_dim' + str(config.embedding_space._DIM) + '.npy',
            sum_prob_over_T)
    np.save(image_path.split('/')[-1].split('.')[0] + '_Entropy_dim' + str(config.embedding_space._DIM) + '.npy',
            Entropy)
# TODO check the output
