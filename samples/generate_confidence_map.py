import math
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from hesp.config.config import Config
from hesp.models.model import ModelFactory

# HERE I'm loading the model I have trained.
# To load the model, you need to set the hyperparameters and arguments based on the trained model
# ================================================================================================
dataset = 'pascal'
mode = 'segmenter'
config = Config(dataset=dataset, base_save_dir="", gpu_idx=0, mode=mode)

config.embedding_space._GEOMETRY = 'hyperbolic'
config.embedding_space._DIM = 256
config.embedding_space._INIT_CURVATURE = 2.0
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

model.embedding_space.normals_npy = normals_npy
model.embedding_space.offsets_npy = offsets_npy
model.embedding_space.c_npy = np.array(c_value, dtype=np.float32)

origin_path = "path to images"
image_paths = []
image_list_file = "../datasets/pascal/pascal_data/all.txt"  # PATH to LIST OF IMAGES
with open(image_list_file) as f:
    content = f.readlines()
    for l in content:
        image_paths.append(l.split('\n')[0])

for image_path in image_paths:
    img = cv2.cvtColor(cv2.imread(os.path.join(origin_path, image_path + '.jpg')), cv2.COLOR_BGR2RGB)

    model.embedding_space.normals_npy = normals_npy
    model.embedding_space.offsets_npy = offsets_npy
    model.embedding_space.c_npy = np.array(c_value, dtype=np.float32)

    results = model.predict(img.astype(np.float32), np.zeros_like(img))
    confidence_map = np.linalg.norm(results['embeddings'], axis=-1)
    radius = 1.0 / math.sqrt(config.embedding_space._INIT_CURVATURE)
    normalized_confidence_map = confidence_map / radius

    plt.imshow(confidence_map, cmap='gist_earth')
    plt.show()
