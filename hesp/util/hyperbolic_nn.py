"""
Atomic hyperbolic neural networks operatos. Mostly adapted or copied from the 
Hyperbolic Neural Networks, (Ganea et. al. 2018) author implementation:
https://github.com/dalab/hyperbolic_nn
"""

import logging

import numpy as np
import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# hyperparams

PROJ_EPS = 1e-3
EPS = 1e-15
MAX_TANH_ARG = 15.0


### HYPERBOLIC FUNCTIONS
def tf_lambda_x(x, c):
    return 2.0 / (1 - c * tf_dot(x, x))


def tf_dot(x, y):
    return tf.reduce_sum(x * y, axis=1, keepdims=True)


def tf_norm(x):
    return tf.norm(x, axis=1, keepdims=True)


## GRADIENT FUNCTIONS ###

def riemannian_gradient_c(u, c):
    return ((1.0 - c * tf_dot(u, u)) ** 2) / 4.0


## HELPER FUNCTIONS ###

def tf_project_hyp_vecs(x, c, axes=[-1]):
    # Projection op. Need to make sure hyperbolic embeddings are inside the unit ball.
    return tf.clip_by_norm(t=x, clip_norm=(1.0 - PROJ_EPS) / tf.sqrt(c), axes=axes)


def tf_atanh(x):
    return tf.atanh(tf.minimum(x, 1.0 - EPS))  # Only works for positive real x.


def tf_tanh(x):
    return tf.tanh(tf.minimum(tf.maximum(x, -MAX_TANH_ARG), MAX_TANH_ARG))


def np_project(x, c):
    max_norm = (1.0 - PROJ_EPS) / np.sqrt(c)
    old_norms = np.linalg.norm(x, axis=-1)
    clip_idx = old_norms > max_norm
    old_x = x.copy()
    x[clip_idx, :] /= (np.linalg.norm(x, axis=-1, keepdims=True)[clip_idx, :]) / max_norm
    return x


def exp_map_zero(inputs, c):
    inputs = inputs + EPS
    norm = np.linalg.norm(inputs, axis=-1)
    gamma = np.tanh(np.minimum(np.maximum(np.sqrt(c) * norm, -MAX_TANH_ARG), MAX_TANH_ARG)) / (np.sqrt(c) * norm)
    scaled_inputs = gamma[:, None] * inputs
    return np_project(scaled_inputs, c)


def tf_exp_map_zero(inputs, c):
    # exp map of a [n,d] vector
    sqrt_c = tf.sqrt(c)
    inputs = inputs + EPS
    norm = tf.norm(inputs, axis=-1)  # protect div b 0
    #
    gamma = tf.divide(tf_tanh(sqrt_c * norm), (sqrt_c * norm))  # sh ncls
    # gamma = tf.Print(gamma,[tf.reduce_mean(gamma)],'gamma')
    scaled_inputs = gamma[..., None] * inputs
    # scaled_inputs = tf.Print(scaled_inputs,[tf.reduce_mean(scaled_inputs)],'scaled inputs')
    return tf_project_hyp_vecs(scaled_inputs, c, axes=[-1])


def tf_log_map_zero_batch(y, c):
    diff = y
    norm_diff = tf.maximum(tf.norm(y, axis=-1, keepdims=True), EPS)  # b,H,W,1
    return 1.0 / tf.sqrt(c) * atanh(tf.sqrt(c) * norm_diff) / norm_diff * diff


def tf_sqnorm(u, keepdims=True, axis=-1):
    # performs sq norm over last dim
    return tf.reduce_sum(u * u, axis=axis, keepdims=keepdims)


def tf_exp_map_x(x, v, c):
    v = v + EPS  # Perturbe v to avoid dealing with v = 0
    norm_v = tf_norm(v)
    second_term = (
                          tf_tanh(tf.sqrt(c) * tf_lambda_x(x, c) * norm_v / 2) / (tf.sqrt(c) * norm_v)
                  ) * v
    return tf_mob_add(x, second_term, c)


def tf_mob_add(u, v, c):
    v = v + EPS
    tf_dot_u_v = 2.0 * c * tf_dot(u, v)
    tf_norm_u_sq = c * tf_dot(u, u)
    tf_norm_v_sq = c * tf_dot(v, v)
    denominator = 1.0 + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
    result = (1.0 + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (
            1.0 - tf_norm_u_sq
    ) / denominator * v
    return tf_project_hyp_vecs(result, c)


def tf_mob_add_batch(u, v, c):
    # adds two feature batches of shape [B,H,W,D]
    v = v + EPS
    tf_dot_u_v = (
            2.0 * c * tf.reduce_sum(u * v, axis=-1, keepdims=True)
    )  # B,H,W,1 #tf_dot(u, v)
    tf_norm_u_sq = c * tf.reduce_sum(u * u, axis=-1, keepdims=True)
    tf_norm_v_sq = c * tf.reduce_sum(v * v, axis=-1, keepdims=True)

    denominator = 1.0 + tf_dot_u_v + tf_norm_v_sq * tf_norm_u_sq
    result = (1.0 + tf_dot_u_v + tf_norm_v_sq) / denominator * u + (
            1.0 - tf_norm_u_sq
    ) / denominator * v
    return tf_project_hyp_vecs(result, c, axes=[-1])


def atanh(inputs):
    x = tf.clip_by_value(inputs, -1 + EPS, 1 - EPS)
    res = tf.log(1 + x) - tf.log(1 - x)
    res = tf.multiply(res, 0.5)
    return res
