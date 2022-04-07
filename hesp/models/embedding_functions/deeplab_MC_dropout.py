"""DeepLab v3 models based on slim library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
from tensorflow.contrib.framework.python.ops import arg_scope
from tensorflow.contrib.layers.python.layers import layers

from hesp.models.embedding_functions import resnet_v2_MC as resnet_v2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def atrous_spatial_pyramid_pooling(
        inputs, output_stride, batch_norm_decay, is_training, depth=256
):
    """Atrous Spatial Pyramid Pooling.

    Args:
      inputs: A tensor of size [batch, height, width, channels].
      output_stride: The ResNet unit's stride. Determines the rates for atrous convolution.
        the rates are (6, 12, 18) when the stride is 16, and doubled when 8.
      batch_norm_decay: The moving average decay when estimating layer activation
        statistics in batch normalization.
      is_training: A boolean denoting whether the input is for training.
      depth: The depth of the ResNet unit output.

    Returns:
      The atrous spatial pyramid pooling output.
    """
    with tf.variable_scope("aspp"):
        if output_stride not in [8, 16]:
            raise ValueError("output_stride must be either 8 or 16.")

        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [2 * rate for rate in atrous_rates]

        with tf.contrib.slim.arg_scope(
                resnet_v2.resnet_arg_scope(batch_norm_decay=batch_norm_decay)
        ):
            with arg_scope([layers.batch_norm], is_training=is_training):
                with arg_scope(
                        [layers_lib.conv2d], outputs_collections=["aspp_collection"]
                ):
                    inputs_size = tf.shape(inputs)[1:3]
                    # (a) one 1x1 convolution and three 3x3 convolutions with rates = (6, 12, 18) when output stride = 16.
                    # the rates are doubled when output stride = 8.
                    conv_1x1 = layers_lib.conv2d(
                        inputs, depth, [1, 1], stride=1, scope="conv_1x1"
                    )
                    conv_3x3_1 = layers_lib.conv2d(
                        inputs,
                        depth,
                        [3, 3],
                        stride=1,
                        rate=atrous_rates[0],
                        scope="conv_3x3_1",
                    )
                    conv_3x3_2 = layers_lib.conv2d(
                        inputs,
                        depth,
                        [3, 3],
                        stride=1,
                        rate=atrous_rates[1],
                        scope="conv_3x3_2",
                    )
                    conv_3x3_3 = layers_lib.conv2d(
                        inputs,
                        depth,
                        [3, 3],
                        stride=1,
                        rate=atrous_rates[2],
                        scope="conv_3x3_3",
                    )

                    # (b) the image-level features
                    with tf.variable_scope("image_level_features"):
                        # global average pooling
                        image_level_features = tf.reduce_mean(
                            inputs, [1, 2], name="global_average_pooling", keepdims=True
                        )
                        tf.add_to_collection("aspp_collection", image_level_features)

                        # 1x1 convolution with 256 filters( and batch
                        # normalization)
                        image_level_features = layers_lib.conv2d(
                            image_level_features,
                            depth,
                            [1, 1],
                            stride=1,
                            scope="conv_1x1",
                        )

                        # bilinearly upsample features
                        image_level_features = tf.image.resize_bilinear(
                            image_level_features, inputs_size, name="upsample"
                        )

                    net = tf.concat(
                        [
                            conv_1x1,
                            conv_3x3_1,
                            conv_3x3_2,
                            conv_3x3_3,
                            image_level_features,
                        ],
                        axis=3,
                        name="concat",
                    )
                    tf.add_to_collection("aspp_collection", net)

                    net = layers_lib.conv2d(
                        net, depth, [1, 1], stride=1, scope="conv_1x1_concat"
                    )
                    # net = layers_lib.dropout(net, .9, is_training=is_training)
                    return net


def deeplab_v3_plus(config):
    """Generator for DeepLab v3 plus models.

    Args:
      config object holding cfg params.
    Returns:
      The model function that takes in `inputs` and `is_training` and
      returns the output tensor of the DeepLab v3 model.
    """
    if config._BACKBONE not in ["resnet_v2_50", "resnet_v2_101"]:
        raise ValueError(
            "'base_architrecture' must be either 'resnet_v2_50' or 'resnet_v2_101'"
        )

    if config._BACKBONE == "resnet_v2_50":
        base_model = resnet_v2.resnet_v2_50
    else:
        base_model = resnet_v2.resnet_v2_101

    def model(inputs, is_training):
        """Constructs model callable."""

        with tf.contrib.slim.arg_scope(
                resnet_v2.resnet_arg_scope(batch_norm_decay=config._BATCH_NORM_DECAY)
        ):
            logits, end_points = base_model(
                inputs,
                num_classes=None,
                is_training=is_training,
                global_pool=False,
                output_stride=config._OUTPUT_STRIDE,
            )

        if is_training:
            exclude = [config._BACKBONE + "/logits", "global_step"]
            if config._PRETRAINED_MODEL is None:
                logger.error('Backbone init is None! This should never happen.')
                raise ValueError
            if config._PRETRAINED_MODEL:
                variables_to_restore = tf.contrib.slim.get_variables_to_restore(
                    exclude=exclude
                )
                variables_to_restore = [v for v in variables_to_restore if config._BACKBONE in v.name]
                logger.debug('Backbone variables being restored:')
                for v in variables_to_restore:
                    logger.debug(v.name)
                variables_to_restore = [v for v in variables_to_restore if 'r']
                tf.train.init_from_checkpoint(
                    config._PRETRAINED_MODEL,
                    {
                        v.name.split(":")[0]: v
                        for v in variables_to_restore
                    },
                )

        net = end_points[config._BACKBONE + "/block4"]

        encoder_output = atrous_spatial_pyramid_pooling(
            net, config._OUTPUT_STRIDE, config._BATCH_NORM_DECAY, is_training
        )

        with tf.variable_scope("decoder"):
            with tf.contrib.slim.arg_scope(
                    resnet_v2.resnet_arg_scope(batch_norm_decay=config._BATCH_NORM_DECAY)
            ):
                with arg_scope([layers.batch_norm], is_training=is_training):
                    with arg_scope(
                            [layers_lib.conv2d], outputs_collections=["decoder_collection"]
                    ):
                        with tf.variable_scope("low_level_features"):
                            # Determines output stride of the decoder
                            # source: official deeplab implementation from paper
                            # 'resnet_v1_101/ 50': {
                            DECODER_END_POINTS = {
                                4: "/block1/unit_2/bottleneck_v2/conv3",
                                8: "/block2/unit_3/bottleneck_v2/conv3",
                                16: "/block3/unit_22/bottleneck_v2/conv3",
                            }

                            decoder_output_stride = config._DECODER_OUPUT_STRIDE

                            low_level_features = end_points[
                                config._BACKBONE
                                + DECODER_END_POINTS[decoder_output_stride]
                                ]

                            low_level_features = layers_lib.conv2d(
                                low_level_features,
                                48,
                                [1, 1],
                                stride=1,
                                scope="conv_1x1",
                            )
                            low_level_features_size = tf.shape(low_level_features)[1:3]

                        with tf.variable_scope("upsampling_logits"):
                            net = tf.image.resize_bilinear(
                                encoder_output,
                                low_level_features_size,
                                name="upsample_1",
                            )
                            net = tf.concat(
                                [net, low_level_features], axis=3, name="concat"
                            )
                            net = layers_lib.conv2d(
                                net, 256, [3, 3], stride=1, scope="conv_3x3_1"
                            )
                            net = layers_lib.conv2d(
                                net, config._EFN_OUT_DIM, [3, 3], stride=1, scope='conv_3x3_2', activation_fn=None,
                                normalizer_fn=None)

        return net

    return model
