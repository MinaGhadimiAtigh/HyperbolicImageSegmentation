"""Utility functions for preprocessing data sets."""

import os

import numpy as np
import tensorflow as tf
from PIL import Image

# colour map
label_colours_file = '/'.join(__file__.split('/')[:-1] + ['label_colours.npy'])
_LABEL_COLOURS = np.load(label_colours_file)


def decode_labels(mask, num_images=1, num_classes=21):
    """Decode batch of segmentation masks.

    Args:
        mask: result of inference after taking argmax.
        num_images: number of images to decode from the batch.
        num_classes: number of classes to predict (including background).

    Returns:
        A batch with num_images RGB images of the same size as the input.
    """
    n, h, w, c = mask.shape
    assert n >= num_images, (
            "Batch size %d should be greater or equal than number of images to save %d."
            % (n, num_images)
    )
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
        img = Image.new("RGB", (len(mask[i, 0]), len(mask[i])))
        pixels = img.load()
        for j_, j in enumerate(mask[i, :, :, 0]):
            for k_, k in enumerate(j):
                if k < num_classes:
                    try:
                        pixels[k_, j_] = tuple(_LABEL_COLOURS[k])
                    except Exception as e:
                        print(k_, j_, len(_LABEL_COLOURS), k)  # 0 0 63 157

                        raise ValueError
        outputs[i] = np.array(img)
    return outputs


def mean_image_addition(image, means):
    """Adds the given means from each image channel.

    For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.

    Returns:
        the centered image.

    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        if image.get_shape().ndims == 4:
            raise ValueError("Input must be of size [height, width, C>0], got size 4")
        if image.get_shape().ndims == 2:
            raise ValueError("Input must be of size [height, width, C>0], got size 2")
        raise ValueError(
            f"Input must be of size [height, width, C>0], got size thats isnt 4 or 2: {image.get_shape()}"
        )
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError("len(means) must match the number of channels")

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] += means[i]
    return tf.concat(axis=2, values=channels)


def mean_image_subtraction(image, means):
    """Subtracts the given means from each image channel.

    For example:
        means = [123.68, 116.779, 103.939]
        image = _mean_image_subtraction(image, means)

    Note that the rank of `image` must be known.

    Args:
        image: a tensor of size [height, width, C].
        means: a C-vector of values to subtract from each channel.

    Returns:
        the centered image.

    Raises:
        ValueError: If the rank of `image` is unknown, if `image` has a rank other
        than three or if the number of channels in `image` doesn't match the
        number of values in `means`.
    """
    if image.get_shape().ndims != 3:
        raise ValueError("Input must be of size [height, width, C>0]")
    num_channels = image.get_shape().as_list()[-1]
    if len(means) != num_channels:
        raise ValueError("len(means) must match the number of channels")

    channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
    for i in range(num_channels):
        channels[i] -= means[i]
    return tf.concat(axis=2, values=channels)


def random_rescale_image_and_label(image, label, min_scale, max_scale):
    """Rescale an image and label with in target scale.

    Rescales an image and label within the range of target scale.

    Args:
        image: 3-D Tensor of shape `[height, width, channels]`.
        label: 3-D Tensor of shape `[height, width, 1]`.
        min_scale: Min target scale.
        max_scale: Max target scale.

    Returns:
        Cropped and/or padded image.
        If `images` was 3-D, a 3-D float Tensor of shape
        `[new_height, new_width, channels]`.
        If `labels` was 3-D, a 3-D float Tensor of shape
        `[new_height, new_width, 1]`.
    """
    if min_scale <= 0:
        raise ValueError("'min_scale' must be greater than 0.")
    elif max_scale <= 0:
        raise ValueError("'max_scale' must be greater than 0.")
    elif min_scale >= max_scale:
        raise ValueError("'max_scale' must be greater than 'min_scale'.")

    shape = tf.shape(image)
    height = tf.to_float(shape[0])
    width = tf.to_float(shape[1])
    scale = tf.random_uniform([], minval=min_scale, maxval=max_scale, dtype=tf.float32)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)
    image = tf.image.resize_images(
        image, [new_height, new_width], method=tf.image.ResizeMethod.BILINEAR
    )
    # Since label classes are integers, nearest neighbor need to be used.
    label = tf.image.resize_images(
        label, [new_height, new_width], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )

    return image, label


def fit_image_and_label(image, label, ignore_label):
    """FOR evaluation, we only need to make sure that the input is square (due to the row norm). So just pad to a square

    Resizes an image to a target width and height by rondomly
    cropping the image or padding it evenly with zeros.

    Args:
        image: 3-D Tensor of shape `[height, width, channels]`.
        label: 3-D Tensor of shape `[height, width, 1]`.
        crop_height: The new height.
        crop_width: The new width.
        ignore_label: Label class to be ignored.

    Returns:
        Cropped and/or padded image.
        If `images` was 3-D, a 3-D float Tensor of shape
        `[new_height, new_width, channels]`.
    """
    pad = False  # pad if network only accepts square inputs
    if pad:
        label = label - ignore_label  # Subtract due to 0 padding.
        label = tf.to_float(label)
        image_height = tf.shape(image)[0]
        image_width = tf.shape(image)[1]

        target = tf.maximum(image_height, image_width)

        image_and_label = tf.concat([image, label], axis=2)
        image_and_label_pad = tf.image.pad_to_bounding_box(
            image_and_label,
            0,
            0,
            target,  # tf.maximum(crop_height, new_height),
            target,
        )  # tf.maximum(crop_width, new_width))
        image_and_label_crop = tf.random_crop(image_and_label_pad, [target, target, 4])

        image_crop = image_and_label_crop[:, :, :3]
        label_crop = image_and_label_crop[:, :, 3:]
        label_crop += ignore_label
        label_crop = tf.to_int32(label_crop)

        return image_crop, label_crop
    else:
        return image, label


def random_crop_or_pad_image_and_label(
        image, label, crop_height, crop_width, ignore_label
):
    """Crops and/or pads an image to a target width and height.

    Resizes an image to a target width and height by rondomly
    cropping the image or padding it evenly with zeros.

    Args:
        image: 3-D Tensor of shape `[height, width, channels]`.
        label: 3-D Tensor of shape `[height, width, 1]`.
        crop_height: The new height.
        crop_width: The new width.
        ignore_label: Label class to be ignored.

    Returns:
        Cropped and/or padded image.
        If `images` was 3-D, a 3-D float Tensor of shape
        `[new_height, new_width, channels]`.
    """
    label = label - ignore_label  # Subtract due to 0 padding.
    label = tf.to_float(label)
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]
    image_and_label = tf.concat([image, label], axis=2)
    image_and_label_pad = tf.image.pad_to_bounding_box(
        image_and_label,
        0,
        0,
        tf.maximum(crop_height, image_height),
        tf.maximum(crop_width, image_width),
    )
    image_and_label_crop = tf.random_crop(
        image_and_label_pad, [crop_height, crop_width, 4]
    )

    image_crop = image_and_label_crop[:, :, :3]
    label_crop = image_and_label_crop[:, :, 3:]
    label_crop += ignore_label
    label_crop = tf.to_int32(label_crop)

    return image_crop, label_crop


def random_flip_left_right_image_and_label(image, label):
    """Randomly flip an image and label horizontally (left to right).

    Args:
        image: A 3-D tensor of shape `[height, width, channels].`
        label: A 3-D tensor of shape `[height, width, 1].`

    Returns:
        A 3-D tensor of the same type and shape as `image`.
        A 3-D tensor of the same type and shape as `label`.
    """
    uniform_random = tf.random_uniform([], 0, 1.0)
    mirror_cond = tf.less(uniform_random, 0.5)
    image = tf.cond(mirror_cond, lambda: tf.reverse(image, [1]), lambda: image)
    label = tf.cond(mirror_cond, lambda: tf.reverse(label, [1]), lambda: label)

    return image, label


def eval_input_fn(image_filenames, config, label_filenames=None, batch_size=1):
    """An input function for evaluation and inference.

    Args:
        image_filenames: The file names for the inferred images.
        label_filenames: The file names for the grand truth labels.
        batch_size: The number of samples per batch. Need to be 1
            for the images of different sizes.

    Returns:
        A tuple of images and labels.
    """

    # Reads an image from a file, decodes it into a dense tensor
    def _parse_function(filename, is_label):
        if not is_label:
            image_filename, label_filename = filename, None
        else:
            image_filename, label_filename = filename

        image_string = tf.read_file(image_filename)
        image = tf.image.decode_image(image_string)
        image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
        image.set_shape([None, None, 3])

        image = mean_image_subtraction(image, means=config.dataset._RGB_MEANS)

        if not is_label:
            return image
        else:
            label_string = tf.read_file(label_filename)
            label = tf.image.decode_image(label_string)
            label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
            label.set_shape([None, None, 1])
            # image , label = fit_image_and_label(image, label, 255)
            return image, label

    if label_filenames is None:
        input_filenames = image_filenames
    else:
        input_filenames = (image_filenames, label_filenames)

    dataset = tf.data.Dataset.from_tensor_slices(input_filenames)
    if label_filenames is None:
        dataset = dataset.map(lambda x: _parse_function(x, False))
    else:
        dataset = dataset.map(lambda x, y: _parse_function((x, y), True))
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)
    iterator = dataset.make_one_shot_iterator()

    if label_filenames is None:
        images = iterator.get_next()
        labels = None
    else:
        images, labels = iterator.get_next()

    return images, labels


def get_filenames(is_training, config):
    """Return a list of filenames.

    Args:
      is_training: A boolean denoting whether the input is for training.
      data_dir: path to the the directory containing the input data.

    Returns:
      A list of file names.
    """
    if "pascal" in config.dataset._NAME:
        prefix = "voc_"
    elif (("coco" in config.dataset._NAME) or ("COCO" in config.dataset._NAME)):
        prefix = "coco_"
    elif ("ADE20K" in config.dataset._NAME) or ("ade" in config.dataset._NAME) or ("ADE" in config.dataset._NAME):
        prefix = "ade_"
    data_dir = config.dataset._DATA_DIR
    if is_training:
        # coco_train.record
        return [os.path.join(data_dir, f"{prefix}train.record")]
    else:
        return [os.path.join(data_dir, f"{prefix}val.record")]  # coco_val.record


def parse_record(raw_record):
    """Parse PASCAL image and label from a tf record."""
    keys_to_features = {
        "image/height": tf.FixedLenFeature((), tf.int64),
        "image/width": tf.FixedLenFeature((), tf.int64),
        "image/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "image/format": tf.FixedLenFeature((), tf.string, default_value="jpeg"),
        "label/encoded": tf.FixedLenFeature((), tf.string, default_value=""),
        "label/format": tf.FixedLenFeature((), tf.string, default_value="png"),
    }

    parsed = tf.parse_single_example(raw_record, keys_to_features)

    # height = tf.cast(parsed['image/height'], tf.int32)
    # width = tf.cast(parsed['image/width'], tf.int32)

    image = tf.image.decode_image(tf.reshape(parsed["image/encoded"], shape=[]), 3)
    image = tf.to_float(tf.image.convert_image_dtype(image, dtype=tf.uint8))
    image.set_shape([None, None, 3])

    label = tf.image.decode_image(tf.reshape(parsed["label/encoded"], shape=[]), 1)
    label = tf.to_int32(tf.image.convert_image_dtype(label, dtype=tf.uint8))
    label.set_shape([None, None, 1])

    return image, label


def preprocess_image(image, label, is_training, config):
    """Preprocess a single image of layout [height, width, depth]."""
    if is_training:
        # Randomly scale the image and label.
        image, label = random_rescale_image_and_label(
            image, label, config.segmenter._MIN_SCALE, config.segmenter._MAX_SCALE
        )
        # new_height = new_width = 513
        ##image = tf.image.resize_images(image, [new_height, new_width],
        #                             method=tf.image.ResizeMethod.BILINEAR)
        # Since label classes are integers, nearest neighbor need to be used.
        # label = tf.image.resize_images(label, [new_height, new_width],
        #                             method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # Randomly crop or pad a [_HEIGHT, _WIDTH] section of the image and label.
        image, label = random_crop_or_pad_image_and_label(
            image, label, config.segmenter._HEIGHT, config.segmenter._WIDTH, config.segmenter._IGNORE_LABEL
        )

        # Randomly flip the image and label horizontally.
        image, label = random_flip_left_right_image_and_label(
            image, label
        )

        image.set_shape([config.segmenter._HEIGHT, config.segmenter._WIDTH, 3])
        label.set_shape([config.segmenter._HEIGHT, config.segmenter._WIDTH, 1])

    image = mean_image_subtraction(image, config.dataset._RGB_MEANS)

    return image, label
