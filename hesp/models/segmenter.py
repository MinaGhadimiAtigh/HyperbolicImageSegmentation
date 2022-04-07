import json as json_lib
import logging
import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers as layers_lib
from tensorflow.python.ops.metrics_impl import _streaming_confusion_matrix

from hesp.config.config import Config
from hesp.embedding_space.embedding_space import EmbeddingSpace
from hesp.hierarchy.tree import Tree
from hesp.models.abstract_model import AbstractModel
from hesp.models.embedding_functions.deeplab import deeplab_v3_plus
# IF YOU WANT TO TRAIN BAYESIAN, UNCOMMENT THIS
# from hesp.models.embedding_functions.deeplab_MC_dropout import deeplab_v3_plus
from hesp.util.data_helpers import parse_record, get_filenames, preprocess_image, mean_image_addition, decode_labels
from hesp.util.layers import get_hyp_update_ops
from hesp.util.loss import CCE
from hesp.util.metrics import compute_mean_iou, tf_metrics, cls_mean_iou_npy, hierarchical_cm, npy_metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Segmenter(AbstractModel):
    def __init__(self, tree: Tree, config: Config, train_embedding_space: bool, prototype_path: str):
        self.train_embedding_space = train_embedding_space
        self.prototype_path = prototype_path
        self.tree = tree
        self.config = config
        self.embedding_space = None

    def init_embedding_space(self, ):
        # need to init embedding space object inside model_fn, or estimator wont play nice.
        return EmbeddingSpace(tree=self.tree, config=self.config, train=self.train_embedding_space,
                              prototype_path=self.prototype_path)

    @property
    def embedding_function(self, ):
        """ Returns embedding function. """
        return deeplab_v3_plus(self.config.segmenter)

    def get_train_op(self, loss):
        logger.info('Defining train op.')
        global_step = tf.train.get_or_create_global_step()

        if not self.config.segmenter._FREEZE_BN:
            train_var_list = [v for v in tf.trainable_variables()]
        else:
            bb_bn_vars = [v for v in tf.trainable_variables() if
                          (self.config.segmenter._BACKBONE in v.name) and ("beta" in v.name or "gamma" in v.name)]
            # train_var_list = [
            #    v for v in tf.trainable_variables() if ("beta" not in v.name) and ("gamma" not in v.name)
            # ]
            train_var_list = [v for v in tf.trainable_variables() if v not in bb_bn_vars]
        if self.config.segmenter._FREEZE_BACKBONE:
            train_var_list = [v for v in train_var_list if self.config.segmenter._BACKBONE not in v.name]

        logger.info('Training these variables:')
        for v in train_var_list:
            logger.info(v.name)
        learning_rate = tf.train.polynomial_decay(
            self.config.segmenter._INITIAL_LEARNING_RATE,
            tf.cast(global_step, tf.int32),
            self.config.segmenter._MAX_ITER,
            self.config.segmenter._END_LEARNING_RATE,
            power=self.config.segmenter._POWER,
            cycle=False,
        )
        tf.identity(learning_rate, name="learning_rate")
        tf.summary.scalar("learning_rate", learning_rate)

        # Batch norm requires update ops to be added as a dependency to the
        # train_op
        logger.info('Defining gradient ops.')
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):

            warm_up = tf.train.polynomial_decay(
                1e-2,
                tf.cast(global_step, tf.int32),
                2500,
                1.0,
                power=0.9,
                cycle=False,
            )
            tf.identity(warm_up, name="warm_up")
            tf.summary.scalar("warm_up", warm_up)

            learning_rate *= warm_up

            train_ops = []

            hyperbolic_vars = [
                v for v in train_var_list if v in tf.get_collection("hyperbolic_vars")
            ]

            eucl_vars = [
                v for v in train_var_list if v not in hyperbolic_vars
            ]

            # train bb with tenth of learning rate
            bb_vars = [v for v in eucl_vars if self.config.segmenter._BACKBONE in v.name]
            # head vars are ALL remaining vars in the embedding function.
            head_vars = [v for v in eucl_vars if v not in bb_vars]

            tvars = bb_vars + head_vars + hyperbolic_vars
            tgrads = tf.gradients(loss, tvars)
            with tf.variable_scope('grad_hist'):
                for i in range(len(tvars)):
                    # logger.info(f'------- {tvars[i].name}')
                    # tf.summary.histogram('grad'+tvars[i].name, tgrads[i])
                    tgrads[i] = tf.debugging.check_numerics(tgrads[i], tvars[i].name)

            bbgrads = tgrads[: len(bb_vars)]

            headgrads = tgrads[len(bb_vars):len(bb_vars) + len(head_vars)]

            bb_updates = zip(bbgrads, bb_vars)
            head_updates = zip(headgrads, head_vars)

            eucl_grad_ops = []

            if bb_vars:
                optimizer_bb = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate / 10, momentum=self.config.segmenter._MOMENTUM
                )
                eucl_grad_ops.append(optimizer_bb.apply_gradients(bb_updates))
            if head_vars:
                optimizer_head = tf.train.MomentumOptimizer(
                    learning_rate=learning_rate, momentum=self.config.segmenter._MOMENTUM
                )
                eucl_grad_ops.append(optimizer_head.apply_gradients(head_updates, global_step))

            if hyperbolic_vars:
                # burn in op

                hyp_grads = tgrads[-len(hyperbolic_vars):]

                hyp_grads, _ = tf.clip_by_global_norm(hyp_grads, self.config.segmenter._GRAD_CLIP)

                hyp_ops = get_hyp_update_ops(
                    var=hyperbolic_vars,
                    grads=hyp_grads,
                    curvature=self.embedding_space.curvature,
                    learning_rate=learning_rate,
                    burnin=warm_up,
                )
                train_ops.append(hyp_ops)

            train_ops.extend(eucl_grad_ops)

            train_op = tf.group(*train_ops)
        return train_op

    def input_fn(self, is_training, batch_size, num_epochs=1, file=""):
        """Input_fn using the tf.data input pipeline.

        Args:
        is_training: A boolean denoting whether the input is for training.
        batch_size: The number of samples per batch.
        num_epochs: The number of epochs to repeat the dataset.
        file: Optional. Make tf record generator from tf record at <file>

        Returns:
        A tuple of images and labels.
        """
        if not file:
            files = get_filenames(is_training, self.config)
        else:
            files = [file]

        dataset = tf.data.Dataset.from_tensor_slices(files)
        dataset = dataset.flat_map(tf.data.TFRecordDataset)

        if is_training:
            # When choosing shuffle buffer sizes, larger sizes result in better
            # randomness, while smaller sizes have better performance.
            # is a relatively small dataset, we choose to shuffle the full epoch.
            dataset = dataset.shuffle(buffer_size=10000)
            # dataset = dataset.shard(45, 0) # reduce dset size for testing
        dataset = dataset.map(parse_record)
        dataset = dataset.map(
            lambda image, label: preprocess_image(image, label, is_training, self.config)
        )
        dataset = dataset.prefetch(batch_size)

        # We call repeat after shuffling, rather than before, to prevent separate
        # epochs from blending together.
        dataset = dataset.repeat(num_epochs)
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()

        images, labels = iterator.get_next()

        return images, labels

    def response2embeddings(self, response):
        """ Output of embedders might be different dimensionality than embedding space.
        This uses a non-normalized/non-activated linear transformation to bring dim to embedding space."""
        if self.config.segmenter._EFN_OUT_DIM != self.embedding_space.dim:
            embeddings = layers_lib.conv2d(
                response,
                self.embedding_space.dim,
                [1, 1],
                activation_fn=None,
                normalizer_fn=None,
                scope="down_dim_1x1",
            )
        else:
            embeddings = response
        return embeddings

    def train_val_fn(self, features, labels, mode):
        """ General purpose model function call for estimator.  """
        tf.logging.set_verbosity(tf.logging.ERROR)  # hide the flood of tf deprecation warnings
        self.embedding_space = self.init_embedding_space()
        if self.config.base_model_name is None:
            response = self.embedding_function(inputs=features,
                                               is_training=(mode == tf.estimator.ModeKeys.TRAIN))
        elif self.config.base_model_name == 'deeplab':
            response = self.embedding_function(inputs=features,
                                               is_training=(mode == tf.estimator.ModeKeys.TRAIN))
        else:
            print('Your choice is not available. We run deeplab.')
            response = self.embedding_function(inputs=features,
                                               is_training=(mode == tf.estimator.ModeKeys.TRAIN))
        embeddings = self.response2embeddings(response)
        tf.logging.set_verbosity(tf.logging.INFO)
        # tf.summary.histogram('embeddings', embeddings)
        projected_embeddings = self.embedding_space.project(embeddings)
        tf.summary.histogram('projected_embeddings', projected_embeddings)
        probs, cprobs = self.embedding_space.run(projected_embeddings)
        # tf.summary.histogram('joint_probabilities', probs)
        predictions = self.embedding_space.decide(probs)

        return_dict = {
            "embeddings": projected_embeddings,
            "predictions": predictions,
            "probabilities": probs,
            "features": features,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=return_dict,
                export_outputs={
                    "preds": tf.estimator.export.PredictOutput(
                        return_dict
                    )
                },
            )

        ## HANDLE OUTPUTS TO MAKE LOSS
        # resize labels to network output
        resize_label2output = True
        if resize_label2output:  # resize labels to output
            labels_s = tf.image.resize_images(
                labels, tf.shape(embeddings)[1:3], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR,
            )
            labels_flat = tf.reshape(labels_s, [-1])
        else:
            probs = tf.image.resize_bilinear(
                probs, tf.shape(features)[1:3],
                name="upsample_probs2labels"
            )
            labels_flat = tf.reshape(labels, [-1])

        if self.config.segmenter._ZERO_LABEL:
            # give unseen classes the 255='ignore' label
            masking_table = self.init_masking_unseen_lookup(self.tree)
            labels_flat = masking_table.lookup(labels_flat)

        # only take pixels labelled with valid labels to loss
        num_targets = self.tree.M
        valid_indices = tf.to_int32(labels_flat <= num_targets - 1)
        valid_labels = tf.boolean_mask(labels_flat, valid_indices)

        flat_cprobs = tf.reshape(cprobs, [-1, self.tree.M])
        valid_cprobs = tf.boolean_mask(flat_cprobs, valid_indices)

        ## LOSS
        cross_entropy = CCE(cond_probs=valid_cprobs, labels=valid_labels, tree=self.tree)
        tf.identity(cross_entropy, "CCE")
        tf.summary.scalar("CCE", cross_entropy)

        with tf.variable_scope("total_loss"):
            # Weight decay regularization
            l2_loss = self.config.segmenter._WEIGHT_DECAY * tf.add_n(
                [
                    tf.nn.l2_loss(v)
                    for v in tf.trainable_variables()
                    if v not in tf.get_collection("hyperbolic_vars")
                ]
            )
            total_loss = l2_loss + cross_entropy

        ## METRICS 
        # make predictions on target/leaf classes

        preds_flat = tf.reshape(predictions, [-1])
        valid_preds = tf.boolean_mask(preds_flat, valid_indices)

        valid_preds = tf.cast(valid_preds, tf.int64)
        valid_labels = tf.cast(valid_labels, tf.int64)

        if True:  # hierarchical metrics
            (parent_lookup, parent_count), (gparent_lookup, gparent_count) = self.init_hierarchy_lookup_tables()

            with tf.variable_scope('standard_metrics'):
                standard_metrics = tf_metrics(valid_labels, valid_preds, num_classes=self.tree.K)

            with tf.variable_scope('sibling_metrics'):
                valid_parent_labels = parent_lookup.lookup(valid_preds)
                valid_parent_pred = parent_lookup.lookup(valid_labels)
                sibling_metrics = tf_metrics(valid_parent_labels, valid_parent_pred, num_classes=parent_count)

            with tf.variable_scope('cousin_metrics'):
                valid_gparent_pred = gparent_lookup.lookup(valid_preds)
                valid_gparent_labels = gparent_lookup.lookup(valid_labels)
                cousin_metrics = tf_metrics(valid_gparent_labels, valid_gparent_pred, num_classes=gparent_count)

            # merge dicts
            metrics = {prefix + k: v for prefix, metric_dict in
                       zip(["", "sibling_", "cousin_"], [standard_metrics, sibling_metrics, cousin_metrics]) for k, v in
                       metric_dict.items()}
        else:
            metrics = tf_metrics(valid_labels, valid_preds, num_classes=self.tree.K)

        for metric_name, metric_tensor in metrics.items():
            if 'px_accuracy' in metric_name:
                tf.identity(metric_tensor[1], name=f'train_{metric_name}')
                tf.summary.scalar(f'train_{metric_name}', metric_tensor[1])

        train_mean_iou = compute_mean_iou(total_cm=metrics['mean_iou'][1], tree=self.tree)
        tf.identity(train_mean_iou, name="train_mean_iou")
        tf.summary.scalar("train_mean_iou", train_mean_iou)

        if resize_label2output:
            # upscale predictions
            large_probs = tf.image.resize_bilinear(
                probs, tf.shape(features)[1:3],
                name="large_probs"
            )
            large_predictions = tf.expand_dims(self.embedding_space.decide(large_probs), axis=3)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # logging predictions
            def _mean_image_addition(x, rgb_means=self.config.dataset._RGB_MEANS):
                return mean_image_addition(x, means=rgb_means)

            images = tf.cast(tf.map_fn(_mean_image_addition, features),
                             tf.uint8)
            pred_decoded_labels = tf.py_func(
                decode_labels,
                [large_predictions, self.config.segmenter._BATCH_SIZE, self.tree.K],
                tf.uint8,
            )

            gt_decoded_labels = tf.py_func(
                decode_labels,
                [labels, self.config.segmenter._BATCH_SIZE, self.tree.K],
                tf.uint8,
            )
            # tf.summary.image(
            #    "images",
            #    tf.concat(axis=2, values=[images, gt_decoded_labels, pred_decoded_labels]),
            #    max_outputs=1,
            # )  # Concatenate row-wise.

            train_op = self.get_train_op(total_loss)
        else:
            train_op = None

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=return_dict,
            loss=total_loss,
            train_op=train_op,
            eval_metric_ops=metrics,
        )

    def init_model(self, _model_fn):
        # Set up a RunConfig to only save checkpoints once per training cycle.
        session_cfg = tf.ConfigProto()
        session_cfg.gpu_options.visible_device_list = str(self.config._GPU_IDX)
        session_cfg.gpu_options.allow_growth = True
        session_cfg.intra_op_parallelism_threads = 8
        session_cfg.inter_op_parallelism_threads = 2
        session_cfg.allow_soft_placement = True

        run_config = tf.estimator.RunConfig(session_config=session_cfg).replace(
            save_checkpoints_secs=30 * 60,
            keep_checkpoint_max=2,
        )

        self.model = tf.estimator.Estimator(
            model_fn=_model_fn,
            model_dir=self.config._SEGMENTER_SAVE_DIR,
            config=run_config,
        )

    def train(self):
        """ Performs training. """
        logger.info("______________")
        logger.info("___TRAINING___")
        logger.info("______________")
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
        self.init_model(_model_fn=self.train_val_fn)
        tensors_to_log = {
            "learning_rate": "learning_rate",
            "warm_up": "warm_up",
            "train_px_accuracy": "train_px_accuracy",
            "train_mean_iou": "train_mean_iou",
            "CCE": "CCE",
            "global_step": "global_step",

        }
        logger.info(f'Training {self.config.segmenter._NUM_EPOCHS}')
        for edx in range(self.config.segmenter._NUM_EPOCHS // self.config.segmenter._EPOCHS_PER_EVAL):
            logging_hook = tf.train.LoggingTensorHook(
                tensors=tensors_to_log, every_n_iter=1
            )
            train_hooks = [logging_hook]

            logger.info(f"EPOCH {edx} / {self.config.segmenter._NUM_EPOCHS}")
            logger.info("Start training.")
            self.model.train(
                input_fn=lambda: self.input_fn(
                    is_training=True,
                    batch_size=self.config.segmenter._BATCH_SIZE,
                    num_epochs=self.config.segmenter._EPOCHS_PER_EVAL,
                ),
                hooks=train_hooks,
                # steps=1  # For debug
            )

            logger.info("Start evaluation.")
            # Evaluate the model and print results
            eval_results = self.model.evaluate(
                # Batch size must be 1 for testing because the images' size differs
                input_fn=lambda: self.input_fn(
                    is_training=False,
                    batch_size=1,
                    num_epochs=1
                ),
                hooks=[],
            )
            logger.info(f' Validation results: {eval_results}')

            valid_filename = 'validation_results.txt'
            with open(os.path.join(self.config._SEGMENTER_SAVE_DIR, valid_filename), 'a') as f:
                f.write(str(eval_results))
                f.write('\n')

    def init_masking_seen_lookup(self, class_tree):
        """ Builds lookup hash tables for quickly converting tf labels to give 'unseen' classes an
        'unlabelled'/'ignore' label 255 """

        # init hierarchy independently of model.
        labels = class_tree.target_classes
        converted_labels = []

        for n_idx in labels:
            node_name = class_tree.i2n[n_idx]
            if node_name not in self.config.dataset._UNSEEN:  # provide with 'ignore' label
                converted_labels.append(255)
            else:
                converted_labels.append(n_idx)
        converted_labels = np.array(converted_labels).astype(np.int32)
        labels = np.array(labels).astype(np.int32)
        masking_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(labels, converted_labels), default_value=255)
        return masking_table

    def init_masking_unseen_lookup(self, class_tree):
        """ Builds lookup hash tables for quickly converting tf labels to give 'unseen' classes an
        'unlabelled'/'ignore' label 255 """
        # init hierarchy independently of model.
        labels = class_tree.target_classes
        converted_labels = []

        for n_idx in labels:
            node_name = class_tree.i2n[n_idx]
            if node_name in self.config.dataset._UNSEEN:  # provide with 'ignore' label
                converted_labels.append(255)
            else:
                converted_labels.append(n_idx)
        converted_labels = np.array(converted_labels).astype(np.int32)
        labels = np.array(labels).astype(np.int32)
        masking_table = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(labels, converted_labels), default_value=255)
        return masking_table

    def init_hierarchy_lookup_tables(self, ):
        """ Builds lookup hash tables for quickly converting tf labels to parent (sibling_metric_lookup) or
        grandparent (cousin_dict) labels.
        This is used for calculating sibling and cousin IoU/ acc/ whatever. """
        # init hierarchy independently of model.
        tree_params = {
            'i2c': self.config.dataset._I2C,
            'json': json_lib.load(open(self.config.dataset._JSON_FILE))
        }
        class_tree = Tree(**tree_params)
        keys = class_tree.target_classes
        parent_ids = []
        grandparent_ids = []
        for n_idx in keys:
            node_name = class_tree.i2n[n_idx]
            parent = class_tree.get_parent_node(node_name)
            if parent.parent is not None:
                grandparent = class_tree.get_parent_node(parent.name)
            else:
                grandparent = parent
            parent_ids.append(parent.idx)
            grandparent_ids.append(grandparent.idx)

        # As tf metrics expect contigous class ids, we map (grand)parent id's to the range [0,num_(grand)parents]
        parent2contiguous = {pid: cid for cid, pid in enumerate(np.unique(parent_ids))}
        gparent2contiguous = {gpid: cid for cid, gpid in enumerate(np.unique(grandparent_ids))}
        parent_ids = [parent2contiguous[idx] for idx in parent_ids]
        grandparent_ids = [gparent2contiguous[idx] for idx in grandparent_ids]

        sibling_metric_lookup = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, parent_ids), default_value=255)
        cousin_metric_lookup = tf.contrib.lookup.HashTable(
            tf.contrib.lookup.KeyValueTensorInitializer(keys, grandparent_ids), default_value=255)
        return (sibling_metric_lookup, len(parent2contiguous)), (cousin_metric_lookup, len(gparent2contiguous))

    def test_fn(self, features, labels, mode):
        """ Test function call for estimator. Resizes output to labels. """
        if (mode == tf.estimator.ModeKeys.TRAIN):
            logger.error('Only use this function for evaluation.')
            raise SystemError
        # tf.reset_default_graph()
        tf.logging.set_verbosity(tf.logging.ERROR)  # hide the flood of tf deprecation warnings
        self.embedding_space = self.init_embedding_space()
        response = self.embedding_function(inputs=features,
                                           is_training=(mode == tf.estimator.ModeKeys.TRAIN))

        embeddings = self.response2embeddings(response)
        tf.logging.set_verbosity(tf.logging.INFO)
        tf.summary.histogram('embeddings', embeddings)
        projected_embeddings = self.embedding_space.project(embeddings)
        # tf.summary.histogram('projected_embeddings', projected_embeddings)
        probs, cprobs = self.embedding_space.run(projected_embeddings)
        # tf.summary.histogram('joint_probabilities', probs)

        # for evaluation resize output to labels
        probs = tf.image.resize_bilinear(
            probs, tf.shape(features)[1:3],
            name="upsample_probs2labels"
        )
        projected_embeddings = tf.image.resize_bilinear(
            projected_embeddings, tf.shape(features)[1:3],
            name="upsample_projected_embeddings"
        )

        # make predictions on target/leaf classes
        if self.config.segmenter._TEST_ZERO_LABEL:
            unseen_idx = [self.tree.c2i[c] for c in self.config.dataset._UNSEEN]
        else:
            unseen_idx = []
        predictions = self.embedding_space.decide(probs, unseen=unseen_idx)

        return_dict = {
            "embeddings": projected_embeddings,
            "predictions": predictions,
            "probabilities": probs,
            "features": features,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=mode,
                predictions=return_dict,
                export_outputs={
                    "preds": tf.estimator.export.PredictOutput(
                        return_dict
                    )
                },
            )

        labels_flat = tf.reshape(labels, [-1])

        if self.config.segmenter._TEST_ZERO_LABEL:
            # give seen classes the 255='ignore' label
            masking_table = self.init_masking_seen_lookup(self.tree)
            labels_flat = masking_table.lookup(labels_flat)

        # only take pixels labelled with valid labels to loss
        num_targets = self.tree.M
        valid_indices = tf.to_int32(labels_flat <= num_targets - 1)

        valid_labels = tf.cast(tf.boolean_mask(labels_flat, valid_indices), tf.int64)

        # probs_flat = tf.reshape(probs, [-1, self.tree.M])
        # valid_probs = tf.boolean_mask(probs_flat, valid_indices)

        flat_cprobs = tf.reshape(cprobs, [-1, self.tree.M])
        valid_cprobs = tf.boolean_mask(flat_cprobs, valid_indices)

        preds_flat = tf.reshape(predictions, [-1])
        valid_preds = tf.cast(tf.boolean_mask(preds_flat, valid_indices), tf.int64)

        loss = CCE(valid_cprobs, valid_labels, self.tree)

        # METRICS
        (parent_lookup, parent_count), (gparent_lookup, gparent_count) = self.init_hierarchy_lookup_tables()

        with tf.variable_scope('standard_metrics'):
            standard_metrics = tf_metrics(valid_labels, valid_preds, num_classes=self.tree.K)
            cm_metric = _streaming_confusion_matrix(valid_labels, valid_preds, self.tree.K)
        with tf.variable_scope('sibling_metrics'):
            valid_parent_labels = parent_lookup.lookup(valid_preds)
            valid_parent_pred = parent_lookup.lookup(valid_labels)
            sibling_metrics = tf_metrics(valid_parent_labels, valid_parent_pred, num_classes=parent_count)

        with tf.variable_scope('cousin_metrics'):
            valid_gparent_pred = gparent_lookup.lookup(valid_preds)
            valid_gparent_labels = gparent_lookup.lookup(valid_labels)
            cousin_metrics = tf_metrics(valid_gparent_labels, valid_gparent_pred, num_classes=gparent_count)

        # merge dicts
        metrics = {prefix + k: v for prefix, metric_dict in
                   zip(["", "sibling_", "cousin_"], [standard_metrics, sibling_metrics, cousin_metrics]) for k, v in
                   metric_dict.items()}

        metrics['confusion_matrix'] = cm_metric

        return_dict = {
            "embeddings": embeddings,
            "predictions": predictions,
            "probabilities": probs,
            "features": features,
        }

        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=return_dict,
            loss=loss,
            eval_metric_ops=metrics,
        )

    def test(self, ):
        fnfiles = os.listdir(self.config._SEGMENTER_SAVE_DIR)

        assert np.any(
            ['.ckpt' in x for x in fnfiles]), f'No weights present in {self.config._SEGMENTER_SAVE_DIR}! {fnfiles}'

        logger.info("______________")
        logger.info("____TESTING___")
        logger.info("______________")
        os.environ["TF_ENABLE_WINOGRAD_NONFUSED"] = "1"
        self.init_model(_model_fn=self.test_fn)
        test_results = self.model.evaluate(
            # Batch size must be 1 for testing because the images' size differs
            input_fn=lambda: self.input_fn(
                is_training=False,
                batch_size=1,
                num_epochs=1,
                file=self.config.dataset._TEST_FILE,
            ),
            hooks=[],
        )

        tree_params = {
            'i2c': self.config.dataset._I2C,
            'json': json_lib.load(open(self.config.dataset._JSON_FILE))
        }
        class_tree = Tree(**tree_params)

        cm = test_results['confusion_matrix']
        scm, ccm = hierarchical_cm(total_cm=cm, tree=class_tree)
        logger.info("NUMPY METRICS")
        hmetrics = defaultdict(dict)
        for name, _cm in [('~', cm), ('sibling', scm), ('cousin', ccm)]:
            iou, acc, cacc = npy_metrics(_cm, class_tree)
            hmetrics[name][name + 'iou'] = iou
            hmetrics[name][name + 'acc'] = acc
            hmetrics[name][name + 'cacc'] = cacc
            logger.info(f'{name}: mIoU: {iou}, acc: {acc}, c.acc: {cacc}')

        cls_metrics = cls_mean_iou_npy(total_cm=cm, tree=class_tree)
        del test_results['confusion_matrix']

        test_results = {**test_results, **cls_metrics}
        hmetric_file = self.config._SEGMENTER_SAVE_DIR.split('/')[-1] + '_hmetrics.json'
        if self.config.segmenter._TEST_ZERO_LABEL:
            hmetric_file = 'ZL_' + hmetric_file
        dir = os.path.join(os.path.abspath('poincare-HESP/'), 'metrics')
        if not os.path.exists(dir):
            os.mkdir(dir)
        with open(os.path.join(dir, hmetric_file), 'w') as f:
            f.write(json_lib.dumps(hmetrics, indent=2, sort_keys=True))

        test_filename = 'test_results.txt'
        if self.config.segmenter._TEST_ZERO_LABEL:
            test_filename = 'ZL_' + test_filename
        with open(os.path.join(self.config._SEGMENTER_SAVE_DIR, test_filename), 'w') as f:
            f.write(str(test_results))
            logger.info(f' Test results: {test_results}')

    def input_fn_npa(self, image, label, batch_size=1):
        if len(image.shape) == 3:
            image = image[None, ...]
        if len(label.shape) == 3:
            label = label[None, ...]
        dataset = tf.data.Dataset.from_tensor_slices((image, label))
        dataset = dataset.map(
            lambda feature, label: preprocess_image(feature, label, False, self.config))
        dataset = dataset.prefetch(batch_size)
        dataset = dataset.batch(batch_size)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()

    def _init_predict(self, ):
        self.init_model(_model_fn=self.test_fn)
        self.embedding_space = self.init_embedding_space()

    def predict(self, image, label):
        """ Performs training. """
        return next(self.model.predict(lambda: self.input_fn_npa(image, label)))
