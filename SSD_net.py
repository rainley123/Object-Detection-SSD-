import tensorflow as tf
import numpy as np
import math

import tensorflow.contrib.slim as slim
from collections import namedtuple

import custom_layers
import utils

SSDParams = namedtuple('SSDParameters', ['img_shape',                   # Shape of input image
                                         'num_classes',                 # Number of classification
                                         'no_annotation_label',         # No annotation label
                                         'feat_layers',                 # Feature maps
                                         'feat_shapes',                 # Shape of feature maps
                                         'anchor_size_bounds',          # Size of bounding box
                                         'anchor_sizes',                # Initial size of bounding box
                                         'anchor_ratios',               # Ratio of H, W
                                         'anchor_steps',                # The ratio of feature map to raw image
                                         'anchor_offset',               # Offset of center point
                                         'normalizations',              # Normalizations
                                         'prior_scaling'                # Scale of prior box to ground truth box
                                         ])


class SSDNet(object):
    """
    The default features layers with 300x300 image input are:
        conv4 == > 38
        conv7 == > 19
        conv8 == > 10
        conv9 == > 5
        conv10 == > 3
        conv11 == > 1
    """
    default_params = SSDParams(
        img_shape=(300, 300),
        num_classes=2,                                                                  # Background and car
        no_annotation_label=2,
        feat_layers=['block4', 'block7', 'block8', 'block9', 'block10', 'block11'],     # Name of feature maps
        feat_shapes=[(38, 38), (19, 19), (10, 10), (5, 5), (3, 3), (1, 1)],             # Shape of feature maps
        anchor_size_bounds=[0.15, 0.9],              # Initial box [0.15*300, 0.9*300]
        anchor_sizes=[(21., 45.),                    # sk = 0.15 + 0.75/(m-1)*(k-1), m=5
                      (45., 99.),
                      (99., 153.),
                      (153., 207.),
                      (207., 261.),
                      (261., 315.)],
        anchor_ratios=[[2, 0.5],                     # Conv4,conv10,conv11 only have 4 boxes, and the other have 6 boxes
                       [2, 3, 1./2, 1./3],
                       [2, 3, 1./2, 1./3],
                       [2, 3, 1./2, 1./3],
                       [2, 0.5],
                       [2, 0.5]],                    # Ratio = 1 and s = sqrt(sk*sk+1) is defined later
        anchor_steps=[8, 16, 32, 64, 100, 300],
        anchor_offset=0.5,                           # Center of boxes
        normalizations=[20, -1, -1, -1, -1, -1],     # Only conv4 needs to have a L2 normalization,
                                                     # Just on the shape of channel,
                                                     # To make sure there is no difference
                                                     # Between conv4 and another layers
        prior_scaling=[0.1, 0.1, 0.2, 0.2]           # (y,x,h,w) using in decode, this is the prediction???
    )

    def __init__(self, params=None):
        """
        Init the SSD net with some parameters. Use the default ones
        if none provided.
        """
        if isinstance(params, SSDParams):           # If there are input parameters
            self.params = params
        else:
            self.params = SSDNet.default_params

    def net(self, inputs,
            is_training=True,
            update_feat_shapes=True,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
        """
        :param inputs:
        :param is_training:
        :param update_feat_shapes:
        :param prediction_fn:
        :param reuse:
        :param scope:
        :return: the SSD net
        """

        r = ssd_net(inputs,
                    num_classes=self.params.num_classes,
                    feat_layers=self.params.feat_layers,
                    anchor_ratios=self.params.anchor_ratios,
                    normalizations=self.params.normalizations,
                    is_training=is_training,
                    prediction_fn=prediction_fn,
                    reuse=reuse,
                    scope=scope)

        return r

    def arg_scope(self, weight_decay=0.0005):
        """
        :param weight_decay:
        :return: the argument scope
        """
        return ssd_arg_scope(weight_decay)

    def anchors(self, img_shape, dtype=np.float32):
        """
        Compute the default anchor boxes, given an image shape.
        :param img_shape:
        :param dtype:
        :return: all anchors
        """
        return ssd_anchors_all_layers(img_shape,
                                      self.params.feat_shapes,
                                      self.params.anchor_sizes,
                                      self.params.anchor_ratios,
                                      self.params.anchor_steps,
                                      self.params.anchor_offset,
                                      dtype)

    def bboxes_encode(self, labels, bboxes, anchors, batch_size, scope='ssd_bboxes_encode'):
        """
        Encode the label and bounding boxes
        :param labels:
        :param bboxes:
        :param anchors:
        :param scope:
        :return: the labels, scores, relative location of target
        """
        batch_label = []
        batch_score = []
        batch_localisation = []
        for i in range(batch_size):
            glabel, gscore, glocalisation = utils.tf_ssd_bboxes_encode(labels[i], bboxes[i], anchors,
                                                                       self.params.num_classes,
                                                                       self.params.no_annotation_label,
                                                                       ignore_threshold=0.5,
                                                                       prior_scaling=self.params.prior_scaling,
                                                                       scope=scope)
            batch_label.append(glabel)
            batch_score.append(gscore)
            batch_localisation.append(glocalisation)

        return batch_label, batch_score, batch_localisation

    def bboex_decode(self, feature_localizations, anchors, scope='ssd_bboxes_decode'):
        """
        Decode the locations
        :param feature_localizations:
        :param anchors:
        :param scope:
        :return: the location of target
        """
        return utils.tf_ssd_bboxes_decode(feature_localizations, anchors,
                                          prior_scaling=self.params.prior_scaling,
                                          scope=scope)

    def detected_bboxes(self, predictions, localisations,
                        select_threshold=None, nms_threshold=0.5,
                        clipping_bbox=[0., 0., 1., 1.], top_k=400, keep_top_k=200):
        """
        Prediction process: 1.select all bboxes according to classes 2.select top-k 3.apply nms 4.clip the bbox
        :param predictions:
        :param localisations:
        :param select_threshold:
        :param nms_threshold:
        :param clipping_bbox:
        :param top_k:
        :param keep_top_k:
        :return: the scores and bbox after process
        """
        rscores, rbboxes = utils.tf_ssd_bboxes_select(predictions, localisations,
                                                     select_threshold=select_threshold,
                                                     num_classes=self.params.num_classes)

        rscores, rbboxes = utils.bboxes_sort(rscores, rbboxes, top_k)

        rscores, rbboxes = utils.bboxes_nms_batch(rscores, rbboxes,
                                                  nms_threshold=nms_threshold,
                                                  keep_top_k=keep_top_k)

        if clipping_bbox is not None:
            rbboxes = utils.bboxes_clip(clipping_bbox, rbboxes)

        return rscores, rbboxes

    def losses(self, logits, localisations,
               gclasses, glocalisations, gscores,
               batch_size,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               scope='ssd_losses'):
        """
        Define the SSD loss
        :param logits:
        :param localisations:
        :param gclasses:
        :param glocalisations:
        :param gscores:
        :param match_threshold:
        :param negative:
        :param alpha:
        :param label_smoothing:
        :param scope:
        :return: the loss of SSD
        """
        return ssd_losses(logits, localisations, gclasses, glocalisations, gscores, batch_size,
                          match_threshold, negative_ratio, alpha, label_smoothing,
                          scope=scope)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
                           Functional definition   
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


def ssd_net(inputs,
            num_classes,
            feat_layers,
            anchor_ratios,
            normalizations,
            is_training=True,
            prediction_fn=slim.softmax,
            reuse=None,
            scope='ssd_300_vgg'):
    """
    Create the SSD net
    :param inputs:
    :param num_classes:
    :param feat_layers:
    :param anchor_ratios:
    :param normalizations:
    :param is_training:
    :param prediction_fn:
    :param reuse:
    :param scope:
    :return: the SSD net
    """
    end_points = {}
    with tf.variable_scope(scope, 'ssd_300_vgg', [inputs], reuse=reuse):
        """
        Origin VGG-16 block
        """
        # Block 1
        net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
        end_points['block1'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool1')

        # Block 2
        net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
        end_points['block2'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool2')

        # Block 3:
        net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
        end_points['block3'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool3')

        # Block 4 : Out 38x38
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
        end_points['block4'] = net
        net = slim.max_pool2d(net, [2, 2], scope='pool4')

        # Block 5 : Out 19x19
        net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
        end_points['block5'] = net
        net = slim.max_pool2d(net, [3, 3], stride=1, scope='pool5')

        """
        Additional SSD blocks
        """
        # Block 6 : atrous convolution Out 19x19
        net = slim.conv2d(net, 1024, [3, 3], rate=6, scope='conv6')
        end_points['block6'] = net
        net = tf.layers.dropout(net, rate=0.5, training=is_training)

        # Block 7 : 1x1 conv Out 19x19
        net = slim.conv2d(net, 1024, [1, 1], scope='conv7')
        end_points['block7'] = net
        net = tf.layers.dropout(net, rate=0.5, training=is_training)

        """
        Block 8-11, 1x1 3x3 kernel stride 2(except last)
        """
        with tf.variable_scope('block8'):
            net = slim.conv2d(net, 256, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 512, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points['block8'] = net

        with tf.variable_scope('block9'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = custom_layers.pad2d(net, pad=(1, 1))
            net = slim.conv2d(net, 256, [3, 3], stride=2, scope='conv3x3', padding='VALID')
        end_points['block9'] = net

        with tf.variable_scope('block10'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points['block10'] = net

        with tf.variable_scope('block11'):
            net = slim.conv2d(net, 128, [1, 1], scope='conv1x1')
            net = slim.conv2d(net, 256, [3, 3], scope='conv3x3', padding='VALID')
        end_points['block11'] = net

        # Prediction and localisations layers
        prediction = []
        logits = []
        localisations = []

        for i, layer in enumerate(feat_layers):
            with tf.variable_scope(layer + '_box'):
                pred, loca = ssd_multibox_layer(end_points[layer],
                                          num_classes,
                                          anchor_ratios[i],
                                          normalizations[i])
            prediction.append(prediction_fn(pred))                  # softmax, get the probability
            logits.append(pred)
            localisations.append(loca)

        return prediction, localisations, logits, end_points


def ssd_multibox_layer(inputs,                      # Feature maps
                       num_classes,
                       ratios=[1],                  # Default H:W = 1:1
                       normalization=-1):
    """
    Construct a multibox layer, return a class and localization predictions.
    :param inputs:
    :param num_classes:
    :param ratios:
    :param normalization:
    :return: the class and localization prediction
    """
    net = inputs
    if normalization > 0:
        net = custom_layers.l2_normalization(net)

    # Number of anchors
    num_anchors = len(ratios) + 2

    # Location: each anchor has 4 location
    num_loc_pred = num_anchors * 4

    # Have a 3x3 conv on feature map to get location prediction, and the number of location prediction is num_loc_pred
    loc_pred = slim.conv2d(net, num_loc_pred, [3, 3], activation_fn=None, scope='conv_loc')
    loc_shape = [tf.shape(loc_pred)[0]] + loc_pred.get_shape().as_list()[1:3] + [num_anchors, 4]
    loc_pred = tf.reshape(loc_pred, loc_shape)

    # Class prediction : each anchor has 2 classifaction
    num_cls_pred = num_anchors * num_classes
    cls_pred = slim.conv2d(net, num_cls_pred, [3, 3], activation_fn=None, scope='conv_cls')
    cls_shape = [tf.shape(cls_pred)[0]] + cls_pred.get_shape().as_list()[1:3] + [num_anchors, num_classes]
    cls_pred = tf.reshape(cls_pred, cls_shape)

    return cls_pred, loc_pred


def ssd_arg_scope(weight_decay=0.0005):
    """
    :param weight_decay:
    :return: the argument scope
    """
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_regularizer=slim.l2_regularizer(weight_decay),
                        weights_initializer=tf.contrib.layers.xavier_initializer(),
                        biases_initializer=tf.zeros_initializer()):
            with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                                padding='SAME') as sc:
                    return sc


def ssd_anchors_all_layers(img_shape,                   # Shape of raw image
                           layers_shape,                # Shape of feature maps
                           anchor_sizes,                # Initial size of anchors
                           anchor_ratios,               # H : W
                           anchor_steps,                # Ratio of anchor to raw image
                           offset=0.5,                  # Center
                           dtype=np.float32):
    """
    :param img_shape:
    :param layers_shape:
    :param anchor_sizes:
    :param anchor_ratios:
    :param anchor_steps:
    :param offset:
    :param dtype:
    :return: anchors of all layers
    """
    layers_anchors = []
    for i, layer in enumerate(layers_shape):
        anchor_boxes = ssd_anchors_one_layer(img_shape, layer, anchor_sizes[i], anchor_ratios[i],
                                             anchor_steps[i], offset=offset, dtype=dtype)
        layers_anchors.append(anchor_boxes)

    return layers_anchors


def ssd_anchors_one_layer(img_shape,
                          feature_shape,
                          sizes,
                          ratios,
                          step,
                          offset=0.5,
                          dtype=np.float32):
    """
    :param img_shape:
    :param feature_shape:
    :param sizes:
    :param ratios:
    :param step:
    :param offset:
    :param dtype:
    :return: anchors of one layer
    """
    # Compute the relative x,y
    y, x = np.mgrid[0:feature_shape[0], 0:feature_shape[1]]
    y = (y.astype(dtype) + offset) * step / img_shape[0]
    x = (x.astype(dtype) + offset) * step / img_shape[1]        # The Center of anchor (x,y) on the raw image

    # Expand the dims
    y = np.expand_dims(y, axis=-1)
    x = np.expand_dims(x, axis=-1)                              # The shape of x, y is (h, w, 1)

    # Compute the relative height and width
    num_anchors = len(ratios) + 2
    h = np.zeros((num_anchors, ), dtype=dtype)
    w = np.zeros((num_anchors, ), dtype=dtype)

    # Add first anchor boxes with ratio=1
    h[0] = sizes[0] / img_shape[0]
    w[0] = sizes[0] / img_shape[1]

    # Add anthor anchor boxes with s=sqrt(sk*sk+1)
    h[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[0]
    w[1] = math.sqrt(sizes[0] * sizes[1]) / img_shape[1]

    for i, ratio in enumerate(ratios):
        h[i+2] = sizes[0] / (math.sqrt(ratio)) / img_shape[0]
        w[i+2] = sizes[0] * (math.sqrt(ratio)) / img_shape[1]

    return y, x, h, w


def ssd_losses(logits, localisations,
               gclasses, glocalisations, gscores,
               batch_size,
               match_threshold=0.5,
               negative_ratio=3.,
               alpha=1.,
               label_smoothing=0.,
               device='/gpu:0',
               scope=None):
    with tf.name_scope(scope, 'ssd_losses'):
        # Batch x Layer x Tensor  ->  Layer x Batch x Tensor
        gclasses = np.transpose(np.array(gclasses), [1, 0]).tolist()
        glocalisations = np.transpose(np.array(glocalisations), [1, 0]).tolist()
        gscores = np.transpose(np.array(gscores), [1, 0]).tolist()

        lshape = logits[0].get_shape().as_list()
        num_classes = lshape[-1]

        # Flatten out all vectors!
        flogits = []
        fgclasses = []
        fgscores = []
        flocalisations = []
        fglocalisations = []
        for i in range(len(logits)):
            flogits.append(tf.reshape(logits[i], [-1, num_classes]))
            fgclasses.append(tf.reshape(gclasses[i], [-1]))
            fgscores.append(tf.reshape(gscores[i], [-1]))
            flocalisations.append(tf.reshape(localisations[i], [-1, 4]))
            fglocalisations.append(tf.reshape(glocalisations[i], [-1, 4]))

        # And concat the crap!
        logits = tf.concat(flogits, axis=0)
        gclasses = tf.concat(fgclasses, axis=0)
        gscores = tf.concat(fgscores, axis=0)
        localisations = tf.concat(flocalisations, axis=0)
        glocalisations = tf.concat(fglocalisations, axis=0)
        dtype = logits.dtype

        # Compute positive matching mask...
        pmask = gscores > match_threshold
        fpmask = tf.cast(pmask, dtype)
        n_positives = tf.reduce_sum(fpmask)

        # Hard negative mining...
        no_classes = tf.cast(pmask, tf.int32)
        predictions = slim.softmax(logits)
        nmask = tf.logical_and(tf.logical_not(pmask),
                               gscores > -0.5)
        fnmask = tf.cast(nmask, dtype)
        nvalues = tf.where(nmask,
                           predictions[:, 0],
                           1. - fnmask)
        nvalues_flat = tf.reshape(nvalues, [-1])

        # Number of negative entries to select.
        max_neg_entries = tf.cast(tf.reduce_sum(fnmask), tf.int32)
        n_neg = tf.cast(negative_ratio * n_positives, tf.int32) + batch_size
        n_neg = tf.minimum(n_neg, max_neg_entries)

        val, idxes = tf.nn.top_k(-nvalues_flat, k=n_neg)
        max_hard_pred = -val[-1]

        # Final negative mask.
        nmask = tf.logical_and(nmask, nvalues < max_hard_pred)
        fnmask = tf.cast(nmask, dtype)

        fn_neg = tf.cast(n_neg, tf.float32)
        fn_positives = tf.cast(n_positives, tf.float32)
        # Add cross-entropy loss.
        with tf.name_scope('cross_entropy_pos'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits+1e-10,
                                                                  labels=gclasses)
            # loss = tf.math.divide(tf.reduce_sum(loss * fpmask), tf.cast(batch_size, tf.float32), name='value')
            loss = tf.math.divide(tf.reduce_sum(loss * fpmask), fn_positives, name='value')
            tf.losses.add_loss(loss)

        with tf.name_scope('cross_entropy_neg'):
            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                                  labels=no_classes)
            # loss = tf.math.divide(tf.reduce_sum(loss * fnmask), tf.cast(batch_size, tf.float32), name='value')
            loss = tf.math.divide(tf.reduce_sum(loss * fnmask), fn_neg, name='value')
            tf.losses.add_loss(loss)

        # Add localization loss: smooth L1, L2, ...
        with tf.name_scope('localization'):
            # Weights Tensor: positive mask + random negative.
            weights = tf.expand_dims(alpha * fpmask, axis=-1)
            loss = custom_layers.abs_smooth(localisations - glocalisations)
            loss = tf.math.divide(tf.reduce_sum(loss * weights), tf.cast(batch_size, tf.float32), name='value')
            tf.losses.add_loss(loss)
