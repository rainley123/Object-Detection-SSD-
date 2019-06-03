# Help function for SSD
import tensorflow as tf
import tensorflow.contrib.slim as slim
import preprocess

PADDED_LENGTH = 45

def tf_ssd_bboxes_encode(labels,
                         bboxes,
                         anchors,
                         num_classes,
                         no_annotation_label,
                         ignore_threshold=0.5,
                         prior_scaling=[0.1, 0.1, 0.2, 0.2],
                         dtype=tf.float32,
                         scope='ssd_bboxes_encode'):
    """
    Get the location, label, IOU scores of all feature layers
    :param labels:
    :param bboxes:
    :param anchors:
    :param num_classes:
    :param no_annotation_label:
    :param ignore_threshold:
    :param prior_scaling:
    :param dtype:
    :param scope:
    :return: the location, label, IOU of all layers
    """
    with tf.name_scope(scope):
        target_labels = []
        target_localizations = []
        target_scores = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_encode_%i' % i):
                feature_labels, feature_scores, feature_location = \
                    tf_ssd_bboxes_encode_layer(labels, bboxes, anchors_layer, num_classes,
                                               no_annotation_label, ignore_threshold,
                                               prior_scaling, dtype)
                target_labels.append(feature_labels)
                target_scores.append(feature_scores)
                target_localizations.append(feature_location)

        return target_labels, target_scores, target_localizations


def tf_ssd_bboxes_encode_layer(labels,                                  # Class of the object, dicm = 1 (int)
                               bboxes,                                  # Ground truth box, dicm = 4 (float)
                               anchors_layer,
                               num_classes,
                               no_annotation_label,
                               ignore_threshold=0.5,
                               prior_scaling=[0.1, 0.1, 0.2, 0.2],
                               dtype=tf.float32):
    """
    Get the prediction location, label, IOU scores of one feature layer
    :param labels:
    :param bboxes:
    :param anchors_layer:
    :param num_classes:
    :param no_annotation_label:
    :param ignore_threshold:
    :param prior_scaling:
    :param dtype:
    :return:
    """
    yref, xref, href, wref = anchors_layer
    ymin = yref - href / 2.
    xmin = xref - wref / 2.
    ymax = yref + href / 2.
    xmax = xref + wref / 2.
    vol_anchors = (xmax - xmin) * (ymax - ymin)

    # Initialize the tensor
    shape = ymin.shape
    feature_labels = tf.zeros(shape=shape, dtype=tf.int64)
    feature_scores = tf.zeros(shape=shape, dtype=tf.float32)

    feature_ymin = tf.zeros(shape, dtype=dtype)
    feature_xmin = tf.zeros(shape, dtype=dtype)
    feature_ymax = tf.ones(shape, dtype=dtype)
    feature_xmax = tf.ones(shape, dtype=dtype)

    def jaccard_with_anchors(bbox):
        """
        :param bbox:
        :return: the IOU of each bbox
        """
        inter_ymin = tf.maximum(ymin, bbox[0])
        inter_xmin = tf.maximum(xmin, bbox[1])
        inter_ymax = tf.minimum(ymax, bbox[2])
        inter_xmax = tf.minimum(xmax, bbox[3])

        inter_h = tf.maximum((inter_ymax - inter_ymin), 0.0)
        inter_w = tf.maximum((inter_xmax - inter_xmin), 0.0)

        # Volume
        inter_volum = inter_h * inter_w
        bbox_volum = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        union_volum = vol_anchors + bbox_volum - inter_volum
        jaccard = tf.math.divide(inter_volum, union_volum)
        return jaccard


    def condition(i, feature_labels, feature_scores,
                  feature_ymin, feature_xmin, feature_ymax, feature_xmax):
        r = tf.less(i, tf.shape(labels))
        return r[0]

    # Get the labels, scores, GT's position of all anchors
    def body(i, feature_labels, feature_scores,
                  feature_ymin, feature_xmin, feature_ymax, feature_xmax):
        label = labels[i]
        bbox = bboxes[i]
        jaccard = jaccard_with_anchors(bbox)

        mask = tf.greater(jaccard, feature_scores)
        mask = tf.logical_and(mask, feature_scores > -0.5)
        mask = tf.logical_and(mask, label < num_classes)

        imask = tf.cast(mask, tf.int64)
        fmask = tf.cast(mask, dtype)

        # Place the labels and scores of all anchors into the matrix below
        feature_labels = imask * label + (1 - imask) * feature_labels
        feature_scores = tf.where(mask, jaccard, feature_scores)

        # Place the GT's position of all anchors into the matrix below
        feature_ymin = fmask * bbox[0] + (1 - fmask) * feature_ymin
        feature_xmin = fmask * bbox[1] + (1 - fmask) * feature_xmin
        feature_ymax = fmask * bbox[2] + (1 - fmask) * feature_ymax
        feature_xmax = fmask * bbox[3] + (1 - fmask) * feature_xmax

        return [i+1, feature_labels, feature_scores, feature_ymin, feature_xmin, feature_ymax, feature_xmax]

    # Main loop
    i = 0
    [i, feature_labels, feature_scores,
     feature_ymin, feature_xmin,
     feature_ymax, feature_xmax] = tf.while_loop(condition, body, [i, feature_labels, feature_scores,
                                                                   feature_ymin, feature_xmin,
                                                                   feature_ymax, feature_xmax])
    # Transform to center and H, W
    feature_centery = (feature_ymax + feature_ymin) / 2.0
    feature_centerx = (feature_xmax + feature_xmin) / 2.0
    feature_height = feature_ymax - feature_ymin
    feature_width = feature_xmax - feature_xmin

    # Encode the features
    feature_centery = (feature_centery - yref) / href / prior_scaling[0]
    feature_centerx = (feature_centerx - xref) / wref / prior_scaling[1]
    feature_height = tf.log(feature_height / href) / prior_scaling[2]
    feature_width = tf.log(feature_width / wref) / prior_scaling[3]

    # Use SSD order : x, y, h, w. for example:the shape of centerx is 38x38x4
    # Then it will become 38x38x4x4, and the last shape is [x, y, w, h]
    feature_location = tf.stack([feature_centerx, feature_centery, feature_width, feature_height], axis=-1)

    return feature_labels, feature_scores, feature_location


def tf_ssd_bboxes_decode(feature_localizations,
                         anchors,
                         prior_scaling,
                         scope='ssd_bboxes_decode'):
    """
    Get the real location of target
    :param feature_localizations:
    :param anchors:
    :param prior_scaling:
    :param scope:
    :return: bboxes of target
    """
    with tf.name_scope(scope):
        bboxes = []
        for i, anchors_layer in enumerate(anchors):
            with tf.name_scope('bboxes_decode_%i' % i):
                bbox = tf_ssd_bboxes_decode_layer(feature_localizations[i], anchors_layer, prior_scaling)
                bboxes.append(bbox)

        return bboxes


def tf_ssd_bboxes_decode_layer(feature_localizations,
                               anchors_layer,
                               prior_scaling):
    """
    :param feature_localizations:
    :param anchors_layer:
    :param prior_scaling:
    :return: ymin xmin ymax xmax
    """
    yref, xref, href, wref = anchors_layer

    # Calculate the GT
    centerx = feature_localizations[:, :, :, :, 0] * wref * prior_scaling[0] + xref
    centery = feature_localizations[:, :, :, :, 1] * href * prior_scaling[1] + yref
    width = tf.exp(feature_localizations[:, :, :, :, 2]) * wref * prior_scaling[2]
    height = tf.exp(feature_localizations[:, :, :, :, 3]) * href * prior_scaling[3]

    ymin = centery - height / 2.0
    xmin = centerx - width / 2.0
    ymax = centery + height / 2.0
    xmax = centerx + width / 2.0

    bbox = tf.stack([ymin, xmin, ymax, xmax], axis=-1)
    return bbox


def tf_ssd_bboxes_select_layer(predictions_layer, localizations_layer,
                               select_threshold=None,
                               num_classes=2,
                               ignore_class=0,
                               scope=None):
    """
    Select the boxes from one layer
    :param predictions_layer: Batch x 38 x 38 x 4 x 2
    :param localizations_layer: Batch x 38 x 38 x 4 x 4
    :param select_threshold:
    :param num_classes:
    :param ignore_class:
    :param scope:
    :return: the dirctiory of all classes and their scores and location
    """
    with tf.name_scope(scope, 'ssd_boxes_select_layer',
                       [predictions_layer, localizations_layer]):

        # Reshape the features -> Batch x N x num_classes | 4
        p_shape = tf.shape(predictions_layer)
        predictions_layer = tf.reshape(predictions_layer, tf.stack([p_shape[0], -1, p_shape[-1]]))

        l_shape = tf.shape(localizations_layer)
        localizations_layer = tf.reshape(localizations_layer, tf.stack([l_shape[0], -1, l_shape[-1]]))

        directory_scores = {}
        directory_bboxes = {}
        for classes in range(num_classes):
            if classes != ignore_class:
                scores = predictions_layer[:, :, classes]

                # Remove the scores < 0.5
                fmask = tf.cast(tf.greater_equal(scores, select_threshold), scores.dtype)
                scores = scores * fmask
                bboxes = localizations_layer * tf.expand_dims(fmask, axis=-1)

                directory_scores[classes] = scores
                directory_bboxes[classes] = bboxes

        return directory_scores, directory_bboxes


def tf_ssd_bboxes_select(predictions_net, localizations_net,
                         select_threshold=None,
                         num_classes=2,
                         ignore_class=0,
                         scope=None):
    """
    Select the boxes from all layers
    :param predictions_net: Layer x Batch x 38 x 38 x 4 x num_classes
    :param localizations_net: Layer x Batch x 38 x 38 x 4 x 4
    :param select_threshold:
    :param num_classes:
    :param ignore_class:
    :param scope:
    :return: the dirctiory of all classes and their scores and location
    """
    with tf.name_scope(scope, 'ssd_bboxes_select',
                       [predictions_net, localizations_net]):
        l_scores = []
        l_bboxes = []
        for i in range(len(predictions_net)):
            scores, bboxes = tf_ssd_bboxes_select_layer(predictions_net[i], localizations_net[i],
                                                        select_threshold, num_classes, ignore_class)
            l_scores.append(scores)
            l_bboxes.append(bboxes)

        directory_scores = {}
        directory_bboxes = {}
        for classes in l_scores[0].keys():
            ls = [s[classes] for s in l_scores]
            lb = [b[classes] for b in l_bboxes]

            # -> Batch x N | Batch x N x 4
            directory_scores[classes] = tf.concat(ls, axis=1)
            directory_bboxes[classes] = tf.concat(lb, axis=1)

        return directory_scores, directory_bboxes


def bboxes_sort(scores, bboxes, top_k=400, scope=None):
    """
    Select the top 400 boxes
    :param scores: Batch x N
    :param bboxes: Batch x N x 4
    :param top_k:
    :param scope:
    :return: the directory of top-400 scores and location
    """
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_sort_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_sort(scores[c], bboxes[c], top_k=top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

    # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_sort', [scores, bboxes]):
        # Sort scores...
        scores, idxes = tf.nn.top_k(scores, k=top_k, sorted=True)

        # Trick to be able to use tf.gather: map for each element in the first dim.
        def fn_gather(bboxes, idxes):
            bb = tf.gather(bboxes, idxes)
            return [bb]

        r = tf.map_fn(lambda x: fn_gather(x[0], x[1]),
                      [bboxes, idxes],
                      dtype=[bboxes.dtype],
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        bboxes = r[0]
        return scores, bboxes


def bboxes_nms_batch(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """
    Apply non-maximum to boxes
    :param scores: dict class 400 x 1
    :param bboxes: dict class 400 x 4
    :param nms_threshold:
    :param keep_top_k:
    :param scope:
    :return: the scores and boxes after nms
    """
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_nms_batch_dict'):
            d_scores = {}
            d_bboxes = {}
            for c in scores.keys():
                s, b = bboxes_nms_batch(scores[c], bboxes[c],
                                        nms_threshold=nms_threshold,
                                        keep_top_k=keep_top_k)
                d_scores[c] = s
                d_bboxes[c] = b
            return d_scores, d_bboxes

        # Tensors inputs.
    with tf.name_scope(scope, 'bboxes_nms_batch'):
        r = tf.map_fn(lambda x: bboxes_nms(x[0], x[1],
                                           nms_threshold, keep_top_k),
                      (scores, bboxes),
                      dtype=(scores.dtype, bboxes.dtype),
                      parallel_iterations=10,
                      back_prop=False,
                      swap_memory=False,
                      infer_shape=True)
        scores, bboxes = r
        return scores, bboxes


def bboxes_nms(scores, bboxes, nms_threshold=0.5, keep_top_k=200, scope=None):
    """
    Apply nms to all bboxes
    :param scores: Batch x N x 1
    :param bboxes: Batch x N x 4
    :param nms_threshold:
    :param keep_top_k:
    :param scope:
    :return: Batch x N x 1 or 4
    """
    with tf.name_scope(scope, 'bboxes_nms_single', [scores, bboxes]):
        # Apply NMS algorithm.
        idxes = tf.image.non_max_suppression(bboxes, scores,
                                             keep_top_k, nms_threshold)
        scores = tf.gather(scores, idxes)
        bboxes = tf.gather(bboxes, idxes)

        # pad
        shape = tf.shape(scores)
        pad_size = tf.maximum(keep_top_k - shape[0], 0)
        pad1 = [[0, pad_size]]
        pad2 = [[0, pad_size], [0, 0]]

        scores = tf.pad(scores, paddings=pad1, mode='CONSTANT')
        bboxes = tf.pad(bboxes, paddings=pad2, mode='CONSTANT')

        # Get fully defined shape
        scores = tf.reshape(scores, shape=[200])
        bboxes = tf.reshape(bboxes, shape=[200, 4])
        return scores, bboxes


def bboxes_clip(bbox_ref, bboxes, scope=None):
    """
    Clop the bboxes to a reference bbox
    :param bbox_ref:
    :param bboxes:
    :param scope:
    :return: clipped bboxes
    """
    if isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_clip_dict'):
            d_bbox = {}
            for c in bboxes.keys():
                d_bbox[c] = bboxes_clip(bbox_ref, bboxes[c])
            return d_bbox

    with tf.name_scope(scope, 'bboxes_clip'):
        bbox_ref = tf.transpose(bbox_ref)               # 4 x N x Batch
        bboxes = tf.transpose(bboxes)

        # Intersection bboxes and reference bbox.
        ymin = tf.maximum(bboxes[0], bbox_ref[0])       # N x Batch
        xmin = tf.maximum(bboxes[1], bbox_ref[1])
        ymax = tf.minimum(bboxes[2], bbox_ref[2])
        xmax = tf.minimum(bboxes[3], bbox_ref[3])

        # Double check! Empty boxes when no-intersection.
        ymin = tf.minimum(ymin, ymax)
        xmin = tf.minimum(xmin, xmax)
        bboxes = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))       # Batch x N x 4
    return bboxes


def parser(serialized_example):   
    features = tf.parse_single_example(
        serialized_example,
        features={
            'xmin': tf.VarLenFeature(tf.float32),
            'ymin': tf.VarLenFeature(tf.float32),
            'xmax': tf.VarLenFeature(tf.float32),
            'ymax': tf.VarLenFeature(tf.float32),
            'image': tf.FixedLenFeature([], tf.string),
            'label': tf.VarLenFeature(tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'height': tf.FixedLenFeature([], tf.int64),
            'depth': tf.FixedLenFeature([], tf.int64),
        })

    image = features['image']
    width = features['width']
    height = features['height']
    depth = features['depth']

    xmin = tf.sparse.to_dense(features['xmin'])
    ymin = tf.sparse.to_dense(features['ymin'])
    xmax = tf.sparse.to_dense(features['xmax'])
    ymax = tf.sparse.to_dense(features['ymax'])
    label = tf.sparse.to_dense(features['label'])

    # convert string to int
    image = tf.decode_raw(image, tf.uint8)
    image_shape = [height, width, depth]
    image = tf.reshape(image, image_shape)
    
    bbox = tf.stack([ymin, xmin, ymax, xmax], axis=1)
    
    image, label, bbox = preprocess.preprocess_for_train(image, image_shape, label, bbox)

    bbox = tf.transpose(bbox)
    ymin = tf.reshape(bbox[0], shape=[PADDED_LENGTH, ])
    xmin = tf.reshape(bbox[1], shape=[PADDED_LENGTH, ])
    ymax = tf.reshape(bbox[2], shape=[PADDED_LENGTH, ])
    xmax = tf.reshape(bbox[3], shape=[PADDED_LENGTH, ])
    bbox = tf.stack([ymin, xmin, ymax, xmax], axis=1)

    label = tf.reshape(label, shape=[PADDED_LENGTH, ])

    return image, label, bbox


def precision(tp, fp):
    class_accuracy = []
    for i in tp.keys():
        num_tp = tf.reduce_sum(tf.cast(tp[i], tf.float32))
        num_fp = tf.reduce_sum(tf.cast(fp[i], tf.float32))
        batch_accuracy = num_tp / (num_tp + num_fp)
        class_accuracy.append(batch_accuracy)
    accuracy = tf.reduce_mean(class_accuracy)
    return accuracy


def bboxes_matching_batch(labels, scores, bboxes, glabels, gbboxes, gdifficults,
                        matching_threshold=0.5, scope=None):
    if isinstance(scores, dict) or isinstance(bboxes, dict):
        with tf.name_scope(scope, 'bboxes_matching_batch_dict'):
            dict_n_gbboxes = {}
            dict_tp = {}
            dict_fp = {}
            for c in labels:
                n, tp, fp, _ = bboxes_matching_batch(c, scores[c], bboxes[c], glabels, gbboxes, 
                                gdifficults, matching_threshold) 
                dict_n_gbboxes[c] = n
                dict_tp[c] = tp
                dict_fp[c] = fp
            return dict_n_gbboxes, dict_tp, dict_fp, scores
    
    with tf.name_scope(scope, 'bboxes_matching_batch', [scores, bboxes, glabels, gbboxes]):
        r = tf.map_fn(lambda x: bboxes_matching(labels, x[0], x[1], x[2], x[3], x[4], matching_threshold),
                        (scores, bboxes, glabels, gbboxes, gdifficults),
                        dtype=(tf.int64, tf.bool, tf.bool),
                        parallel_iterations=10,
                        back_prop=False,
                        swap_memory=True,
                        infer_shape=True)
        return r[0], r[1], r[2], scores


def bboxes_matching(label, scores, bboxes, glabels, gbboxes, gdifficults, matching_threshold=0.5, scope=None):
    with tf.name_scope(scope, 'bboxes_matching_single',
                       [scores, bboxes, glabels, gbboxes]):
        rsize = tf.size(scores)
        rshape = tf.shape(scores)
        rlabel = tf.cast(label, glabels.dtype)

        # Number of groundtruth boxes.
        gdifficults = tf.cast(gdifficults, tf.bool)
        n_gbboxes = tf.count_nonzero(tf.logical_and(tf.equal(glabels, label),
                                                    tf.logical_not(gdifficults)))

        # Grountruth matching arrays.
        gmatch = tf.zeros(tf.shape(glabels), dtype=tf.bool)
        grange = tf.range(tf.size(glabels), dtype=tf.int32)

        # True/False positive matching TensorArrays.
        sdtype = tf.bool
        ta_tp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)
        ta_fp_bool = tf.TensorArray(sdtype, size=rsize, dynamic_size=False, infer_shape=True)

        # Loop over returned objects.
        def m_condition(i, ta_tp, ta_fp, gmatch):
            r = tf.less(i, rsize)
            return r

        def m_body(i, ta_tp, ta_fp, gmatch):
            # Jaccard score with groundtruth bboxes.
            rbbox = bboxes[i]
            jaccard = bboxes_jaccard(rbbox, gbboxes)
            jaccard = jaccard * tf.cast(tf.equal(glabels, rlabel), dtype=jaccard.dtype)

            # Best fit, checking it's above threshold.
            idxmax = tf.cast(tf.argmax(jaccard, axis=0), tf.int32)
            jcdmax = jaccard[idxmax]
            match = jcdmax > matching_threshold
            existing_match = gmatch[idxmax]
            not_difficult = tf.logical_not(gdifficults[idxmax])

            # TP: match & no previous match and FP: previous match | no match.
            # If difficult: no record, i.e FP=False and TP=False.
            tp = tf.logical_and(not_difficult,
                                tf.logical_and(match, tf.logical_not(existing_match)))
            ta_tp = ta_tp.write(i, tp)
            fp = tf.logical_and(not_difficult,
                                tf.logical_or(existing_match, tf.logical_not(match)))
            ta_fp = ta_fp.write(i, fp)

            # Update grountruth match.
            mask = tf.logical_and(tf.equal(grange, idxmax),
                                  tf.logical_and(not_difficult, match))
            gmatch = tf.logical_or(gmatch, mask)

            return [i+1, ta_tp, ta_fp, gmatch]

        # Main loop definition.
        i = 0
        [i, ta_tp_bool, ta_fp_bool, gmatch] = \
            tf.while_loop(m_condition, m_body,
                          [i, ta_tp_bool, ta_fp_bool, gmatch],
                          parallel_iterations=1,
                          back_prop=False)

        # TensorArrays to Tensors and reshape.
        tp_match = tf.reshape(ta_tp_bool.stack(), rshape)
        fp_match = tf.reshape(ta_fp_bool.stack(), rshape)

        return n_gbboxes, tp_match, fp_match

def bboxes_jaccard(bbox_ref, bboxes, name=None):
    with tf.name_scope(name, 'bboxes_jaccard'):
        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)
        vol_anchors = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])

        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)

        # Volumes.
        inter_vol = h * w
        bbox_volum = (bbox_ref[2] - bbox_ref[0]) * (bbox_ref[3] - bbox_ref[1])
        union_vol = vol_anchors + bbox_volum - inter_vol
        jaccard = tf.math.divide(inter_vol, union_vol, 'jaccard')
        return jaccard

def get_init_fun(exclude_scope):
    exclusions = []
    exclusions = [scope.strip() for scope in exclude_scope.split(',')]

    variables_to_restore = []
    for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)
    variables_to_restore = {var.op.name.replace('ssd_300_vgg', 'vgg_16'): var 
                            for var in variables_to_restore}

    return variables_to_restore

def get_variables_to_train(trainable_scope):
    scopes = [scope.strip() for scope in trainable_scope.split(',')]

    variables_to_train = []
    for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.append(variables)
    return variables_to_train