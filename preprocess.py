import tensorflow as tf
import random
import numpy as np

OUT_SIZE = (300, 300)
MEANS = [123, 117, 104]


def imresize(image):
    """
    Resize the image to (300, 300, 3)
    :param image:
    :return: the resized image
    """
    resize_image = tf.image.resize_images(image, OUT_SIZE, method=0)
    return resize_image


def imflip(image, bbox):
    """
    Flip the image and bbox
    :param image:
    :param bbox:
    :return: flipped image and bbox
    """
    flipped_image = tf.image.flip_left_right(image)

    bbox = tf.transpose(bbox)
    ymin = bbox[0]
    xmin = 0 - bbox[3] + 1
    ymax = bbox[2]
    xmax = 0 - bbox[1] + 1

    out_bbox = tf.transpose(tf.stack([ymin, xmin, ymax, xmax], axis=0))
    return flipped_image, out_bbox


def whitened(image, means=MEANS):
    """
    Subtracts the each image channel
    :param image:
    :param means:
    :return: the centered image
    """
    mean = tf.constant(means, dtype=tf.float32)
    image = image - mean
    return image


def disorted_bounding_box_crop(image, labels, bboxes,
                               min_object_covered=0.3,
                               aspect_ratio_range=(0.9, 1.1),
                               area_range=(0.1, 1.0),
                               max_attempts=200,
                               clip_bboxes=True,
                               scope='distorted_bounding_box_crop'):
    """
    :param image: H x W x C
    :param labels:
    :param bboxes: N x 4 (have only two shapes)
    :param min_object_covered:
    :param aspect_ratio_range:
    :param area_range:
    :param max_attempts:
    :param clip_bboxes:
    :param scope:
    :return: cropped image and bbox
    """
    with tf.name_scope(scope, [image, bboxes]):
        begin, size, distort_bbox = tf.image.sample_distorted_bounding_box(
            tf.shape(image),
            bounding_boxes=tf.expand_dims(bboxes, 0),
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True,
            name=None)
        distort_bbox = distort_bbox[0, 0]               # distort_bbox shape [1, 1, 4],so get ymin xmin ymax xmax

        # Crop the image
        cropped_image = tf.slice(image, begin, size)
        cropped_image.set_shape([None, None, 3])

        # Update the bounding box
        labels, bboxes = bboxes_filter_overlap(labels, bboxes, distort_bbox)
        bboxes = bboxes_resize(distort_bbox, bboxes)


        return cropped_image, labels, bboxes, distort_bbox


def bboxes_resize(bbox_ref, bboxes, name=None):
    """
    Calculate the relative bbox to the reference bbox
    :param bbox_ref:
    :param bboxes:
    :param name:
    :return: the relative bbox
    """
    with tf.name_scope(name, 'bboxes_resize'):

        # Translate.
        v = tf.stack([bbox_ref[0], bbox_ref[1], bbox_ref[0], bbox_ref[1]])
        bboxes = bboxes - v

        # Scale.
        s = tf.stack([bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1],
                      bbox_ref[2] - bbox_ref[0],
                      bbox_ref[3] - bbox_ref[1]])
        bboxes = bboxes / s

        return bboxes


def bboxes_filter_overlap(labels, bboxes, ref_bboxes, scope=None):
    """
    Remove the bbox which is out of the reference bbox
    :param labels:
    :param bboxes:
    :param scope:
    :return: the new bboxes
    """
    with tf.name_scope(scope, 'bboxes_filter', [labels, bboxes]):
        scores = bboxes_intersection(ref_bboxes, bboxes)
        mask = scores > 0.5
        labels = tf.boolean_mask(labels, mask)
        bboxes = tf.boolean_mask(bboxes, mask)

        return labels, bboxes


def bboxes_intersection(bbox_ref, bboxes, name=None):
    """
    :param bbox_ref: (N, 4)
    :param bboxes: (N, 4)
    :param name:
    :return: Tensor with relative intersection
    """
    with tf.name_scope(name, 'bboxes_intersection'):

        # Should be more efficient to first transpose.
        bboxes = tf.transpose(bboxes)
        bbox_ref = tf.transpose(bbox_ref)

        # Intersection bbox and volume.
        int_ymin = tf.maximum(bboxes[0], bbox_ref[0])
        int_xmin = tf.maximum(bboxes[1], bbox_ref[1])
        int_ymax = tf.minimum(bboxes[2], bbox_ref[2])
        int_xmax = tf.minimum(bboxes[3], bbox_ref[3])
        h = tf.maximum(int_ymax - int_ymin, 0.)
        w = tf.maximum(int_xmax - int_xmin, 0.)

        # Volumes.
        inter_vol = h * w
        bboxes_vol = (bboxes[2] - bboxes[0]) * (bboxes[3] - bboxes[1])
        scores = tf.math.divide(inter_vol, bboxes_vol)
        return scores


def preprocess_for_train(image, shape, labels, bboxes, out_shape=OUT_SIZE, scope='ssd_preprocessing_train'):
    """
    Preprocess the image, labels and bboxes
    :param image:
    :param labels:
    :param bboxes:
    :param out_shape:
    :param scope:
    :return: image labels and bboxes after preprocess
    """
    with tf.name_scope(scope, 'ssd_preprocessing_train', [image, labels, bboxes]):
        if len(image.get_shape().as_list()) != 3:
            raise ValueError('Input must have 3 shapes H W C')

        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)

        # Get the relative bbox
        bboxes = tf.transpose(bboxes)
        ymin = tf.div(bboxes[0], tf.cast(shape[0], tf.float32))
        xmin = tf.div(bboxes[1], tf.cast(shape[1], tf.float32))
        ymax = tf.div(bboxes[2], tf.cast(shape[0], tf.float32))
        xmax = tf.div(bboxes[3], tf.cast(shape[1], tf.float32))
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        # Randomly crop the image, labels and bboxes
        if random.randint(0, 1) == 0:
            image, labels, bboxes, disorted_bbox = disorted_bounding_box_crop(image, labels, bboxes)

        # Resize the image
        image = imresize(image)

        # Randomly flip the image
        if random.randint(0, 1) == 0:
            image, bboxes = imflip(image, bboxes)

        # Compare the border
        bboxes = tf.transpose(bboxes)
        ymin = tf.maximum(bboxes[0], 0.0)
        xmin = tf.maximum(bboxes[1], 0.0)
        ymax = tf.minimum(bboxes[2], 1.0)
        xmax = tf.minimum(bboxes[3], 1.0)
        bboxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        image = image * 255
        image = whitened(image)

        return image, labels, bboxes



