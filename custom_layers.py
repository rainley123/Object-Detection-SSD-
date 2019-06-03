# Custom layers for SSD
import tensorflow as tf


def pad2d(inputs,
          pad=(0, 0),
          mode='CONSTANT',
          scope=None):
    """
    :param inputs:
    :param pad:
    :param mode:
    :param scope:
    :return: the net after pad
    """
    with tf.name_scope(scope, 'pad2d', [inputs]):
        paddings = [[0, 0], [pad[0], pad[0]], [pad[1], pad[1]], [0, 0]]
        net = tf.pad(inputs, paddings, mode=mode)
        return net


def l2_normalization(inputs, trainable=True, scope='L2Normalization'):
    """
    :param inputs:
    :param trainable:
    :param scope:
    :return: the net after normalization
    """
    n_channels = inputs.get_shape().as_list()[-1]
    l2_norm = tf.nn.l2_normalize(inputs, [3], epsilon=1e-12)
    with tf.variable_scope(scope):
        gamma = tf.get_variable("gamma", shape=[n_channels, ], dtype=tf.float32,
                                initializer=tf.ones_initializer(), trainable=trainable)
        return l2_norm * gamma


def abs_smooth(x):
    """
    Define as
    x^2 /2      if abs(x)<1;
    abs(x)-0.5  if abs(x)>1
    :param x:
    :return: x after smooth
    """
    absx = tf.abs(x)
    minx = tf.minimum(absx, 1)
    r = 0.5 * ((absx - 1) * minx + absx)

    return r