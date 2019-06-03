import tensorflow as tf

DIR_TFRECORDS = "./train_data/try.tfrecords"

SHUFFLE_BUFFER = 10
BATCH_SIZE = 10
PADDED_SHAPES = ([300, 300, 3], [10], [10, 4])

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
        })

    image = features['image']
    xmin = tf.sparse.to_dense(features['xmin'])
    ymin = tf.sparse.to_dense(features['ymin'])
    xmax = tf.sparse.to_dense(features['xmax'])
    ymax = tf.sparse.to_dense(features['ymax'])
    label = tf.sparse.to_dense(features['label'])

    # convert string to int
    image = tf.decode_raw(image, tf.float32)
    image = tf.reshape(image, [300, 300, 3])
    xmin = tf.reshape(xmin, shape=[10, ])
    ymin = tf.reshape(ymin, shape=[10, ])
    xmax = tf.reshape(xmax, shape=[10, ])
    ymax = tf.reshape(ymax, shape=[10, ])
    label = tf.reshape(label, shape=[10, ])

    bbox = tf.stack([ymin, xmin, ymax, xmax], axis=1)

    return image, label, bbox

with tf.Session() as sess:
    input_files = [DIR_TFRECORDS]
    dataset = tf.data.TFRecordDataset(input_files)

    dataset = dataset.map(parser)

    dataset = dataset.shuffle(SHUFFLE_BUFFER).padded_batch(batch_size=2, padded_shapes=PADDED_SHAPES)

    iterator = dataset.make_one_shot_iterator()

    image, label, bbox = iterator.get_next()
    # return image, label, bbox
    while True:
        try:
            print(sess.run([image, label, bbox]))
        except tf.errors.OutOfRangeError:
            break