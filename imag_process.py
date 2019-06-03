import tensorflow as tf
import preprocess
import matplotlib.pyplot as plt
import numpy as np


FILE = './img.jpg'

image_raw_data = tf.gfile.GFile(FILE, 'rb').read()
image = tf.image.decode_jpeg(image_raw_data)
image = tf.image.convert_image_dtype(image, dtype=tf.float32)
image = tf.reshape(image, shape=[540, 960, 3])
shape = image.get_shape().as_list()
labels = tf.constant([0])
bbox = tf.constant([[[0.0, 0.0, 0.0, 0.0]]])

bbox = tf.squeeze(bbox,axis=0)

# with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess:
with tf.Session() as sess:
    process_image, process_labels, process_bbox = preprocess.preprocess_for_train(image, labels, bbox)

    image_raw, label_raw, bboxes_raw = sess.run([process_image, process_labels, process_bbox])

    image_raw = image_raw.tobytes()
    image_raw = tf.decode_raw(image_raw, tf.float32)
    image_raw = tf.reshape(image_raw, [300, 300, 3])

    # image_raw = np.expand_dims(image_raw, axis=0)
    image = tf.expand_dims(image, axis=0)
    bboxes_raw = np.expand_dims(bboxes_raw, axis=0)
    # distorted = np.expand_dims(distorted, axis=0)
    # distorted = np.expand_dims(distorted, axis=0)

    # result = tf.image.draw_bounding_boxes(image_raw, bboxes_raw)
    # result = tf.squeeze(result)
    # print sess.run([process_bbox, process_labels])
    # distorted_image = tf.image.draw_bounding_boxes(image, distorted)
    # distorted_image = tf.squeeze(distorted_image)
    print(image_raw.eval())
    # plt.figure(1)
    # plt.imshow(result.eval())
    # plt.figure(2)
    # plt.imshow(distorted_image.eval())
    plt.show()
    # bboxes = tf.constant([[0.1, 0.2, 0.7, 0.8], [0.3, 0.5, 0.6, 0.7]])
    # begin, size, distort_bbox = tf.image.sample_distorted_bounding_box(
    #     tf.shape(image),
    #     bounding_boxes=tf.expand_dims(bboxes, 0),
    #     min_object_covered=0.3,
    #     aspect_ratio_range=(0.9, 1.1),
    #     area_range=(0.1, 1.0),
    #     max_attempts=200,
    #     use_image_if_no_bounding_boxes=True,
    #     name=None)
    # image = tf.expand_dims(image, 0)
    # image_with_bbox = tf.image.draw_bounding_boxes(image, distort_bbox)
    # image_with_bbox = tf.squeeze(image_with_bbox)
    # plt.imshow(image_with_bbox.eval())
    # plt.show()

# with tf.Session() as sess:
#     x = tf.constant([[[1, 1], [2, 2], [3, 3], [4, 4]], [[1, 1], [2, 2], [3, 3], [4, 4]]])
#     indix = tf.constant()
#     y = tf.gather(x, indix)
#     print sess.run(y)
