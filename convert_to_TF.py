import os
import numpy as np
import tensorflow as tf
import preprocess
import cv2

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# define clasify: background and car
LABELS = {
    'background': (0, 'Background'),
    'car': (1, 'Car')
}

TRY_ANNOTATIONS = "./train_data/annotation/train_annotations.txt"
TRY_IMAGE = "./train_data/try_image/"
TRY_TFRECORD = "./train_data/try.tfrecords"

# the path of image and label
TRAIN_ANNOTATIONS = "./train_data/annotation/train_annotations.txt"
TRAIN_IMAGE = "./train_data/train_image3"
TRAIN_TFRECORD = "./train_data/train-3.tfrecords"

VALIDATE_ANNOTATIONS = "./train_data/annotation/validate_annotations.txt"
VALIDATE_IMAGE = "./train_data/validate_image"
VALIDATE_TFRECORD = "./train_data/validate.tfrecords"

TEST_ANNOTATIONS = "./train_data/annotation/test_annotations.txt"
TEST_IMAGE = "./train_data/test_image"
TEST_TFRECORD = "./train_data/test.tfrecords"

ANNOTATIONS = [TRAIN_ANNOTATIONS, VALIDATE_ANNOTATIONS, TEST_ANNOTATIONS]
IMAGE = [TRAIN_IMAGE, VALIDATE_IMAGE, TEST_IMAGE]
TFRECORD = [TRAIN_TFRECORD, VALIDATE_TFRECORD, TEST_TFRECORD]


def _int64_feature(value):
    if not isinstance(value, (tuple, list)):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def _float_feature(value):
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def read_file(image_path):
    image_list = []
    annotation_list = []
    for dir in os.listdir(image_path):
        dir_path = os.path.join(image_path, dir)
        for file in os.listdir(dir_path):
            file_path = os.path.join(dir_path, file)
            annotation_name = os.path.join(dir, file)
            print(file_path)
            image = cv2.imread(file_path)
            image_list.append(image)
            annotation_list.append(annotation_name)
    return image_list, annotation_list


def get_info(file, annotation_name):
    bboxes = []
    labels = []
    read = open(file)
    for lines in read.readlines():
        line_list = lines.split(' ')
        if (annotation_name == line_list[0]):
            length = len(line_list) - 1
            bboxes_num = length / 4
            i = 1
            while (bboxes_num > 0):
                bboxes.append(((float(line_list[i]), float(line_list[i+1]), float(line_list[i+2]), float(line_list[i+3]))))
                labels.append((int(LABELS['car'][0])))
                bboxes_num = bboxes_num - 1
                i = i + 4
    if len(bboxes) == 0:
        bboxes.append([0.0, 0.0, 0.0, 0.0])
        labels.append(0)
    read.close()

    return bboxes, labels


def convert_to_TF(sess, image, bboxes, labels):
    """
    Convert to the TFRecord
    :param sess:
    :param image:
    :param bboxes:
    :param labels:
    :return: the example
    """
    # Preprocess the image
    labels = tf.convert_to_tensor(labels)
    image, labels, bboxes = preprocess.preprocess_for_train(image, labels, bboxes)

    image_raw, label_raw, bboxes_raw = sess.run([image, labels, bboxes])

    bboxes_raw = np.transpose(bboxes_raw)
    ymin = bboxes_raw[0].tolist()
    xmin = bboxes_raw[1].tolist()
    ymax = bboxes_raw[2].tolist()
    xmax = bboxes_raw[3].tolist()
    label_raw = label_raw.tolist()

    image_raw = image_raw.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'xmin': _float_feature(xmin),
        'ymin': _float_feature(ymin),
        'xmax': _float_feature(xmax),
        'ymax': _float_feature(ymax),
        'image': _bytes_feature(image_raw),
        'label': _int64_feature(label_raw),
    }))
    return example

def main():
    with tf.Session() as sess:
        writer = tf.python_io.TFRecordWriter(TRY_TFRECORD)
        image_list, annotation_list = read_file(TRY_IMAGE)
        for i, image in enumerate(image_list):
            bboxes, labels = get_info(TRY_ANNOTATIONS, annotation_list[i])
            example = convert_to_TF(sess, image_list[i], bboxes, labels)
            writer.write(example.SerializeToString())
            print('Convert TFRecord: %i' % i)
        writer.close()


if __name__ == '__main__':
    main()