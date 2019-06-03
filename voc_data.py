import os
import numpy as np
import tensorflow as tf
import preprocess
import cv2
import xml.etree.cElementTree as ET

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# define clasify: background and car
LABELS = {
    'background': (0, 'Background'),
    'person': (1, 'Person'),
    'bird': (2, 'Bird'),
    'cat': (3, 'Cat'),
    'cow': (4, 'Cow'),
    'dog': (5, 'Dog'),
    'horse': (6, 'Horse'),
    'sheep': (7, 'Sheep'),
    'aeroplane': (8, 'Aeroplane'),
    'bicycle': (9, 'Bicycle'),
    'boat': (10, 'Boat'),
    'bus': (11, 'Bus'),
    'car': (12, 'Car'),
    'motorbike': (13, 'Motorbike'),
    'train': (14, 'Train'),
    'bottle': (15, 'Bottle'),
    'chair': (16, 'Chair'),
    'diningtable': (17, 'Diningtable'),
    'pottedplant': (18, 'Pottedplant'),
    'sofa': (19, 'Sofa'),
    'tvmonitor': (20, 'TV')
}

ANNOTATIONS = './VOC_train/Annotations'
TRAIN_IMAGE = './VOC_train/train_JPEG'
VAL_IMAGE = './VOC_train/val_JPEG'

TRAIN_TF = './VOC_train/train.tfrecords'
VAL_TF = './VOC_train/val.tfrecords'

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

def read_file(image_path, annotation_path):
    image_name = []
    annotation_name = []
    for JPEG_name in os.listdir(image_path):
        # JPEG_path = os.path.join(image_path, files)
        xml_name = os.path.splitext(JPEG_name)[0] + ".xml"
        # xml_path = os.path.join(annotation_path, xml_name)

        image_name.append(JPEG_name)
        annotation_name.append(xml_name)

    return image_name, annotation_name

def get_info(image_path, annotation_path, image, annotation):
    bboxes = []
    labels = []
    JPEG_path = os.path.join(image_path, image)
    xml_path = os.path.join(annotation_path, annotation)

    raw_image = cv2.imread(JPEG_path)

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    depth = int(size.find('depth').text)

    for obj in root.findall('object'):
        name = obj.find('name').text
        bndbox = obj.find('bndbox')
        xmin = bndbox.find('xmin').text
        ymin = bndbox.find('ymin').text
        xmax = bndbox.find('xmax').text
        ymax = bndbox.find('ymax').text

        labels.append((int(LABELS[name][0])))
        bboxes.append((float(ymin), float(xmin), float(ymax), float(xmax)))
 
    return raw_image, width, height, depth, labels, bboxes

def convert_to_TF(sess, image, width, height, depth, bboxes, labels):
    bboxes = np.transpose(bboxes)
    ymin = bboxes[0].tolist()
    xmin = bboxes[1].tolist()
    ymax = bboxes[2].tolist()
    xmax = bboxes[3].tolist()

    image_raw = image.tobytes()

    example = tf.train.Example(features=tf.train.Features(feature={
        'xmin': _float_feature(xmin),
        'ymin': _float_feature(ymin),
        'xmax': _float_feature(xmax),
        'ymax': _float_feature(ymax),
        'image': _bytes_feature(image_raw),
        'label': _int64_feature(labels),
        'width': _int64_feature(width),
        'height': _int64_feature(height),
        'depth': _int64_feature(depth)
    }))
    return example


def main():
    with tf.Session() as sess:
        image_name, annotation_name = read_file(TRAIN_IMAGE, ANNOTATIONS)
        writer = tf.python_io.TFRecordWriter(TRAIN_TF)
        for i, image in enumerate(image_name):
            raw_image, width, height, depth, labels, bboxes = get_info(TRAIN_IMAGE, ANNOTATIONS, image_name[i], annotation_name[i])
            example = convert_to_TF(sess, raw_image, width, height, depth, bboxes, labels)
            writer.write(example.SerializeToString())
            print('Convert TFRecord: %i' % i)
        writer.close()
        
        print([index, global_name])
        


if __name__ == "__main__":
    main()




