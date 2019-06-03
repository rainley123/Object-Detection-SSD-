import tensorflow as tf
import SSD_net
import utils
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

TRAIN_TFRECORDS = "./VOC_train/train.tfrecords"
VAL_TFRECORDS = "./VOC_train/val.tfrecords"

# TRY_TFRECORDS = './VOC_train/train-try.tfrecords'

CHECKPOINT_PATH = './checkpoint_file/vgg_16.ckpt'
TRAIN_FILE = "./save_model"

CHECKPOINT_EXCLUDE_SCOPE =  "ssd_300_vgg/conv6,ssd_300_vgg/conv7,ssd_300_vgg/block8,\
                            ssd_300_vgg/block9,ssd_300_vgg/block10,ssd_300_vgg/block11,\
                            ssd_300_vgg/block4_box,ssd_300_vgg/block7_box,ssd_300_vgg/block8_box,\
                            ssd_300_vgg/block9_box,ssd_300_vgg/block10_box,ssd_300_vgg/block11_box"
                            
SHUFFLE_BUFFER = 10000
BATCH_SIZE = 20
NUM_EPOCHES = 200
PADDED_SHAPES = ([300, 300, 3], [utils.PADDED_LENGTH], [utils.PADDED_LENGTH, 4])
LEARNING_RATE = 0.000001
NUM_SMAPLE_PER_EPOCH = 4000
LEARNING_RATE_DECAY_FACTOR = 0.94
AVERAGE_DECAY = 0.99
RMS_DECAY = 0.9
RMS_MOMENTUM = 0.9
RMS_EPSILON = 1.0

def main():
    with tf.Graph().as_default():
        # Get the SSD network and its anchors
        ssd_net = SSD_net.SSDNet()
        ssd_shape = ssd_net.params.img_shape
        ssd_anchors = ssd_net.anchors(ssd_shape)

        # Create global_step
        global_step = tf.train.create_global_step()

        # Define the train dataset
        train_files = [TRAIN_TFRECORDS]
        train_dataset = tf.data.TFRecordDataset(train_files)
        train_dataset = train_dataset.map(utils.parser)
        train_dataset = train_dataset.shuffle(SHUFFLE_BUFFER).padded_batch(batch_size=BATCH_SIZE,
                                                                           padded_shapes=PADDED_SHAPES)

        # Define the validate dataset
        val_files = [VAL_TFRECORDS]
        val_dataset = tf.data.TFRecordDataset(val_files)
        val_dataset = val_dataset.map(utils.parser)
        val_dataset = val_dataset.shuffle(SHUFFLE_BUFFER).padded_batch(batch_size=BATCH_SIZE,
                                                                       padded_shapes=PADDED_SHAPES)

        # # Define the test dataset
        # test_files = [TEST_TFRECORDS]
        # test_dataset = tf.data.TFRecordDataset(test_files)
        # test_dataset = test_dataset.map(utils.parser)
        # test_dataset = test_dataset.shuffle(SHUFFLE_BUFFER).padded_batch(batch_size=BATCH_SIZE,
        #                                                                  padded_shapes=PADDED_SHAPES)

        # Define the common iterator
        iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                                   train_dataset.output_shapes)

        image, label, bbox = iterator.get_next()
        image_summary = tf.summary.image("image_summary", image)

        # Encode the groundtruth labels and bboxes
        gclasses, gscores, glocalisations = ssd_net.bboxes_encode(label, bbox, ssd_anchors, BATCH_SIZE)

        # Construct the SSD net
        arg_scope = ssd_net.arg_scope()
        with slim.arg_scope(arg_scope):
            predictions, localisations, logits, end_points = ssd_net.net(image)

        # Define loss function
        ssd_net.losses(logits, localisations, gclasses, glocalisations, gscores, BATCH_SIZE)
        losses = tf.losses.get_total_loss()
        loss_summary = tf.summary.scalar("loss", losses)

        # Decode the prediction
        decoded_localisations = ssd_net.bboex_decode(localisations, ssd_anchors)

        # Calculate the precision and recall
        true_dict_scores, true_dict_bboxes = ssd_net.detected_bboxes(predictions, decoded_localisations, select_threshold=0.01, nms_threshold=0.45)

        gdifficults = tf.zeros(shape=[BATCH_SIZE, utils.PADDED_LENGTH], dtype=tf.int64)
        num_gbboxes, tp, fp, rscores = utils.bboxes_matching_batch(true_dict_scores.keys(), true_dict_scores, true_dict_bboxes, label, bbox, gdifficults, matching_threshold=0.5)
        accuracy = utils.precision(tp, fp)
        accuracy_summary = tf.summary.scalar("accuracy", accuracy)

        # Define the moving average
        moving_average_variables = slim.get_model_variables()
        ema = tf.train.ExponentialMovingAverage(AVERAGE_DECAY, global_step)
        average_op = ema.apply(moving_average_variables)

        # Define the learning rate
        decay_step = int(NUM_SMAPLE_PER_EPOCH / BATCH_SIZE * 2.0)
        learning_rate = tf.train.exponential_decay(LEARNING_RATE, global_step, decay_step,
                                                   LEARNING_RATE_DECAY_FACTOR, staircase=True)
        learning_rate_summary = tf.summary.scalar("learning_rate", learning_rate)

        # Define the train_step
        train_step = tf.train.RMSPropOptimizer(learning_rate, decay=RMS_DECAY, momentum=RMS_MOMENTUM,
                                               epsilon=RMS_EPSILON).minimize(losses, global_step=global_step)

        train_op = tf.group(train_step, average_op)
        summary_merged = tf.summary.merge_all()

        init_fun = slim.assign_from_checkpoint_fn(CHECKPOINT_PATH,
                        utils.get_init_fun(CHECKPOINT_EXCLUDE_SCOPE),
                        ignore_missing_vars=True)

        saver = tf.train.Saver()
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.8

        with tf.Session(config=config) as sess:
            sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))
            init_fun(sess)
            summary_writer = tf.summary.FileWriter("./log", sess.graph)
            # sess.run(iterator.make_initializer(train_dataset))
            # while True:
            #     try:
            #         image_raw, label_raw, bbox_raw = sess.run([image, label, bbox])
            #         print([image_raw, label_raw, bbox_raw])
            #         result = tf.image.draw_bounding_boxes(image_raw, bbox_raw)
            #         result = tf.squeeze(result)
            #         plt.imshow(result.eval())
            #         plt.show()
            #     except tf.errors.OutOfRangeError:
            #         break

            for EPOCH in range(NUM_EPOCHES):
                sess.run(iterator.make_initializer(train_dataset))
                i = 0
                while True:
                    try:
                        _, summary_result = sess.run([train_op, summary_merged]) 
                        print("EPOCH %d, Batch %d completed" % (EPOCH, i))
                        i = i + 1
                    except tf.errors.OutOfRangeError:
                        break
                summary_writer.add_summary(summary_result, EPOCH)

                saver.save(sess, TRAIN_FILE, global_step=global_step)
                accuracy_list = []
                sess.run(iterator.make_initializer(val_dataset))
                while True:
                    try:
                        val_accuracy = sess.run(accuracy)
                        accuracy_list.append(val_accuracy)
                    except tf.errors.OutOfRangeError:
                        break
                val_accuracy_mean = tf.reduce_mean(accuracy_list)
                print("EPOCH %d : validation accuracy = %.1f%%" % (EPOCH, val_accuracy_mean.eval() * 100))
            summary_writer.close()

            # sess.run(iterator.make_initializer(test_dataset))
            # accuracy_list = []
            # while True:
            #     try:
            #         test_accuracy = sess.run(accuracy)
            #         accuracy_list.append(test_accuracy)
            #     except tf.errors.OutOfRangeError:
            #         break
            # test_accuracy_mean = tf.reduce_mean(accuracy_list)
            # print("test accuracy = %.1f%%" % (test_accuracy_mean.eval() * 100))


if __name__ == '__main__':
    main()