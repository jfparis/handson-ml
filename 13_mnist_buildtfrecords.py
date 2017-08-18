# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import tensorflow as tf
import random
import sys
import tarfile
from six.moves import urllib
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
import os.path
import math


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# load the training set
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("datasets/")

X_train = mnist.train.images
X_test = mnist.test.images
X_validation = mnist.validation.images
y_train = mnist.train.labels.astype("int")
y_test = mnist.test.labels.astype("int")
y_validation = mnist.validation.labels.astype("int")

flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', "datasets/mnist", 'String: Your dataset directory')

# The number of shards to split the dataset into.
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files into')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', "mnist", 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

def get_dataset_filename(dataset_dir, split_name, shard_id, tfrecord_filename, nb_shards):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
        tfrecord_filename, split_name, shard_id, nb_shards)
    return os.path.join(dataset_dir, output_filename)

def build_tfrecord(split_name, dataset, labels, dataset_dir, tfrecord_filename, nb_shards):
    """Converts the given filenames to a TFRecord dataset.

    Args:
        split_name: The name of the dataset, either 'train' or 'validation' or 'validation'.
        filenames: A list of absolute paths to png or jpg images.
        class_names_to_ids: A dictionary from class names (strings) to ids
            (integers).
        dataset_dir: The directory where the converted datasets are stored.
    """
    assert split_name in ['train', 'test', 'validation']

    num_per_shard = int(math.ceil(dataset.shape[0] / float(nb_shards)))
    i = 0

    with tf.Graph().as_default():

        with tf.Session('') as sess:

            for shard_id in range(nb_shards):
                output_filename = get_dataset_filename(
                    dataset_dir, split_name, shard_id, tfrecord_filename = tfrecord_filename, nb_shards = nb_shards)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, dataset.shape[0])

                    for i in range(start_ndx, end_ndx):
                        sys.stdout.write('\r>> Converting record %d/%d shard %d' % (
                            i+1, dataset.shape[0], shard_id))
                        sys.stdout.flush()

                        # Read the filename:
                        data = dataset[i]
                        print(data.dtype)

                        height, width = 28,28

                        class_id = labels[i]

                        example = tf.train.Example(features=tf.train.Features(feature={
                            'height': _int64_feature(height),
                            'width': _int64_feature(width),
                            'image': _bytes_feature(data.tostring()),
                            'label': _int64_feature(class_id)}))
                        tfrecord_writer.write(example.SerializeToString())
                        i = i + 1

    sys.stdout.write('\n')
    sys.stdout.flush()

build_tfrecord('train', X_train, y_train,
                dataset_dir = FLAGS.dataset_dir,
                tfrecord_filename = FLAGS.tfrecord_filename,
                nb_shards = FLAGS.num_shards)
build_tfrecord('test', X_test, y_test,
                dataset_dir = FLAGS.dataset_dir,
                tfrecord_filename = FLAGS.tfrecord_filename,
                nb_shards = FLAGS.num_shards)
build_tfrecord('validation', X_validation, y_validation,
                dataset_dir = FLAGS.dataset_dir,
                tfrecord_filename = FLAGS.tfrecord_filename,
                nb_shards = FLAGS.num_shards)