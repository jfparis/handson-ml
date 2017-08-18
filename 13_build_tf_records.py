# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import tensorflow as tf
import random
import sys
import tarfile
from six.moves import urllib
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
import os.path

FLOWERS_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_PATH = os.path.join("datasets", "flowers")

def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
    if os.path.exists(FLOWERS_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photos.tgz")
    urllib.request.urlretrieve(url, tgz_path)
    flowers_tgz = tarfile.open(tgz_path)
    flowers_tgz.extractall(path=path)
    flowers_tgz.close()
    os.remove(tgz_path)
fetch_flowers()

flowers_root_path = os.path.join(FLOWERS_PATH, "flower_photos")

flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', FLOWERS_PATH, 'String: Your dataset directory')

# Proportion of dataset to be used for evaluation
flags.DEFINE_float('validation_size', 0.3, 'Float: The proportion of examples in the dataset to be used for validation')

# The number of shards to split the dataset into.
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files into')

# Seed for repeatability.
flags.DEFINE_integer('random_seed', 0, 'Int: Random seed to use for repeatability.')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', "Flowers", 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

photo_filenames, class_names = _get_filenames_and_classes(FLAGS.dataset_dir)
class_names_to_ids = dict(zip(class_names, range(len(class_names))))

#Find the number of validation examples we need
num_validation = int(FLAGS.validation_size * len(photo_filenames))

# Divide the training datasets into train and test:
random.seed(FLAGS.random_seed)
random.shuffle(photo_filenames)
training_filenames = photo_filenames[num_validation:]
validation_filenames = photo_filenames[:num_validation]

# First, convert the training and validation sets.
_convert_dataset('train', training_filenames, class_names_to_ids,
                 dataset_dir = FLAGS.dataset_dir,
                 tfrecord_filename = FLAGS.tfrecord_filename,
                 _NUM_SHARDS = FLAGS.num_shards)
_convert_dataset('validation', validation_filenames, class_names_to_ids,
                 dataset_dir = FLAGS.dataset_dir,
                 tfrecord_filename = FLAGS.tfrecord_filename,
                 _NUM_SHARDS = FLAGS.num_shards)

# Finally, write the labels file:
labels_to_class_names = dict(zip(range(len(class_names)), class_names))
write_label_file(labels_to_class_names, FLAGS.dataset_dir)

print('\nFinished converting the %s dataset!' % (FLAGS.tfrecord_filename))
