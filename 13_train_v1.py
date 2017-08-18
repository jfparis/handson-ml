# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.platform import tf_logging as logging
import numpy as np
from datetime import datetime
import matplotlib
import matplotlib.pyplot as plt
import random
import sys
import tarfile
from six.moves import urllib
from dataset_utils import _dataset_exists, _get_filenames_and_classes, write_label_file, _convert_dataset
import os.path
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
import time

from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim

INCEPTION_PATH = os.path.join("datasets", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()

def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)

FLOWERS_PATH = os.path.join("datasets", "flowers")
height = 299
width = 299
channels = 3

flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', FLOWERS_PATH, 'String: Your dataset directory')

# The number of shards to split the dataset into.
flags.DEFINE_integer('num_shards', 2, 'Int: Number of shards to split the TFRecord files into')

#Output filename for the naming the TFRecord file
flags.DEFINE_string('tfrecord_filename', "Flowers", 'String: The output filename to name your TFRecord file')

FLAGS = flags.FLAGS

#State the labels file and read it
labels_file = os.path.join(FLOWERS_PATH, 'labels.txt')
labels = open(labels_file, 'r')

#Create a dictionary to refer each label to their string name
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1] #Remove newline
    labels_to_name[int(label)] = string_name

#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = 'Flowers_%s_*.tfrecord'

#Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}

def get_split(split_name, dataset_dir, file_pattern=file_pattern,):
    """
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later
    on. This function will set up the decoder and dataset information all into one Dataset class so that you can avoid
    the brute work later on.

    Your file_pattern is very important in locating the files later.

    INPUTS:
        - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
        - dataset_dir(str): the dataset directory where the tfrecord files are located
        - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation.
    """
    # First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    # Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    # Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = 'Flowers_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) \
                          if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    # Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
        'image/class/label': tf.FixedLenFeature(
            [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    # Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Image(),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    # Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    # Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern_path,
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        num_classes=5,
        labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset

from scipy.misc import imresize

def prepare_image(image, target_width=299, target_height=299, max_zoom=0.2):
    """Zooms and crops the image randomly for data augmentation."""

    # First, let's find the largest bounding box with the target size ratio that fits within the image
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = tf.cond(crop_vertically,
                         lambda: width,
                         lambda: height * target_image_ratio)
    crop_height = tf.cond(crop_vertically,
                          lambda: width / target_image_ratio,
                          lambda: height)

    # Now let's shrink this bounding box by a random factor (dividing the dimensions by a random number
    # between 1.0 and 1.0 + `max_zoom`.
    resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + max_zoom)
    crop_width = tf.cast(crop_width / resize_factor, tf.int32)
    crop_height = tf.cast(crop_height / resize_factor, tf.int32)
    box_size = tf.stack([crop_height, crop_width, 3])  # 3 = number of channels

    # Let's crop the image using a random bounding box of the size we computed
    image = tf.random_crop(image, box_size)

    # Let's also flip the image horizontally with 50% probability:
    image = tf.image.random_flip_left_right(image)

    # The resize_bilinear function requires a 4D tensor (a batch of images)
    # so we need to expand the number of dimensions first:

    # Finally, let's resize the image to the target dimensions. Note that this function
    # returns a float32 tensor.
    image = tf.image.resize_images(image, [target_height, target_width])
    return image


def load_batch(dataset, batch_size, height=height, width=width, is_training=True):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    #Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    image = prepare_image(raw_image, height, width, 0 if is_training else 0.2)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [tf.reshape(image,(299,299,3)), raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels


#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 70

#State your batch size
batch_size = 10

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

# First create the dataset and load one batch
dataset = get_split('train', FLAGS.dataset_dir, file_pattern=file_pattern)
images, _, labels = load_batch(dataset, batch_size=batch_size)

# Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = int(dataset.num_samples / batch_size)
num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

# Create the model inference
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(images, num_classes=1001, is_training=True)

inception_saver = tf.train.Saver()

# Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])

with tf.name_scope("new_output_layer"):
    flower_logits = tf.layers.dense(prelogits, 5, name="flower_logits")
    probabilities = tf.nn.softmax(flower_logits, name="probabilities")
    predictions = tf.argmax(probabilities, axis=1, name="predictions")

with tf.name_scope("train"):
    xentropy = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=flower_logits)
    loss = tf.reduce_mean(xentropy)
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flower_logits")

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(flower_logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Create the global step for monitoring the learning_rate and training.
global_step = get_or_create_global_step()

# Define your exponentially decaying learning rate
lr = tf.train.exponential_decay(
    learning_rate=initial_learning_rate,
    global_step=global_step,
    decay_steps=decay_steps,
    decay_rate=learning_rate_decay_factor,
    staircase=True)

# Now we can define the optimizer that takes on the learning rate
optimizer = tf.train.AdamOptimizer(learning_rate=lr)

# Create the train_op.
train_op = optimizer.minimize(loss, var_list=flower_vars)

# Now finally create all the summaries you need to monitor and group them into one summary op.
tf.summary.scalar('losses/Total_Loss', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('learning_rate', lr)
my_summary_op = tf.summary.merge_all()


# Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
def train_step(sess, train_op, global_step):
    '''
    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
    '''
    # Check the time for each sess run
    start_time = time.time()
    _, global_step_count, loss_val = sess.run([train_op, global_step, loss])
    time_elapsed = time.time() - start_time

    # Run the logging to print some results
    logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, loss_val, time_elapsed)

    return loss_val, global_step_count


# Now we create a saver function that actually restores the variables from a checkpoint file in a sess
saver = tf.train.Saver()
init = tf.global_variables_initializer()


def restore_fn(sess):
    sess.run(init)
    return inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)


# Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
sv = tf.train.Supervisor(logdir=log_dir("inception"), summary_op=None, init_fn=restore_fn)

# Run the managed session
with sv.managed_session() as sess:
    for step in range(num_steps_per_epoch * num_epochs):
        # At the start of every epoch, show the vital information:
        if step % num_batches_per_epoch == 0:
            logging.info('Epoch %s/%s', step / num_batches_per_epoch + 1, num_epochs)
            learning_rate_value, accuracy_value = sess.run([lr, accuracy])
            logging.info('Current Learning Rate: %s', learning_rate_value)
            logging.info('Current Streaming Accuracy: %s', accuracy_value)

            # optionally, print your logits and predictions for a sanity check that things are going fine.
            flower_logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                [flower_logits, probabilities, predictions, labels])
            print('logits: \n', flower_logits_value)
            print('Probabilities: \n', probabilities_value)
            print('predictions: \n', predictions_value)
            print('Labels:\n:', labels_value)

        # Log the summaries every 10 step.
        if step % 10 == 0:
            loss, _ = train_step(sess, train_op, sv.global_step)
            print(loss)
            summaries = sess.run(my_summary_op)
            sv.summary_computed(sess, summaries)

        # If not, simply run the training step
        else:
            loss, _ = train_step(sess, train_op, sv.global_step)
            print(loss)

    # We log the final training loss and accuracy
    logging.info('Final Loss: %s', loss)
    logging.info('Final Accuracy: %s', sess.run(accuracy))

    # Once all the training has been done, save the log files and checkpoint model
    logging.info('Finished training! Saving model to disk now.')
    sv.saver.save(sess, sv.save_path, global_step=sv.global_step)


