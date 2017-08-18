# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
from  datetime import datetime
import os.path
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

slim = tf.contrib.slim

height = 28
width = 28
channels = 1
n_inputs = height * width

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def log_dir(prefix="", date=True):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run" + "" if not date else "-" + now
    return "{}/{}/".format(root_logdir, name)


#with tf.name_scope("inputs"):
#    X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
#    X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#    y = tf.placeholder(tf.int32, shape=[None], name="y")

flags = tf.app.flags

#State your dataset directory
flags.DEFINE_string('dataset_dir', "datasets/mnist", 'String: Your dataset directory')

FLAGS = flags.FLAGS

#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = 'mnist_%s_*.tfrecord'

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
    if split_name not in ['train', 'validation', 'test']:
        raise ValueError('The split_name %s is not recognized. Please input either train or validation as the split_name' % (split_name))

    # Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    # Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = 'mnist_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) \
                          if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    # Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    # Create the keys_to_features dictionary for the decoder
    keys_to_features = {
        'height': tf.FixedLenFeature((), tf.int64),
        'width': tf.FixedLenFeature((), tf.int64),
        'image': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature((), tf.int64),
    }

    # Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
        'image': slim.tfexample_decoder.Tensor('image'),
        'label': slim.tfexample_decoder.Tensor('label'),
    }

    # Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    # Create the labels_to_name file
    #labels_to_name_dict = labels_to_name

    # Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern_path,
        decoder=decoder,
        reader=reader,
        num_readers=4,
        num_samples=num_samples,
        num_classes=5,
        #labels_to_name=labels_to_name_dict,
        items_to_descriptions=items_to_descriptions)

    return dataset

def load_batch(dataset, batch_size, height=height, width=width):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing

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
    image = tf.reshape(tf.decode_raw(raw_image, tf.float32), (height, width,1))


    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, labels = tf.train.batch(
        [image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, labels

n_epochs = 20
batch_size = 500

with tf.name_scope("inputs"):
    dataset = get_split('train', FLAGS.dataset_dir, file_pattern=file_pattern)
    images, labels = load_batch(dataset,batch_size, height, width)

num_batches_per_epoch = int(dataset.num_samples / batch_size)
num_steps_per_epoch = num_batches_per_epoch
#print(num_batches_per_epoch)

conv1_fmaps = 32
conv1_ksize = 5
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 1
conv2_pad = "SAME"

conv3_fmaps = 128
conv3_ksize = 3
conv3_stride = 2
conv3_pad = "SAME"

n_fc1 = 64
outputs = 10


def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius,
                                              alpha=alpha, beta=beta,
                                              bias=bias, name=name)


he_init = tf.contrib.layers.variance_scaling_initializer()

training = tf.placeholder_with_default(False, None, "training")

conv1 = tf.layers.conv2d(images, filters=conv1_fmaps, kernel_size=conv1_ksize,
                         strides=conv1_stride, padding=conv1_pad, activation=tf.nn.relu, name="conv1")

print(conv1.shape)

norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')

conv2 = tf.layers.conv2d(norm1, filters=conv2_fmaps, kernel_size=conv2_ksize,
                         strides=conv2_stride, padding=conv2_pad, activation=tf.nn.relu, name="conv2")

print(conv2.shape)

norm2 = lrn(conv2, 4, 2e-05, 0.75, name='norm2')

conv3 = tf.layers.conv2d(norm2, filters=conv3_fmaps, kernel_size=conv3_ksize,
                         strides=conv3_stride, padding=conv3_pad, activation=tf.nn.relu, name="conv3")

print(conv3.shape)

norm3 = lrn(conv3, 4, 2e-05, 0.75, name='norm3')

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(norm3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    print(pool3.shape)

    pool_flat = tf.reshape(pool3, shape=[-1, conv3_fmaps * 7 * 7])  # can be guessed by pool.shape

fc1 = tf.layers.dense(pool_flat, n_fc1, activation=tf.nn.relu, kernel_initializer=he_init, name="fc1")

fc1_dropped = tf.layers.dropout(fc1, training=training)

logits = tf.layers.dense(fc1, outputs, kernel_initializer=he_init, name="logits")

Y_proba = tf.nn.softmax(logits, name="Y_proba")





# Create the global step for monitoring the learning_rate and training.
global_step = get_or_create_global_step()

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)

# Define your exponentially decaying learning rate
lr = tf.train.exponential_decay(
    learning_rate=initial_learning_rate,
    global_step=global_step,
    decay_steps=decay_steps,
    decay_rate=learning_rate_decay_factor,
    staircase=True)

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    training_op = optimizer.minimize(loss, global_step=global_step)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, labels, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

# Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
def train_step(sess, train_op, global_step):
    '''
    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
    '''
    # Check the time for each sess run
    #start_time = time.time()
    _, global_step_count, loss_val = sess.run([train_op, global_step, loss])
    #time_elapsed = time.time() - start_time

    # Run the logging to print some results
    #logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, loss_val, time_elapsed)

    return loss_val, global_step_count


# Now finally create all the summaries you need to monitor and group them into one summary op.
tf.summary.scalar('losses', loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('learning_rate', lr)
my_summary_op = tf.summary.merge_all()

init = tf.global_variables_initializer()
config = tf.ConfigProto(
    device_count={'GPU': 0}
)

# Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
sv = tf.train.Supervisor(logdir=log_dir("mnist", False), summary_op=None, init_fn=None)

with sv.managed_session() as session:
    #init.run()

    for step in range(n_epochs * num_batches_per_epoch):

        #_, loss_train = session.run([training_op, loss]) #, feed_dict={training: True, X: X_batch, y: y_batch})
        #loss_train = session.run(loss)
        #acc_test = accuracy.eval(feed_dict={X: mnist.test.images[:2000], y: mnist.test.labels[:2000]})
        #print(epoch, "Train loss:", loss_train) #, "Test accuracy:", acc_test)

        # Log the summaries every 10 step.
        if step % 10 == 0:
            loss_val, global_step_count = train_step(session, training_op, sv.global_step)
            print("Loss:", loss_val, "at", global_step_count)
            summaries = session.run(my_summary_op)
            sv.summary_computed(session, summaries)
        else:
            loss_val, _ = train_step(session, training_op, sv.global_step)

    sv.saver.save(session, sv.save_path)
