import tensorflow as tf
import numpy as np
import os, os.path
from  datetime import datetime
import math
import urllib, sys, tarfile

import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v2

TF_MODELS_URL = "http://download.tensorflow.org/models"
RESNET_V2_URL = TF_MODELS_URL + "/resnet_v2_152_2017_04_14.tar.gz"
RESNET_V2_PATH = os.path.join("datasets", "resnet_v2")
RESNET_V2_CHECKPOINT_PATH = os.path.join(RESNET_V2_PATH, "resnet_v2_152.ckpt")
CLASSES_URL="https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/inception/imagenet_class_names.txt"
CLASSES_PATH=os.path.join(RESNET_V2_PATH, "imagenet_class_names.txt")


##
#
# Following code section is to download the resnet model pre training data if needed
#
#########

def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading: {}%".format(percent))
    sys.stdout.flush()

def fetch_pretrained_resnet_v2(url=RESNET_V2_URL, path=RESNET_V2_PATH):
    if os.path.exists(RESNET_V2_CHECKPOINT_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "resnet_v2.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    resnet_tgz = tarfile.open(tgz_path)
    resnet_tgz.extractall(path=path)
    resnet_tgz.close()
    os.remove(tgz_path)

def fetch_class_list(url=RESNET_V2_URL, path=RESNET_V2_PATH):
    if os.path.exists(CLASSES_PATH):
        return
    os.makedirs(path, exist_ok=True)
    urllib.request.urlretrieve(CLASSES_URL, CLASSES_PATH, reporthook=download_progress)



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

checkpoint_and_log_dir = log_dir("cats_dogs", False)
checkpoint_path = os.path.join(checkpoint_and_log_dir,"catdogs")
checkpoint_file = os.path.join(checkpoint_and_log_dir,"checkpoint")

def build_picture_dataset(folder, randomise=True):
    
    # build a dictionary with the labels and associated filenames
    datapoint = {}
    num_samples = 0 # count total numbers of items for randomisation
    labels = os.listdir(folder)
    for label in labels:
        datapoint[label] = os.listdir(os.path.join(folder,label))
        datapoint[label] = [os.path.join(folder,label, v) for v in datapoint[label]]
        num_samples += len(datapoint[label])

    # create a dict for the labels
    labels_encoding = {}
    labels = []
    for idx, val in enumerate(datapoint.keys()):
        labels_encoding[idx] = val
        labels += [idx] * len(datapoint[val])
    
    # list of all the file names and all the labels
    filenames = tf.constant(sum(datapoint.values(), []))
    labels = tf.constant(labels)
        
    dataset =  tf.data.Dataset.from_tensor_slices((filenames, labels))
    if randomise:
        dataset = dataset.shuffle(num_samples)
    
    def load_image(filename, label):
        image_string = tf.read_file(filename)
        image_decoded = tf.image.decode_jpeg(image_string)
        image_decoded = tf.image.convert_image_dtype(image_decoded, tf.float32)
        return image_decoded, label
    
    dataset = dataset.map(load_image)
    
    return dataset, labels_encoding, num_samples


def preprocessing_images_2(dataset, target_height, target_width, max_zoom=1, flip=False):
    
    def prep_image(image, label):    
        
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
        # between 1.0 and `max_zoom`.
        resize_factor = tf.random_uniform(minval=1, maxval=max_zoom, shape=[1])

        crop_width = tf.cast(crop_width / resize_factor, tf.int32)
        crop_height = tf.cast(crop_height / resize_factor, tf.int32)

        
        box_size = tf.stack([crop_height[0], crop_width[0], tf.constant(3)])   # 3 = number of channels
        # Let's crop the image using a random bounding box of the size we computed
        image = tf.random_crop(image, box_size)
        if flip:
            image = tf.image.random_flip_left_right(image)
            
        # The resize_bilinear function requires a 4D tensor (a batch of images)
        # so we need to expand the number of dimensions first:
        image_batch = tf.expand_dims(image, 0)

        # Finally, let's resize the image to the target dimensions. Note that this function
        # returns a float32 tensor.
        image_batch = tf.image.resize_bilinear(image_batch, [target_height, target_width])
        
        image = image_batch[0] 
    
        return (image,label)
    
    dataset = dataset.map(prep_image)
    
    return dataset


width = 299
height = 299
channels = 3


reset_graph()
fetch_pretrained_resnet_v2()
fetch_class_list()

flags = tf.app.flags

n_epochs = 3
batch_size = 20

training = tf.placeholder_with_default(False, shape=[])

# training dataset
tr_dataset, labels_encoding, tr_num_samples = build_picture_dataset("datasets/dogscats/train", randomise=True)
print(f"Loading dataset with {tr_num_samples} samples")
tr_dataset = preprocessing_images_2(tr_dataset,299,299,max_zoom=1.2,flip=True)
tr_dataset = tr_dataset.repeat()
tr_dataset = tr_dataset.batch(batch_size)

num_batches_per_epoch = int(tr_num_samples / batch_size)
num_steps_per_epoch = num_batches_per_epoch

#validation dataset
va_dataset, labels_encoding, va_num_samples = build_picture_dataset("datasets/dogscats/valid")
va_dataset = preprocessing_images_2(va_dataset,299,299)
va_dataset = va_dataset.batch(batch_size)

iterator = tr_dataset.make_initializable_iterator()
X , y = iterator.get_next()

#X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
with slim.arg_scope(resnet_v2.resnet_arg_scope()):
    logits, end_points = resnet_v2.resnet_v2_152(
    X, num_classes=1001, is_training=training)

resnetsaver = tf.train.Saver()

intermediary = 100
n_outputs = len(labels_encoding)

with tf.name_scope("new_output_layer"):
    dogcats_logits = tf.layers.dense(logits, intermediary, activation=tf.nn.relu, name="dogscats_logits")
    dogcats_logits = tf.layers.dense(dogcats_logits, n_outputs, name="dogscats_logits2")

    dogcats_logits = tf.squeeze(dogcats_logits)

    y_proba = tf.nn.softmax(dogcats_logits)


# Create the global step for monitoring the learning_rate and training.
global_step = tf.train.get_or_create_global_step()
    
    
with tf.name_scope("train"):
    loss =  tf.losses.sparse_softmax_cross_entropy(y, y_proba)# tf.losses.log_loss(y, y_proba)
    optimizer = tf.train.AdamOptimizer()
    dogs_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="dogscats_logits")
    training_op = optimizer.minimize(loss, var_list=dogs_vars, global_step=global_step)
    
with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(dogcats_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))


    
# Now finally create all the summaries you need to monitor and group them into one summary op.
tf.summary.scalar('losses', loss)
tf.summary.scalar('accuracy', accuracy)

my_summary_op = tf.summary.merge_all()

summary_writer = tf.summary.FileWriter(checkpoint_and_log_dir)

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver() 
    
# Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
def train_step(sess, train_op, global_step, with_summary = False):
    '''
    Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
    '''
    summaries = None
    if with_summary:
        _, global_step_count, loss_val, summaries = sess.run([train_op, global_step, loss, my_summary_op])
    else:
        _, global_step_count, loss_val = sess.run([train_op, global_step, loss])

    return loss_val, summaries, global_step_count


with tf.Session() as session:

    if os.path.exists(checkpoint_file):
        print("Restoring from checkpoint")
        saver.restore(session,tf.train.latest_checkpoint(checkpoint_and_log_dir))
    else:
        print("Starting fresh")
        session.run(init)

    print("Global step is now ",tf.train.global_step(session, global_step))
    summary_writer.add_graph(session.graph, tf.train.global_step(session, global_step))

    session.run(iterator.initializer)
    start_iteration = math.floor(tf.train.global_step(session, global_step))
    
    if start_iteration == 0:
        print("loading the pretrained model")
        resnetsaver.restore(session, RESNET_V2_CHECKPOINT_PATH)
    
    print("total number of steps: ",n_epochs * num_batches_per_epoch)
    for step in range(start_iteration, n_epochs * num_batches_per_epoch):

        #_, loss_train = session.run([training_op, loss]) #, feed_dict={training: True, X: X_batch, y: y_batch})
        #loss_train = session.run(loss)
        #acc_test = accuracy.eval(feed_dict={X: mnist.test.images[:2000], y: mnist.test.labels[:2000]})
        #print(epoch, "Train loss:", loss_train) #, "Test accuracy:", acc_test)

        # Log the summaries every 10 step.
        if step % 10 == 0:
            loss_val, summaries, global_step_count = train_step(session, training_op, global_step, with_summary = True)
            print("Loss:", loss_val, "at", global_step_count)
            summary_writer.add_summary(summaries, global_step_count)

        else:
            loss_val, _, _ = train_step(session, training_op, global_step)

        if step % 50 == 0:
            saver.save(session, checkpoint_path, global_step=global_step)

    saver.save(session, checkpoint_path, global_step=global_step)