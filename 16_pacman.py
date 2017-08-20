# To support both python 2 and python 3
from __future__ import division, print_function, unicode_literals

# Common imports
import numpy as np
import numpy.random as rnd
import gym
import tensorflow as tf
from datetime import datetime
import math

from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step

env = gym.make("MsPacman-v0")

mspacman_color = np.array([210, 164, 74]).mean()

def log_dir(prefix="", date=True):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run" + "" if not date else "-" + now
    return "{}/{}/".format(root_logdir, name)

def preprocess_observation(obs):
    img = obs[1:176:2, ::2] # crop and downsize
    img = img.mean(axis=2) # to greyscale
    img[img==mspacman_color] = 0 # improve contrast
    img = (img - 128) / 128 - 1 # normalize from -1. to 1.
    return img.reshape(88, 80, 1)

from gym.envs.classic_control import rendering
def repeat_upsample(rgb_array, k=1, l=1, err=[]):
    # repeat kinda crashes if k/l are zero
    if k <= 0 or l <= 0:
        if not err:
            #print "Number of repeats must be larger than 0, k: {}, l: {}, returning default array!".format(k, l)
            err.append('logged')
        return rgb_array

    # repeat the pixels k times along the y axis and l times along the x axis
    # if the input image is of shape (m,n,3), the output image will be of shape (k*m, l*n, 3)

    return np.repeat(np.repeat(rgb_array, k, axis=0), l, axis=1)

viewer = rendering.SimpleImageViewer()
def render(env):
    rgb = env.render('rgb_array')
    upscaled=repeat_upsample(rgb,4, 4)
    viewer.imshow(upscaled)


# replay memory

from collections import deque

replay_memory_size = 10000
replay_memory = deque([], maxlen=replay_memory_size)

def sample_memories(batch_size):
    indices = rnd.permutation(len(replay_memory))[:batch_size]
    cols = [[], [], [], [], []] # state, action, reward, next_state, continue
    for idx in indices:
        memory = replay_memory[idx]
        for col, value in zip(cols, memory):
            col.append(value)
    cols = [np.array(col) for col in cols]
    return (cols[0], cols[1], cols[2].reshape(-1, 1), cols[3], cols[4].reshape(-1, 1))




input_height = 88
input_width = 80
input_channels = 1
conv_n_maps = [32, 64, 64]
conv_kernel_sizes = [(8,8), (4,4), (3,3)]
conv_strides = [4, 2, 1]
conv_paddings = ["SAME"] * 3
conv_activation = [tf.nn.relu] * 3
n_hidden_in = 64 * 11 * 10 # conv3 has 64 maps of 11x10 each
n_hidden = 512
hidden_activation = tf.nn.relu
n_outputs = env.action_space.n # 9 discrete actions are available
initializer = tf.contrib.layers.variance_scaling_initializer()


def q_network(X_state, name):
    prev_layer = X_state
    conv_layers = []
    with tf.variable_scope(name) as scope:
        for n_maps, kernel_size, stride, padding, activation in zip(conv_n_maps, conv_kernel_sizes, conv_strides,
                                                                    conv_paddings, conv_activation):
            prev_layer = tf.layers.conv2d(prev_layer, filters=n_maps, kernel_size=kernel_size,
                                          strides=stride, padding=padding, activation=activation,
                                          kernel_initializer=initializer)
            conv_layers.append(prev_layer)
        last_conv_layer_flat = tf.reshape(prev_layer, shape=[-1, n_hidden_in])
        hidden = tf.layers.dense(last_conv_layer_flat, n_hidden,
                                 activation=hidden_activation,
                                 kernel_initializer=initializer)
        outputs = tf.layers.dense(hidden, n_outputs,
                                  kernel_initializer=initializer)
    trainable_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                       scope=scope.name)
    trainable_vars_by_name = {var.name[len(scope.name):]: var
                              for var in trainable_vars}
    return outputs, trainable_vars_by_name

X_state = tf.placeholder(tf.float32, shape=[None, input_height, input_width, input_channels])

actor_q_values, actor_vars = q_network(X_state, name="q_networks/actor")
critic_q_values, critic_vars = q_network(X_state, name="q_networks/critic")

copy_ops = [actor_var.assign(critic_vars[var_name])
            for var_name, actor_var in actor_vars.items()]
copy_critic_to_actor = tf.group(*copy_ops)

global_step = get_or_create_global_step()

# epsilon greedy policy

eps_min = tf.constant(0.05, tf.float32)
eps_max = tf.constant(1.0, tf.float32)
eps_decay_steps = tf.constant(50000, tf.float32)

epsilon = tf.maximum(eps_min, tf.subtract(eps_max, tf.divide(tf.multiply(tf.subtract(eps_max, eps_min), tf.cast(global_step, tf.float32)), eps_decay_steps)))

#max_reward = tf.placeholder(tf.float32)
#min_reward = tf.placeholder(tf.float32)

epsilon_summary = tf.summary.scalar("epsilon", epsilon)
#max_reward_summary = tf.summary.scalar("max_reward", max_reward)
#min_reward_summary = tf.summary.scalar("min_reward", min_reward)
my_summary_op = tf.summary.merge_all()


learning_rate = 0.01

X_action = tf.placeholder(tf.int32, shape=[None])
q_value = tf.reduce_sum(critic_q_values * tf.one_hot(X_action, n_outputs), axis=1, keep_dims=True)

y = tf.placeholder(tf.float32, shape=[None, 1])
cost = tf.reduce_mean(tf.square(y - q_value))
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(cost, global_step=global_step)
init = tf.global_variables_initializer()
saver = tf.train.Saver()

n_steps = 100000 # total number of training steps
training_start = 1000 # start training after 1,000 game iterations
training_interval = 3 # run a training step every 3 game iterations
save_steps = 50 # save the model every 50 training steps
copy_steps = 25 # copy the critic to the actor every 25 training steps
discount_rate = 0.95
skip_start = 90 # skip the start of every game (it's just waiting time)
batch_size = 50
iteration = 0 # game iterations
checkpoint_path = "./my_dqn.ckpt"
done = True # env needs to be reset

render_flag = False

config = tf.ConfigProto(
        device_count = {'GPU': 0}
    )

sv = tf.train.Supervisor(logdir=log_dir("pacman", False), summary_op=None, init_fn=None)

with sv.managed_session() as sess:   #config=config

#    start_iteration = math.floor(tf.train.global_step(sess, global_step))

#    for iteration in range(start_iteration, n_iterations):

    while True:
        step = tf.train.global_step(sess, global_step)
        if step % 10 ==0:
            print("epoch ", step)
        if step >= n_steps:
            break

        iteration += 1
        if done:  # game over, start again
            obs = env.reset()
            if render_flag:
                render(env)
            for skip in range(skip_start):  # skip the start of each game
                obs, reward, done, info = env.step(0)
                if render_flag:
                    render(env)
            state = preprocess_observation(obs)

        # Actor evaluates what to do
        q_values = sess.run(actor_q_values, feed_dict={X_state: [state]})

        #action = epsilon_greedy(q_values, step)

        epsilon_val = sess.run(epsilon)

        if rnd.rand() < epsilon_val:
            action =  rnd.randint(n_outputs)  # random action
        else:
            action =  np.argmax(q_values)  # optimal action

        # Actor plays
        obs, reward, done, info = env.step(action)
        if render_flag:
            render(env)
        next_state = preprocess_observation(obs)

        # Let's memorize what just happened
        replay_memory.append((state, action, reward, next_state, 1.0 - done))
        state = next_state

        if iteration < training_start or iteration % training_interval != 0:
            continue

        # Critic learns
        X_state_val, X_action_val, rewards, X_next_state_val, continues = (
            sample_memories(batch_size))
        next_q_values = sess.run( actor_q_values,
            feed_dict={X_state: X_next_state_val})

        max_next_q_values = np.max(next_q_values, axis=1, keepdims=True)
        y_val = rewards + continues * discount_rate * max_next_q_values
        sess.run(training_op, feed_dict={X_state: X_state_val,
                                   X_action: X_action_val, y: y_val})

        # Regularly copy critic to actor
        if step % copy_steps == 0:
            sess.run(copy_critic_to_actor)

        summaries = sess.run(my_summary_op) #, feed_dict={avg_reward: np.mean(rewards_outcome), max_reward: np.max(rewards_outcome), min_reward: np.min(rewards_outcome)})
        sv.summary_computed(sess, summaries)


        sv.saver.save(sess, sv.save_path)