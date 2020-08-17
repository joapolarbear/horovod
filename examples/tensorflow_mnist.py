# Copyright 2019 Uber Technologies, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import os
import errno
import tensorflow as tf
import horovod.tensorflow as hvd
import numpy as np
import argparse

from tensorflow import keras

layers = tf.layers


# Training settings
parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='output summary for tensorboard visualization')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use automatically mixed precision')
parser.add_argument('--precision_loss', action='store_true', default=False,
                    help='analyze the precision loss (only run for one iteration)')
parser.add_argument('--batch_size', type=int, default="1000",
                    help='batch size')
parser.add_argument('--dense_size', type=int, default="1024",
                    help='the output size of the first dense layer')
parser.add_argument('--kernel_size', type=int, default="5",
                    help='kernel size')
args = parser.parse_args()


import struct, math
def float_to_bits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]

LARGEST_NORMAL_FP16 = 65504              # 2^15 * (1 + 1023/1024)
SMALLEST_NORMAL_FP16 = 0.000060976       # 2^-14
SMALLEST_SUBNOMRAL_FP16 = 0.000000059605 # 2^-24
def precision_loss(fp32):
    assert "float" in str(type(fp32)), "Type error: %s"%(str(type(fp32)))
    if fp32 == 0:
        return 0.0
    elif fp32 < 0:
        fp32 = - fp32
    fp32_bits = float_to_bits(fp32)
    sign = (fp32_bits >> 31) & 0x1
    expo = ((fp32_bits >> 23) & 0xff) - 0x7f
    prec = (fp32_bits & 0x7fffff) / (1 << 23)
    # print(hex(fp32_bits), sign, expo, prec)
    if fp32 > LARGEST_NORMAL_FP16:
        # print("Overflow")
        ret = (fp32 - LARGEST_NORMAL_FP16) / fp32
    elif fp32 < SMALLEST_SUBNOMRAL_FP16:
        # print("Underflow")
        ret = 1.0
    elif fp32 < SMALLEST_NORMAL_FP16:
        # print("Subnormal")
        ###  additional precision loss: (-14) - (exp_fp_32 - 127)
        addition_bit = -14 - expo - 1
        ret = ((((1 << (14 + addition_bit)) - 1) & (fp32_bits & 0x7fffff)) * math.pow(2, expo - 23)) / fp32
    else:
        # print("Normal")
        ret = ((fp32_bits & 0x1fff) * math.pow(2, expo - 23)) / fp32

    if ret > 1:
        if fp32 > LARGEST_NORMAL_FP16:
            print("Overflow")
        elif fp32 < SMALLEST_SUBNOMRAL_FP16:
            print("Underflow")
        elif fp32 < SMALLEST_NORMAL_FP16:
            print("Subnormal")
        else:
            print("Normal")
            print(fp32)
            print(fp32_bits)
            raise

    # fp16 = np.float16(fp32)
    # diff = np.abs(fp32 - fp16) / fp32
    # if ret != diff:
    #     print(fp32, hex(fp32_bits), ret, diff)

    return ret

def precision_loss_stat(fp32, stat):
    assert "float" in str(type(fp32)), "Type error: %s"%(str(type(fp32)))
    if fp32 == 0:
        return 0.0
    elif fp32 < 0:
        fp32 = - fp32
    # print(hex(fp32_bits), sign, expo, prec)
    if fp32 > LARGEST_NORMAL_FP16:
        stat["overflow"] += 1
    elif fp32 < SMALLEST_SUBNOMRAL_FP16:
        stat["underflow"] += 1
    elif fp32 < SMALLEST_NORMAL_FP16:
        stat["subnormal"] += 1
    else:
        stat["normal"] += 1

def precision_loss_np(fp32):
    assert isinstance(fp32, np.ndarray)
    assert fp32.dtype is np.dtype('float32')
    fp32 = np.abs(fp32)
    fp16 = np.float16(fp32)
    pl = np.abs(fp32 - fp16) / fp32
    pl[np.isnan(pl)] = 0
    return np.average(pl)

def half_precision(layer_f, input_, *args_, **kwargs_):
    if args.amp:
        input_fp16 = tf.dtypes.cast(input_, tf.float16)
        output_fp16 = layer_f(input_fp16, *args_, **kwargs_)
        if args.tensorboard:
            tf.summary.histogram("weights--%s"%(output_fp16.name), output_fp16)
        output_fp32 = tf.dtypes.cast(output_fp16, tf.float32)
    else:
        output_fp32 = layer_f(input_, *args_, **kwargs_)
        if args.tensorboard:
            tf.summary.histogram("weights--%s"%(output_fp32.name), output_fp32)
    return output_fp32

def single_precision(layer_f, input_, *args_, **kwargs_):
    output_fp32 = layer_f(input_, *args_, **kwargs_)
    if args.tensorboard:
        tf.summary.histogram("weights--%s"%(output_fp32.name), output_fp32)
    return output_fp32

import tensorflow.contrib.slim as slim
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)



def conv_model(feature, target, mode):
    """2-layer convolution model."""
    # Convert the target to a one-hot tensor of shape (batch_size, 10) and
    # with a on-value of 1 for each one-hot vector of length 10.
    target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)
    # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
    # image width and height final dimension being the number of color channels.
    feature = tf.reshape(feature, [-1, 28, 28, 1])
    # First conv layer will compute 32 features for each 5x5 patch
    with tf.variable_scope('conv_layer1'):
        # h_conv1 = layers.conv2d(feature_16, 32, kernel_size=[5, 5],
        #                         activation=tf.nn.relu, padding="SAME")
        h_conv1 = single_precision(layers.conv2d, feature, 32, kernel_size=[args.kernel_size, args.kernel_size],
                                activation=tf.nn.relu, padding="SAME")
        h_pool1 = tf.nn.max_pool(
            h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    # Second conv layer will compute 64 features for each 5x5 patch.
    with tf.variable_scope('conv_layer2'):
        h_conv2 = single_precision(layers.conv2d, h_pool1, 64, kernel_size=[args.kernel_size, args.kernel_size],
                                activation=tf.nn.relu, padding="SAME")
        h_pool2 = tf.nn.max_pool(
            h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # reshape tensor into a batch of vectors
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    # Densely connected layer with 1024 neurons.
    h_fc1 = layers.dropout(
        single_precision(layers.dense, h_pool2_flat, args.dense_size, activation=tf.nn.relu),
        rate=0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    # Compute logits (1 per class) and compute loss.
    logits = single_precision(layers.dense, h_fc1, 10, activation=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)
    tf.summary.scalar("loss", loss)
    return tf.argmax(logits, 1), loss

def train_input_generator(x_train, y_train, batch_size=64):
    assert len(x_train) == len(y_train)
    while True:
        p = np.random.permutation(len(x_train))
        x_train, y_train = x_train[p], y_train[p]
        index = 0
        while index <= len(x_train) - batch_size:
            yield x_train[index:index + batch_size], \
                  y_train[index:index + batch_size],
            index += batch_size

def main(_):
    # Horovod: initialize Horovod.
    hvd.init()
    tf.get_logger().setLevel('WARN')

    # Keras automatically creates a cache directory in ~/.keras/datasets for
    # storing the downloaded MNIST data. This creates a race
    # condition among the workers that share the same filesystem. If the
    # directory already exists by the time this worker gets around to creating
    # it, ignore the resulting exception and continue.
    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if not os.path.exists(cache_dir):
        try:
            os.mkdir(cache_dir)
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(cache_dir):
                pass
            else:
                raise

    # Download and load MNIST dataset.
    (x_train, y_train), (x_test, y_test) = \
        keras.datasets.mnist.load_data('MNIST-data-%d' % hvd.rank())

    # The shape of downloaded data is (-1, 28, 28), hence we need to reshape it
    # into (-1, 784) to feed into our network. Also, need to normalize the
    # features between 0 and 1.
    x_train = np.reshape(x_train, (-1, 784)) / 255.0
    x_test = np.reshape(x_test, (-1, 784)) / 255.0

    # Build model...
    with tf.name_scope('input'):
        image = tf.placeholder(tf.float32, [None, 784], name='image')
        label = tf.placeholder(tf.float32, [None], name='label')
    predict, loss = conv_model(image, label, tf.estimator.ModeKeys.TRAIN)

    model_summary()

    lr_scaler = hvd.size()
    # By default, Adasum doesn't need scaling when increasing batch size. If used with NCCL,
    # scale lr by local_size
    if args.use_adasum:
        lr_scaler = hvd.local_size() if hvd.nccl_built() else 1

    # Horovod: adjust learning rate based on lr_scaler.
    opt = tf.train.AdamOptimizer(0.001 * lr_scaler, epsilon=(1e-4 if args.amp else 1e-8))

    # auto mixed precision training
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt, op=hvd.Adasum if args.use_adasum else hvd.Average)
    global_step = tf.train.get_or_create_global_step()

    # train_op = opt.minimize(loss, global_step=global_step)
    gradients = opt.compute_gradients(loss)
    train_op = opt.apply_gradients(gradients, global_step=global_step)
    
    if args.tensorboard:
        grad_summ_op = tf.summary.merge([tf.summary.histogram("gradients--%s"%g[1].name, g[0]) for g in gradients])
        summary_op = tf.summary.merge_all()

    hooks = [
        # Horovod: BroadcastGlobalVariablesHook broadcasts initial variable states
        # from rank 0 to all other processes. This is necessary to ensure consistent
        # initialization of all workers when training is started with random weights
        # or restored from a checkpoint.
        hvd.BroadcastGlobalVariablesHook(0),

        # Horovod: adjust number of steps based on number of GPUs.
        tf.train.StopAtStepHook(last_step=205 // hvd.size()),

        tf.train.LoggingTensorHook(tensors={'step': global_step, 'loss': loss},
                                   every_n_iter=10),

        hvd.TimelineHook(),
    ]


    # Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())

    # Horovod: save checkpoints only on worker 0 to prevent other workers from
    # corrupting them.
    checkpoint_dir = './checkpoints' if hvd.rank() == 0 else None
    training_batch_generator = train_input_generator(x_train,
                                                     y_train, batch_size=args.batch_size)
    # The MonitoredTrainingSession takes care of session initialization,
    # restoring from a checkpoint, saving to a checkpoint, and closing when done
    # or an error occurs.
    with tf.train.MonitoredTrainingSession(
                                        # checkpoint_dir=checkpoint_dir,
                                           hooks=hooks,
                                           config=config) as mon_sess:
        if args.tensorboard:
            summary_writer = tf.summary.FileWriter(os.path.join(mon_sess.trace_dir, "board"), mon_sess.graph) 
        while not mon_sess.should_stop():
            # Run a training step synchronously.
            image_, label_ = next(training_batch_generator)
            
            if args.tensorboard:
                _, summary_str, global_step_, gradients_ = mon_sess.run([train_op, summary_op, global_step, gradients], feed_dict={image: image_, label: label_})
                summary_writer.add_summary(summary_str, global_step=global_step_)
                summary_writer.flush()
            else:
                _, global_step_, gradients_ = mon_sess.run([train_op, global_step, gradients], feed_dict={image: image_, label: label_})

            if args.precision_loss:
                for i in range(len(gradients_)):
                    grad = gradients_[i][0].flatten()

                    import time

                    t = time.time()
                    pl_sum = pl_cnt = 0
                    for g in grad:
                        pl_sum += precision_loss(g)
                        pl_cnt += 1
                    print("Theoretical: Gradient: {}, precision loss: {}, time: {}".format(gradients[i][1].name, pl_sum/pl_cnt, time.time() - t))

                    # stat = {"overflow": 0, "underflow": 0, "subnormal":0, "normal":0}
                    # for g in grad:
                    #     precision_loss_stat(g, stat)
                    # sum_ = sum(stat.values())
                    # print("Gradient: {} -- {}".format(gradients[i][1].name, [k + " " + str(v / sum_) for k, v in stat.items()]))

                    t = time.time()
                    pl = precision_loss_np(grad)
                    print("Minus: Gradient: {}, precision loss: {}, time: {}".format(gradients[i][1].name, pl, time.time() - t))
                break


if __name__ == "__main__":
    tf.app.run()





