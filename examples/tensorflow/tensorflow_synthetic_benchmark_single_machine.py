import argparse
import os
import numpy as np
import timeit
import time

import tensorflow as tf
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

parser.add_argument('--amp', action='store_true', default=False,
                    help='Use amp')
parser.add_argument('--comm_backend', type=str, default='hvd',
                    help='Communication backend')
parser.add_argument('--classes', type=int, default=1000,
                    help='number of batches per benchmark iteration')

args = parser.parse_args()
args.cuda = not args.no_cuda

import tensorflow.contrib.slim as slim
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)   

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(0)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.allow_growth = False
    config.gpu_options.visible_device_list = ''

# Set up standard model.
model = getattr(applications, args.model)(weights=None, classes=args.classes)

_size = 1
lr_scaler = _size

global_step = tf.train.get_or_create_global_step()
opt = tf.train.GradientDescentOptimizer(0.01 * lr_scaler)

if args.amp:
    # auto mixed precision training
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

init = tf.global_variables_initializer()
hooks = []
data = tf.random_uniform([args.batch_size, 224, 224, 3])
target = tf.random_uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)

probs = model(data, training=True)
loss = tf.losses.sparse_softmax_cross_entropy(target, probs)
train_opt = opt.minimize(loss, global_step=global_step)
if os.environ.get("BPF_TEST_MEMORY", "") == "1":
    memory_summary = tf.contrib.memory_stats.MaxBytesInUse()

def log(s, nl=True):
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, _size))

def run(benchmark_step):
    # Warm-up
    log('Running warmup...')
    timeit.timeit(benchmark_step, number=args.num_warmup_batches)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        time_s = time.time()
        dur = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / dur
        iter_time = (time.time() - time_s) / args.num_batches_per_iter
        log('Iter #%d: %.1f img/sec per %s, iteration time %f ms' % (x, img_sec, device, iter_time * 1000))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
    log('Total img/sec on %d %s(s): %.1f +-%.1f' %
        (_size, device, _size * img_sec_mean, _size * img_sec_conf))

with tf.train.MonitoredTrainingSession(hooks=hooks, config=config) as mon_sess:
    run(lambda: mon_sess.run(train_opt))
    if os.environ.get("BPF_TEST_MEMORY", "") == "1":
        print("Rank %d: Peak memory: %.2f MB" % (_rank, mon_sess.run(memory_summary) / (1024**2)))

# with tf.train.MonitoredTrainingSession(hooks=hooks, config=config) as mon_sess:
#     init.run()
#     bcast_op.run()

#     loss = loss_function()
#     train_opt = opt.minimize(loss)
#     # Warm-up
#     log('Running warmup...')
#     for _ in range(args.num_warmup_batches):
#         mon_sess.run(train_opt)

#     # Benchmark
#     log('Running benchmark...')
#     img_secs = []
#     for x in range(args.num_iters):
#         time_s = time.time()
#         for _ in range(args.num_batches_per_iter):
#             mon_sess.run(train_opt)
#         img_sec = args.batch_size * args.num_batches_per_iter / (time.time() - time_s)
#         log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
#         img_secs.append(img_sec)

#     # Results
#     img_sec_mean = np.mean(img_secs)
#     img_sec_conf = 1.96 * np.std(img_secs)
#     log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
#     log('Total img/sec on %d %s(s): %.1f +-%.1f' %
#         (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))
