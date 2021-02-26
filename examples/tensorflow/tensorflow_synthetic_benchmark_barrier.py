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

if args.comm_backend.lower() == "byteps":
    import byteps.tensorflow as bps
    bps.init()
    _local_rank = bps.local_rank()
    _size = bps.size()
    _local_size = bps.local_size()
    _rank = bps.rank()
else:
    import horovod.tensorflow as hvd
    hvd.init()
    _local_rank = hvd.local_rank()
    _size = hvd.size()
    _local_size = hvd.local_size()
    _rank = hvd.rank()

import tensorflow.contrib.slim as slim
def model_summary():
    model_vars = tf.trainable_variables()
    slim.model_analyzer.analyze_vars(model_vars, print_info=True)   

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(_local_rank)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.allow_growth = False
    config.gpu_options.visible_device_list = ''

# Set up standard model.
model = getattr(applications, args.model)(weights=None, classes=args.classes)

lr_scaler = _size
# By default, Adasum doesn't need scaling when increasing batch size. If used with NCCL,
# scale lr by local_size
if args.use_adasum:
    if args.comm_backend.lower() == "byteps":
        lr_scaler = _local_size if args.cuda else 1
    else:
        lr_scaler = _local_size if args.cuda and hvd.nccl_built() else 1

global_step = tf.train.get_or_create_global_step()
opt = tf.train.GradientDescentOptimizer(0.01 * lr_scaler)


if args.amp:
    # auto mixed precision training
    opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)

init = tf.global_variables_initializer()
if args.comm_backend.lower() == "byteps":
    opt = bps.DistributedOptimizer(opt)
    hooks = [bps.TimelineHook(batch_size=args.batch_size), ]
    bcast_op = bps.broadcast_global_variables(0)
else:
    # Horovod: (optional) compression algorithm.
    compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none
    # Horovod: wrap optimizer with DistributedOptimizer.
    opt = hvd.DistributedOptimizer(opt, compression=compression,
        op=hvd.Adasum if args.use_adasum else hvd.Average,
        gradient_predivide_factor=args.gradient_predivide_factor)
    hooks = [hvd.TimelineHook(batch_size=args.batch_size), ]
    bcast_op = hvd.broadcast_global_variables(0)

# data = tf.random_uniform([args.batch_size, 224, 224, 3])
# target = tf.random_uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)

with tf.name_scope("input_barrier"):
    min_val_tensor = tf.random_uniform([], maxval=0.1,name="barrier_tensor")
    if args.comm_backend.lower() == "byteps":
        min_val_tensor = bps.push_pull(min_val_tensor)
    else:
        min_val_tensor = hvd._allreduce(min_val_tensor)
    data = tf.random_uniform([args.batch_size, 224, 224, 3], minval=min_val_tensor)
target = tf.random_uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)

probs = model(data, training=True)
loss = tf.losses.sparse_softmax_cross_entropy(target, probs)
train_opt = opt.minimize(loss, global_step=global_step)

def log(s, nl=True):
    if _rank != 0:
        return
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
    bcast_op.run(session=mon_sess)
    run(lambda: mon_sess.run(train_opt))

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
