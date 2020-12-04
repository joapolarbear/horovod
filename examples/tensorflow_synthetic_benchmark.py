from __future__ import absolute_import, division, print_function

import argparse
import os
import numpy as np
import timeit
import time

import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.keras import applications

# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

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

args = parser.parse_args()
args.cuda = not args.no_cuda

hvd.init()

from google.protobuf.json_format import MessageToJson
from tensorflow.python.client import timeline
import json
import networkx as nx
class TimelineSession:
    def __init__(self, sess):
        self.sess = sess
        self.graph = sess.graph
        self.step_cnt = 0

        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(hvd.local_rank()))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)
        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            return
        self._end_trace = False
        self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))
        self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))

        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")   

        ### Timeline configuratoin
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.traces = {"traceEvents":[]}

        self.dag = None

    def run(self, *args_, **kwargs_):
        if self._end_trace:
            ret = self.sess.run(*args_, **kwargs_)
        elif not self._end_trace and self.step_cnt < self.start_step:
            ret = self.sess.run(*args_, **kwargs_)
            self.step_cnt += 1
        elif not self._end_trace and self.step_cnt < self.end_step:
            ret = self.sess.run(*args_, options=self.run_options, run_metadata=self.run_metadata, **kwargs_)
            # Create the Timeline object, and write it to a json
            tl = timeline.Timeline(self.run_metadata.step_stats)
            ctf = json.loads(tl.generate_chrome_trace_format())
            self.traces["traceEvents"] += ctf["traceEvents"]
            print("Add the {}th step of traces".format(self.step_cnt))
            self.step_cnt += 1

            ### Create the DAG
            if self.dag is None:
                self.dag = nx.DiGraph()
                for trace in ctf["traceEvents"]:
                    if trace["ph"] == "M" or "args" not in trace:
                        continue
                    op = trace["args"]["op"]
                    name = trace["args"]["name"]

                    ### Add nodes to the DAG
                    if name not in self.dag.nodes:
                        self.dag.add_node(name)

                    ### Add dependency info
                    for k, v in trace["args"].items():
                        if "input" in k:
                            self.dag.add_edge(v, name)

            try:
                not_found = False
                nx.find_cycle(self.dag.cycle)
            except:
                not_found = True
            assert not_found


            ### Output traces
            if self.step_cnt == self.end_step:
                self._end_trace = True
                self.output_traces()

        ### Return all fetches
        return ret

    
    def output_traces(self):
        with open(os.path.join(self.trace_dir, "temp.json"), "w") as f:
            json.dump(self.traces, f, indent=4)

        ### collect graph info
        graphdef = tf.get_default_graph().as_graph_def()
        graph_str = json.loads(MessageToJson(graphdef))
        with open(os.path.join(self.trace_dir, "graph.json"), "w") as f:
            json.dump(graph_str, f, indent=4)

        nx.write_gml(self.dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
        print("Stop tracing, output trace: %s" % self.trace_dir)

    def should_stop(self):
        return self.sess.should_stop()

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

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
if args.cuda:
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    config.gpu_options.allow_growth = False
    config.gpu_options.visible_device_list = ''

# Set up standard model.
model = getattr(applications, args.model)(weights=None)

lr_scaler = hvd.size()
# By default, Adasum doesn't need scaling when increasing batch size. If used with NCCL,
# scale lr by local_size
if args.use_adasum:
    lr_scaler = hvd.local_size() if args.cuda and hvd.nccl_built() else 1

global_step = tf.train.get_or_create_global_step()
opt = tf.train.GradientDescentOptimizer(0.01 * lr_scaler)

# Horovod: (optional) compression algorithm.
compression = hvd.Compression.fp16 if args.fp16_allreduce else hvd.Compression.none

# Horovod: wrap optimizer with DistributedOptimizer.
# auto mixed precision training
# opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
opt = hvd.DistributedOptimizer(opt, compression=compression, op=hvd.Adasum if args.use_adasum else hvd.Average)

init = tf.global_variables_initializer()
bcast_op = hvd.broadcast_global_variables(0)

data = tf.random_uniform([args.batch_size, 224, 224, 3])
target = tf.random_uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)

probs = model(data, training=True)
loss = tf.losses.sparse_softmax_cross_entropy(target, probs)
train_opt = opt.minimize(loss, global_step=global_step)


def log(s, nl=True):
    if hvd.rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, hvd.size()))


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
        (hvd.size(), device, hvd.size() * img_sec_mean, hvd.size() * img_sec_conf))

hooks = [hvd.TimelineHook(),]
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