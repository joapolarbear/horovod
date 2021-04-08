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
import argparse
import os
import numpy as np
import timeit

import tensorflow as tf
# import horovod.tensorflow as hvd
from tensorflow.keras import applications

try:
    tf.train.SecondOrStepTimer
    import tensorflow as _tf
except AttributeError:
    import tensorflow.compat.v1 as _tf

class _SecondOrStepTimer(_tf.train.SecondOrStepTimer):
    def __init__(self, every_secs=None, every_steps=None, step_bound=None):
        if step_bound is not None:
            if not (isinstance(step_bound, list) or isinstance(step_bound, tuple)):
                raise ValueError("step bound must be a list or a tuple, but {} is given".format(step_bound))
            self._start_step = step_bound[0]
            self._end_step = step_bound[1]
            if self._start_step > self._end_step:
                raise ValueError("Profiling start step must be smaller than the end step.")
        else:
            self._start_step = self._end_step = None

        super(_SecondOrStepTimer, self).__init__(every_secs, every_steps)

    def should_trigger_for_step(self, step):
        if self._start_step is not None:
            if step < self._start_step or step >= self._end_step:
                return False

        return super(_SecondOrStepTimer, self).should_trigger_for_step(step)

class TimelineHook(_tf.train.ProfilerHook):
    def __init__(self, _summary=False, batch_size=None):
        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(0))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)

        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            self.start_step = self.end_step = 0
        else:
            self._end_trace = False
            self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))
            self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))
        
        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")
        
        print("TimelineHook enable: {}  start_step: {} end_step: {}".format(not self._end_trace, self.start_step, self.end_step))
        self.dag = None
        self.has_data = False

        self.shape_dict = {}
        self.run_metadata = None
        self.partition_dag = None
        self.step_stats = []

        self._output_file = os.path.join(self.trace_dir, "timeline-{}.json")
        self._file_writer = _tf.summary.FileWriterCache.get(self.trace_dir) if _summary else None
        self._show_dataflow = True
        self._show_memory = False
        self._timer = _SecondOrStepTimer(
            every_secs=None, every_steps=1, step_bound=(self.start_step, self.end_step))
        self.batch_size = batch_size
        assert self.batch_size is not None

    def before_run(self, run_context):
        t = time.time()
        if not self._end_trace:
            self._request_summary = (
                self._next_step is not None and
                self._timer.should_trigger_for_step(self._next_step))
            
            if self._request_summary and not self.has_data:
                ### the first step to collect traces, self.has_data tells there are data that need outputing
                self.has_data = True
            if self.has_data and not self._request_summary:
                ### the step after the last trace step, output data
                self._end_trace = True
                partition_graphs = []
                for idx in range(len(self.run_metadata.partition_graphs)):
                    graph_def = self.run_metadata.partition_graphs[idx]
                    partition_graphs.append(graph_def)
                _t = threading.Thread(target=self.output_traces, args=(tf.get_default_graph().get_operations(), partition_graphs))
                _t.start()
        else:
            self._request_summary = False
                
        requests = {"global_step": self._global_step_tensor}
        opts = (_tf.RunOptions(trace_level=_tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
            if self._request_summary else None)

        t = time.time() - t
        # print("Before run takes: {} seconds".format(t))
        return _tf.train.SessionRunArgs(requests, options=opts)

    def after_run(self, run_context, run_values):
        t = time.time()
        stale_global_step = run_values.results["global_step"]
        if self._next_step is None:
        # Update the timer so that it does not activate until N steps or seconds
        # have passed.
            self._timer.update_last_triggered_step(stale_global_step)
        global_step = stale_global_step + 1
        if self._request_summary:
            self.run_metadata = run_values.run_metadata
            self._timer.update_last_triggered_step(global_step)
            # _t = multiprocessing.Process(target=self._save, args=(global_step, self._output_file.format(global_step),
            #          run_values.run_metadata.step_stats))
            # _t.start()
            self.step_stats.append(run_values.run_metadata.step_stats)
            # self._save(global_step, self._output_file.format(global_step),
            #         run_values.run_metadata.step_stats)
            # get shapes from step_stats
            if True:
                if not self.shape_dict:
                    for dev_stats in run_values.run_metadata.step_stats.dev_stats:
                        for node_stats in dev_stats.node_stats:
                            for node_outputs in node_stats.output:
                                slot = node_outputs.slot
                                dtype = node_outputs.tensor_description.dtype
                                shape = []
                                if node_outputs.tensor_description.shape.unknown_rank:
                                    shape.append("Unknown")
                                else:
                                    for shape_in_dim in node_outputs.tensor_description.shape.dim:
                                        shape.append(shape_in_dim.size)
                                if node_stats.node_name+":{}".format(slot) not in self.shape_dict:
                                    self.shape_dict[node_stats.node_name+":{}".format(slot)] = {}
                                self.shape_dict[node_stats.node_name+":{}".format(slot)]["shape"] = shape
                                self.shape_dict[node_stats.node_name+":{}".format(slot)]["dtype"] = dtype
            if self._file_writer is not None:
                self._file_writer.add_run_metadata(run_values.run_metadata,
                                         "step_%d" % global_step)
        self._next_step = global_step + 1
        t = time.time() - t
        # print("After run takes: {} seconds".format(t))

    def output_traces(self, ops, partition_graphs):
        self.traces = {"traceEvents":[]}
        ### the ProfilerHook of tensorflow will output the timeline to self.trace_dir/timeline-{global_step}.json
        # for file in sorted(os.listdir(self.trace_dir)):
        #     if file.startswith('timeline-'):
        #         with open(os.path.join(self.trace_dir, file), 'r') as fp:
        #             ctf = json.load(fp)
        #         convert_traces = self.chome_trace_MBE2X(ctf["traceEvents"])
        #         self.traces["traceEvents"] += convert_traces 

        for step_stats in self.step_stats:
            trace = timeline.Timeline(step_stats)
            events_str = trace.generate_chrome_trace_format(
                    show_dataflow=self._show_dataflow, show_memory=self._show_memory)
            events = json.loads(events_str)
            self.traces["traceEvents"] += self.chome_trace_MBE2X(events["traceEvents"])
        
        with open(os.path.join(self.trace_dir, "temp.json"), "w") as fp:
            json.dump(self.traces, fp, indent=4)

        if os.getenv("BYTEPS_PURE_TF_TRACE", '1') == '1':
            ### delete all intermediate redults
            _output_files = os.path.join(self.trace_dir, "timeline-*.json")
            os.system('rm {}'.format(_output_files))

        def serialize_tensor(t):
            _shape = t.shape.as_list() if t.shape.dims is not None else []
            if len(_shape) > 0 and _shape[0] is None:
                _shape[0] = self.batch_size
            return {
                "name": t.name,
                "shape": _shape,
                "dtype": t.dtype.name
            }

        for idx, graph_def in enumerate(partition_graphs):
            graph_json = json.loads(MessageToJson(graph_def))
            with open(os.path.join(self.trace_dir, "partition_def_{}.json".format(idx)), "w") as f:
                json.dump(graph_json, f, indent=4)
            
            if idx == 0:
                # generate dag
                self.partition_dag = nx.DiGraph()
                # clean node names in graph def
                pruned_node = set()
                all_node_names = set([node["name"] if node["name"][0] != "_" else node["name"][1:] \
                                                                    for node in graph_json["node"]])
                for node in graph_json["node"]:
                    if node["name"][0] == "_":
                        node["name"] = node["name"][1:]
                    last_slash_pos = node["name"].rfind("/")
                    if last_slash_pos != -1 and last_slash_pos < len(node["name"])-1 \
                                            and node["name"][last_slash_pos+1] == "_":
                        if node["name"][:last_slash_pos] in all_node_names:
                            pruned_node.add(node["name"])
                            continue
                        else:
                            node["name"] = node["name"][:last_slash_pos]
                    if "input" in node:
                        for idx, input_node in enumerate(node["input"]):
                            if input_node[0] == "_":
                                node["input"][idx] = input_node[1:]
                                input_node = input_node[1:]
                            last_slash_pos = input_node.rfind("/")
                            if last_slash_pos != -1 and last_slash_pos < len(input_node)-1 \
                                                    and input_node[last_slash_pos+1] == "_":
                                node["input"][idx] = input_node[:last_slash_pos]
                            self.partition_dag.add_edge(node["input"][idx].split(":")[0], node["name"])

        if True:
            ### Only dump these info for rank 0   
            op_dict = {}
            for op in ops:
                op_dict[op.name] = {
                    "output":[serialize_tensor(e) for e in op.outputs],
                    "input": [serialize_tensor(e) for e in op.inputs._inputs],
                    "op": op.type
                }
            with open(os.path.join(self.trace_dir, "metadata.json"), "w") as f:
                json.dump(op_dict, f, indent=4)

            if self.partition_dag is not None:
                nx.write_gml(self.partition_dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
            
            with open(os.path.join(self.trace_dir, "tensor_shapes.json"), "w") as f:
                json.dump(self.shape_dict, f, indent=4)

        print("Stop tracing, output trace at %s" % self.trace_dir)

    def chome_trace_MBE2X(self, raw_traces):
        ret = []
        pid_table = {}
        if self.dag is None:
            _dag = nx.DiGraph()
        for trace in raw_traces:
            ### Create the DAG
            if self.dag is None:
                if trace["ph"] == "M" or "args" not in trace:
                    continue
                op = trace["args"]["op"]
                name = trace["args"]["name"]
                if name.startswith("^"):
                    name = name[1:]
                ### Add dependency info
                for k, v in trace["args"].items():
                    if "input" in k:
                        if v.startswith("^"):
                            v = v[1:]
                        _dag.add_edge(v, name)
                    
            if trace["ph"] == "M":
                if trace["name"] == "process_name":
                    assert trace["pid"] not in pid_table
                    if trace["args"]["name"] == "":
                        continue
                    process_name = trace["args"]["name"]
                    if "stream:all Compute" in process_name and "device:GPU" in process_name:
                        pid_table[trace["pid"]] = {"process_name": process_name}
                else:
                    pass
            elif trace["ph"] == "i":
                trace["pid"] = trace["tid"] = "mark"
                ret.append(trace)
            elif trace["pid"] in pid_table and trace["ph"] == "X":
                cur_pid = pid_table[trace["pid"]]
                trace["pid"] = cur_pid["process_name"]
                ret.append(trace)
            else:
                pass
        if self.dag is None:
            self.dag = _dag
        return ret



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


args = parser.parse_args()
args.cuda = not args.no_cuda

# Horovod: initialize Horovod.
# hvd.init()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
if args.cuda:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Set up standard model.
model = getattr(applications, args.model)(weights=None)
opt = tf.optimizers.SGD(0.01)
hooks = [TimelineHook(batch_size=args.batch_size)]
data = tf.random.uniform([args.batch_size, 224, 224, 3])
target = tf.random.uniform([args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)


global_step = _tf.train.get_or_create_global_step()
probs = model(data, training=True)
loss = tf.losses.sparse_categorical_crossentropy(target, probs)
train_opt = opt.minimize(loss)


def log(s, nl=True):
    # if hvd.rank() != 0:
    #     return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, 1))

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

with tf.device(device):
    with _tf.train.MonitoredTrainingSession(hooks=hooks) as mon_sess:
        run(lambda: mon_sess.run(train_opt))
'''
with tf.device(device):
    # Warm-up
    log('Running warmup...')
    benchmark_step(first_batch=True)
    timeit.timeit(lambda: benchmark_step(first_batch=False),
                  number=args.num_warmup_batches)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(lambda: benchmark_step(first_batch=False),
                             number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
    log('Total img/sec on %d %s(s): %.1f +-%.1f' %
        (1, device, 1 * img_sec_mean, 1 * img_sec_conf))
'''