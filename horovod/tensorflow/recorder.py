import json
import networkx as nx
import struct, math
import numpy as np
import os, sys
import threading
import time

from horovod.tensorflow.mpi_ops import local_rank, rank

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
from google.protobuf.json_format import MessageToJson
from tensorflow.python.client import timeline

def host_log(s):
    if rank() == 0:
        print("[Horovod] " + s)

def serialize_tensor(t):
    return {
        "name": t.name,
        "shape": t.shape.as_list() if t.shape.dims is not None else [],
        "dtype": t.dtype.name
    }

class TimelineSession:
    def __init__(self, sess):
        self.sess = sess
        self.graph = sess.graph
        self.step_cnt = 0

        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(local_rank()))
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
                _t = threading.Thread(target=self.output_traces, args=(tf.get_default_graph().get_operations(),))
                _t.start()

        ### Return all fetches
        return ret

    
    def output_traces(self, ops):
        with open(os.path.join(self.trace_dir, "temp.json"), "w") as f:
            json.dump(self.traces, f, indent=4)

        ### collect graph info
        # graphdef = tf.get_default_graph().as_graph_def()
        # graph_str = json.loads(MessageToJson(graphdef))
        # with open(os.path.join(self.trace_dir, "graph.json"), "w") as f:
        #     json.dump(graph_str, f, indent=4)
            
        op_dict = {}
        for op in ops:
            op_dict[op.name] = {
                "output":[serialize_tensor(e) for e in op.outputs],
                "input": [serialize_tensor(e) for e in op.inputs._inputs],
                "op": op.type
            }
        with open(os.path.join(self.trace_dir, "metadata.json"), "w") as f:
            json.dump(op_dict, f, indent=4)

        nx.write_gml(self.dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
        print("Stop tracing, output trace: %s" % self.trace_dir)

    def should_stop(self):
        return self.sess.should_stop()

def profile(recorder):
    def decorate(func):
        def wrapper(*args, **kwargs):
            func(*args, **kwargs)
            recorder.schedule()
        return wrapper
    return decorate

class Recorder(object):
    def __init__(self, model=None, batch_size=None, opt=None):
        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            return
        self._end_trace = False

        if model is None or batch_size is None:
            raise ValueError("The `model` and `batch_size` must be given to enable auto-profiling")
        self.model = model
        self.batch_size = batch_size
        self.opt = opt
        
        ### Profiling related env
        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(local_rank()))
        self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))
        self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)
        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")
        
        self.gradient_name_list = []
        self.step_cnt = 0
        self.start_time_ns = 0

        host_log("Auto-profiling from step {} to step {}, save traces at {}".format(self.start_step, self.end_step, self.trace_dir))

    def register_tensors(self, grads):
        new_grads = []
        for grad in grads:
            if grad.name not in self.gradient_name_list:
                self.gradient_name_list.append(grad.name)
            grad = tf.identity(grad, name=str(self.gradient_name_list.index(grad.name)))
            new_grads.append(grad)
        if self._end_trace:
            return new_grads
        _t = threading.Thread(target=self.output_traces)
        _t.start()
        return new_grads

    def schedule(self):
        # host_log("step {}".format(self.step_cnt))
        if self._end_trace:
            return
        if self.step_cnt == self.start_step:
            self.trace_start()
        elif self.step_cnt == self.end_step:
            self.trace_end()
            self._end_trace = True
        self.step_cnt += 1

    # def output_traces(self):
    #     if rank() != 0:
    #         return
    #     with open(os.path.join(self.trace_dir, "gradient_name_list.json"), "w") as f:
    #         json.dump({"gradient_name_list": self.gradient_name_list}, f, indent=4)
    
    def trace_start(self):
        ### https://tensorflow.google.cn/api_docs/python/tf/profiler/experimental/ProfilerOptions?hl=zh-cn
        #   `host_tracer_level`: Adjust CPU tracing level. Values are: 1 - critical info 
        #       only, 2 - info, 3 - verbose. [default value is 2]
        #   `python_tracer_level`: Toggle tracing of Python function calls. Values are: 1
        #       enabled, 0 - disabled [default value is 0]
        #   `device_tracer_level`: Adjust device (TPU/GPU) tracing level. 
        #       Values are: 1 - enabled, 0 - disabled [default value is 1]
        #   `delay_ms`: Requests for all hosts to start profiling at a timestamp 
        #       that is delay_ms away from the current time. delay_ms is in milliseconds. 
        #       If zero, each host will start profiling immediately upon receiving the request. 
        #       Default value is None, allowing the profiler guess the best value.
        options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 2,
                                                   python_tracer_level = 0,
                                                   device_tracer_level = 1)
        self.start_time_ns = tf.profiler.experimental.start(self.trace_dir, options = options)
        host_log("Start profiling ...")
    
    def trace_end(self):
        ts = time.time()
        tf.profiler.experimental.stop()
        host_log("It takes {:.3f} s to stop the TF profiler".format(time.time() - ts))
        _t = threading.Thread(target=self.output_traces)
        _t.start()
        # self.serializeGraph()
        # self.output_traces()

    def dump_dfg(self):
        ''' Dump the DFG
            NOTE: Assumptions:
                * Model outputs are one dimentional
                * MOdel outputs share the same data type as inputs
        '''
        ### Create the ConcreateFunction
        def _full_model(x, y):
            with tf.GradientTape() as tape:
                probs = self.model(x, training=True)
                loss = tf.losses.sparse_categorical_crossentropy(y, probs)
            gradients = tape.gradient(loss, self.model.trainable_variables)
            self.opt.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        new_shape_x = []
        for dim in self.model.inputs[0].shape:
            if dim is None:
                new_shape_x.append(self.batch_size)
            else:
                new_shape_x.append(dim)
        new_shape_y = [self.batch_size]

        full_model = tf.function(_full_model)
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(new_shape_x, self.model.inputs[0].dtype), tf.TensorSpec(new_shape_y, self.model.inputs[0].dtype))
        
        ### Dump the metadata
        op_dict = {}
        for op in full_model.graph.get_operations():
            op_dict[op.name] = {
                        "output":[serialize_tensor(e) for e in op.outputs],
                        "input": [serialize_tensor(e) for e in op.inputs],
                        "op": op.type
                    }
        with open(os.path.join(self.trace_dir, "metadata.json"), "w") as f:
            json.dump(op_dict, f, indent=4)

        ### Parse the DFG
        graph_def = full_model.graph.as_graph_def()
        graph_json = json.loads(MessageToJson(graph_def))
        with open(os.path.join(self.trace_dir, "partition_def_0.json"), "w") as f:
            json.dump(graph_json, f, indent=4)

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

        if self.partition_dag is not None:
            nx.write_gml(self.partition_dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
        
        ### TODO (huhanpeng): shape_dict, used for XLA cost model, needs to be adapted to TF 2.4
        # with open(os.path.join(self.trace_dir, "tensor_shapes.json"), "w") as f:
        #     json.dump(self.shape_dict, f, indent=4)

        
    def output_traces(self):
        ts = time.time()
        if not os.path.exists(self.trace_dir):
            os.mkdir(self.trace_dir)

        ### 1. clean up the traces
        dirs = os.listdir(os.path.join(self.trace_dir, "plugins/profile"))
        os.system(
            "cp {}/plugins/profile/{}/*.trace.json.gz {}/trace.json.gz && ".format(self.trace_dir, dirs[0], self.trace_dir) + \
            "cp {}/plugins/profile/{}/*.memory_profile.json.gz {}/memory_profile.json.gz  && ".format(self.trace_dir, dirs[0], self.trace_dir) + \
            "rm -rf {}/plugins {}/events*".format(self.trace_dir, self.trace_dir)
        )

        trace_path = "{}/trace.json.gz".format(self.trace_dir)
        catapult_path = os.environ.get("BYTEPS_CATAPULT_DIR", None)
        json_path = trace_path.replace(".gz", "")
        os.system("gzip -fdk {}".format(trace_path))
        ### re-align the timestamps
        host_log("Re-algin timestamps, convert relative time to absolute time {} ...".format(self.start_time_ns))
        with open(json_path, 'r') as fp:
            traces = json.load(fp)
        for trace in traces["traceEvents"]:
            if "ts" in trace:
                trace["ts"] = int(trace["ts"] + self.start_time_ns / 1000)
        with open(json_path, 'w') as fp:
            json.dump(traces, fp, indent=4)
    
        if catapult_path and os.path.exists(catapult_path):
            os.system("python {} {}".format(os.path.join(catapult_path, "tracing/bin/trace2html"), json_path))

        os.system("rm {} && gzip {}".format(trace_path, json_path))

        if rank() != 0:
            return
            
        ### 2. convert the dynamic graph to the static graph
        full_model = tf.function(lambda x: self.model(x))
        new_shape = []
        for dim in self.model.inputs[0].shape:
            if dim is None:
                new_shape.append(self.batch_size)
            else:
                new_shape.append(dim)

        ### 3. Get the `ConcreteFunction`
        host_log("get concrete function {} {}".format(new_shape, self.model.inputs[0].dtype))
        full_model = full_model.get_concrete_function(tf.TensorSpec(new_shape, self.model.inputs[0].dtype))
        host_log("Freeze variables")
        frozen_func = convert_variables_to_constants_v2(full_model)

        ### 4. dump graph_def
        # graph_def = frozen_func.graph.as_graph_def()
        # graph_json = json.loads(MessageToJson(graph_def))
        # with open(os.path.join(self.trace_dir, "graph_def.json"), 'w') as f:
        #     json.dump(graph_json, f, indent=4)

        ### 5. dump tensor shapes
        # op_dict = {}
        # for op in frozen_func.graph.get_operations():
        #     op_dict[op.name] = {
        #                 "output":[serialize_tensor(e) for e in op.outputs],
        #                 "input": [serialize_tensor(e) for e in op.inputs],
        #                 "op": op.type
        #             }

        # with open(os.path.join(self.trace_dir, "metadata.json"), "w") as f:
        #     json.dump(op_dict, f, indent=4)
        
        ### 6. Dump the metadata and DFG
        self.dump_dfg()

        host_log("Succcessfully dump metadata in {:.3f} s".format(time.time() - ts))

### Compatible to TF 1.15

class _SecondOrStepTimer(tf.train.SecondOrStepTimer):
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

class TimelineHook(tf.train.ProfilerHook):
    def __init__(self, _summary=False, batch_size=None):
        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(local_rank()))
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
        self._file_writer = tf.summary.FileWriterCache.get(self.trace_dir) if _summary else None
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
        opts = (tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE, output_partition_graphs=True)
            if self._request_summary else None)

        t = time.time() - t
        # print("Before run takes: {} seconds".format(t))
        return tf.train.SessionRunArgs(requests, options=opts)

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
            if rank() == 0 and local_rank() == 0:
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

        if rank() == 0:
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
