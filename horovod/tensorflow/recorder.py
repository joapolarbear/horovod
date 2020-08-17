import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import json
import networkx as nx
import struct, math
import numpy as np
import os, sys
from horovod.tensorflow.mpi_ops import local_rank
from tensorflow.python.client import timeline
import threading

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

        def serialize_tensor(t):
            return {
                "name": t.name,
                "shape": t.shape.as_list() if t.shape.dims is not None else [],
                "dtype": t.dtype.name
            }
            
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

def float_to_bits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]

def precision_loss(fp32):
    assert "float" in str(type(fp32)), "Type error: %s"%(str(type(fp32)))
    if fp32 == 0:
        return 0.0
    elif fp32 < 0:
        fp32 = - fp32
    LARGEST_NORMAL_FP16 = 65504              # 2^15 * (1 + 1023/1024)
    SMALLEST_NORMAL_FP16 = 0.000060976       # 2^-14
    SMALLEST_SUBNOMRAL_FP16 = 0.000000059605 # 2^-24
    fp32_bits = float_to_bits(fp32)
    sign = (fp32_bits >> 31) & 0x1
    expo = (fp32_bits >> 23) & 0xff - 0x7f
    prec = fp32_bits & 0x7fffff
    # print(hex(fp32_bits), sign, expo, prec)
    if fp32 > LARGEST_NORMAL_FP16:
        # print("Overflow")
        return (fp32 - LARGEST_NORMAL_FP16) / fp32
    elif fp32 < SMALLEST_SUBNOMRAL_FP16:
        # print("Underflow")
        return 1.0
    elif fp32 < SMALLEST_NORMAL_FP16:
        # print("Subnormal")
        ###  additional precision loss: (-14) - (exp_fp_32 - 127)
        addition_bit = -14 - expo
        return ((((1 << (14 + addition_bit)) - 1) & fp32_bits) * math.pow(2, expo - 23)) / fp32
    else:
        # print("Normal")
        return ((fp32_bits & 0x1fff) * math.pow(2, expo - 23)) / fp32

def precision_loss_np(fp32):
    assert isinstance(fp32, np.ndarray)
    assert fp32.dtype is np.dtype('float32')
    fp16 = np.float16(fp32)
    pl = np.abs(fp32 - fp16) / fp32
    pl[np.isnan(pl)] = 0
    return np.average(pl)


class Recorder(object):
    def __init__(self):
        self.step_cnt = 0

        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            return
        self._end_trace = False
        self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))
        self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))
        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(local_rank()))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)

        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")   

        ### Timeline configuratoin
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.traces = {"traceEvents":[]}

        self.dag = None

    def get_dag(self):
        raise NotImplementedError()
        for _dict in self.graph_dict["node"]:
            if _dict["op"].lower in ["const", "variable", "variablev2"]:
                ###
                pass 

    def run(self, *args, **kwargs):
        if self._end_trace:
            pass
        elif not self._end_trace and self.step_cnt < self.start_step:
            self.step_cnt += 1
        elif not self._end_trace and self.step_cnt < self.end_step:
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

    def scheduler(self, grads, vars):
        # print(grads)
        for i in range(len(grads)):
            grad = grads[i]
            print(grad)
            pl_sum = pl_cnt = 0
            for g in grad:
                pl_sum += precision_loss(g)
                pl_cnt += 1
            print("%f" % (pl_sum/pl_cnt))

            print(precision_loss_np(grad))
            raise

    def output_traces(self, ops):
        with open(os.path.join(self.trace_dir, "temp.json"), "w") as f:
            json.dump(self.traces, f, indent=4)

        ### collect graph info
        # graphdef = tf.get_default_graph().as_graph_def()
        # graph_str = json.loads(MessageToJson(graphdef))
        # with open(os.path.join(self.trace_dir, "graph.json"), "w") as f:
        #     json.dump(graph_str, f, indent=4)

        def serialize_tensor(t):
            return {
                "name": t.name,
                "shape": t.shape.as_list() if t.shape.dims is not None else [],
                "dtype": t.dtype.name
            }
            
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
            if step < self._start_step or step > self._end_step:
                return False

        return super(_SecondOrStepTimer, self).should_trigger_for_step(step)

class TimelineHook(tf.train.ProfilerHook):
    def __init__(self, _summary=False):
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
            
        self.dag = None
        self.has_data = False

        self._output_file = os.path.join(self.trace_dir, "timeline-{}.json")
        self._file_writer = tf.summary.FileWriterCache.get(self.trace_dir) if _summary else None
        self._show_dataflow = True
        self._show_memory = False
        self._timer = _SecondOrStepTimer(
            every_secs=None, every_steps=1, step_bound=(self.start_step, self.end_step))

    def before_run(self, run_context):
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
                _t = threading.Thread(target=self.output_traces, args=(tf.get_default_graph().get_operations(),))
                _t.start() 

        return super(TimelineHook, self).before_run(run_context)

    def after_run(self, run_context, run_values):
        stale_global_step = run_values.results["global_step"]
        if self._next_step is None:
        # Update the timer so that it does not activate until N steps or seconds
        # have passed.
            self._timer.update_last_triggered_step(stale_global_step)
        global_step = stale_global_step + 1
        if self._request_summary:
            global_step = run_context.session.run(self._global_step_tensor)
            self._timer.update_last_triggered_step(global_step)
            self._save(global_step, self._output_file.format(global_step),
                     run_values.run_metadata.step_stats)
            if self._file_writer is not None:
            	self._file_writer.add_run_metadata(run_values.run_metadata,
                                         "step_%d" % global_step)
        self._next_step = global_step + 1

    def create_dag(self, ctf):
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
        assert not_found, "Cycles are not allowd for the DAG"

    def output_traces(self, ops):
        self.traces = {"traceEvents":[]}    
        ### the ProfilerHook of tensorflow will output the timeline to self.trace_dir/timeline-{global_step}.json
        for file in os.listdir(self.trace_dir):
            if file.startswith('timeline-'):
                with open(os.path.join(self.trace_dir, file), 'r') as fp:
                    ctf = json.load(fp)
                self.traces["traceEvents"] += ctf["traceEvents"]
                ### Create the DAG
                if self.dag is None:
                    self.create_dag(ctf)
        with open(os.path.join(self.trace_dir, "temp.json"), "w") as fp:
            json.dump(self.traces, fp, indent=4)
        ### delete all intermediate redults
        _output_files = os.path.join(self.trace_dir, "timeline-*.json")
        os.system('rm {}'.format(_output_files))

        def serialize_tensor(t):
            return {
                "name": t.name,
                "shape": t.shape.as_list() if t.shape.dims is not None else [],
                "dtype": t.dtype.name
            }
            
        op_dict = {}
        for op in ops:
            op_dict[op.name] = {
                "output":[serialize_tensor(e) for e in op.outputs],
                "input": [serialize_tensor(e) for e in op.inputs._inputs],
                "op": op.type
            }
        with open(os.path.join(self.trace_dir, "metadata.json"), "w") as f:
            json.dump(op_dict, f, indent=4)

        if self.dag is not None:
            nx.write_gml(self.dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))

        print("Stop tracing, output trace at %s" % self.trace_dir)


