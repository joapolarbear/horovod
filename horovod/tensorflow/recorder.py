import tensorflow as tf
from google.protobuf.json_format import MessageToJson
import json
import networkx as nx
import struct, math
import numpy as np
    
def float_to_bits(f):
    s = struct.pack('>f', f)
    return struct.unpack('>l', s)[0]

def precision_loss(fp32):
    assert "float" in str(type(fp32))
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
        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(hvd.local_rank()))
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

    def run(self, *args, **kwargs):
        if self._end_trace:
            self.sess.run(*args, **kwargs)
        elif not self._end_trace and self.step_cnt < self.start_step:
            self.sess.run(*args, **kwargs)
            self.step_cnt += 1
        elif not self._end_trace and self.step_cnt < self.end_step:
            self.sess.run(*args, options=self.run_options, run_metadata=self.run_metadata, **kwargs)
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

    def scheduler(self, grads, vars):
    	for i in len(grads):
    		grad = grads[i]
    		pl_sum = pl_cnt = 0
            for g in grad.flatten():
            	pl_sum += precision_loss(g)
            	pl_cnt += 1
            print("%f" % (pl_sum/pl_cnt))

            print(precision_loss_np(grad))
        	raise


    
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

        		




