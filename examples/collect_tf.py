import sys, os
import json
import argparse

parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
parser.add_argument('--trace_path', type=str, default="/root/traces/0/temp.json", help='trace_path')
parser.add_argument('--rst_dir', type=str, default="./", help='rst_dir')
parser.add_argument('--model', type=str, default="mnist", help='model')
parser.add_argument('--save_names', type=str, default="None", choices=["None", "fp16", "fp32"], help='save names')
args = parser.parse_args()


MAX_CNT = None

MNIST_CANDIDATES = ["Conv2D", "BiasAdd", "Relu", "MatMul", "Mul", "Cast", "BiasAddGrad", "ApplyAdam", "ReluGrad", "Conv2DBackpropInput", "Conv2DBackpropFilter"]
RESNET50_CANDIDATES = ["Conv2D", "BiasAdd", "Relu", "MatMul", "Mul", "Cast"]

class TraceUtil:
	def __init__(self, traces):
		self.traces = traces["traceEvents"]
		self.ret_stat()

	def _is_ignore_for_sta(self, event):
		if event["ph"] == "M" or event["ph"] == "s" or event["ph"] == "t" \
			or event["name"] == "unknown":
			return True
		if "resnet" in args.model.lower():
			_CANDIDATES = RESNET50_CANDIDATES
		else:
			_CANDIDATES = MNIST_CANDIDATES
		for target in _CANDIDATES:
			if target in event["name"]:
				return False
		return True

	def ret_unique_name(self, event):
		return event["args"]["name"]

	def ret_stat(self):
		""" Basic Statistic """
		self.name2sta = {}
		for event in self.traces:
		    if self._is_ignore_for_sta(event):
		        continue
		    unique_name = self.ret_unique_name(event)
		    if unique_name in self.name2sta:
		        if MAX_CNT is not None and self.name2sta[unique_name]["cnt"] >= MAX_CNT:
		            event["args"]["cnt"] = -1
		            continue
		        self.name2sta[unique_name]["cnt"] += 1
		        self.name2sta[unique_name]["time"] += event["dur"] / 1000.0
		        self.name2sta[unique_name]["min_t"] = min(self.name2sta[unique_name]["min_t"], event["dur"] / 1000.0)
		        self.name2sta[unique_name]["max_t"] = max(self.name2sta[unique_name]["max_t"], event["dur"] / 1000.0)
		    else:
		        self.name2sta[unique_name] = {
		            "cnt": 1, 
		            "time": event["dur"] / 1000.0, 
		            "min_t": event["dur"] / 1000.0, 
		            "max_t": event["dur"] / 1000.0,
		            # \TODO: add `cat` field for communication traces
		            # "cat": event["cat"] 
		            "cat": event["cat"]
		            }
		    event["args"]["cnt"] = self.name2sta[unique_name]["cnt"] - 1
		        
		"""calculate the avg """
		for name, statistic in self.name2sta.items():
		    statistic["avg"] = statistic["time"] / statistic["cnt"]
		    statistic["var"] = 0.0

		"""calculate the variance"""
		for event in self.traces:
		    if self._is_ignore_for_sta(event):
		        continue
		    unique_name = self.ret_unique_name(event)
		    self.name2sta[unique_name]["var"] += pow(event["dur"] / 1000.0 - self.name2sta[unique_name]["avg"], 2)
		for name, statistic in self.name2sta.items():
		    statistic["var"] = statistic["var"] / float(statistic["cnt"])

	def show_stat(self):
		print("Profile Statistics.")
		print("===================")
		print("%-60s\t Total Count\t Time (ms)\t Min Time (ms)\t Max Time (ms)\t Avg Time (ms)\t Variance (ms^2)" % "Name")
		print("%-60s\t -----------\t ---------\t -------------\t -------------\t -------------\t ---------------" % "----")
		for name, statistic in sorted(self.name2sta.items()):		
			print("%-60s\t %11d\t %9.4f\t %12.4f\t %13.4f\t %13.4f\t %13.4f" % 
					(name,
					statistic["cnt"],
					statistic["time"],
					statistic["min_t"],
					statistic["max_t"],
					statistic["avg"],
					statistic["var"]
					))
	def used_for_plot(self):
		nameL = []
		avg = []
		for name, statistic in sorted(self.name2sta.items()):		
			nameL.append(name)
			avg.append(statistic["avg"])
		# print(nameL, avg)
		if args.save_names != "None":
			with open(os.path.join(args.rst_dir, "name.txt"), "a") as f:
				f.write("{}:{}\n".format(args.save_names, str(nameL)))
		with open(os.path.join(args.rst_dir, "avg.txt"), "a") as f:
			f.write(str(avg) + "\n")

with open(args.trace_path, "r") as f:
	traces = json.load(f)
trace_util1 = TraceUtil(traces)
trace_util1.used_for_plot()








