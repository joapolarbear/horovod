import sys, os
import json
import argparse
import xlsxwriter

parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
parser.add_argument('--trace_path', type=str, default="/root/traces/0/temp.json", help='trace_path')
parser.add_argument('--rst_dir', type=str, default="./", help='rst_dir')
parser.add_argument('--model', type=str, default="mnist", help='model')
parser.add_argument('--save_names', type=str, default="None", choices=["None", "fp16", "fp32"], help='save names')
parser.add_argument('--option', type=str, default='plot', choices=['plot', 'print', 'xlsx'])
parser.add_argument('--platform', type=str, default='mx', choices=['mx', 'tf'])
parser.add_argument('--combine_method', type=str, default='add', choices=['add', 'minus'])
args = parser.parse_args()


MAX_CNT = None

if args.platform == 'tf':
    MNIST_CANDIDATES = ["Conv2D", "BiasAdd", "Relu", "MatMul", "Mul", "Cast", "BiasAddGrad", "ApplyAdam", "ReluGrad", "Conv2DBackpropInput", "Conv2DBackpropFilter"]
    RESNET50_CANDIDATES = ["Conv2D", "BiasAdd", "Relu", "MatMul", "Mul", "Cast"]
else:
    MNIST_CANDIDATES = RESNET50_CANDIDATES = ["conv", "BiasAdd", "Relu", "MatMul", "Mul", "Cast"]


def is_the_same_op(e1, e2):
    for key in ["name", "cat", "ph", "pid", "tid"]:
        if e1.get(key) != e2.get(key):
            return False
    for key in ["name", "op"]:
        if e1["args"].get(key) != e2["args"].get(key):
            return False
    return True

def mx_collect_traces():
    with open(args.trace_path, "r") as f:
        temp = json.load(f)
    return temp

def tf_collect_traces():
    with open(args.trace_path, "r") as f:
        temp = json.load(f)
    traces = {"traceEvents": []}
    pid_dict = {}
    op_dict = {}
    idx = 0
    while idx < len(temp["traceEvents"]):
        event = temp["traceEvents"][idx]
        idx += 1
        if event["ph"] == "M" and event["name"] == "process_name":
            if event["pid"] not in pid_dict:
                pid_dict[event["pid"]] = event["args"]["name"]
                traces["traceEvents"].append(event)
            else:
                pass
            continue
        
        if event["ph"] == "X" and "#id=" in event["name"]:
            step_id = int(event["name"].split("#id=")[1].split("#")[0])
            op_name = event["args"]["name"]
            if op_name not in op_dict:
                op_dict[op_name] = {}
            if step_id not in op_dict[op_name]:
                ### first sub_op for this op in this step
                op_dict[op_name][step_id] = len(traces["traceEvents"])
                traces["traceEvents"].append(event)
            else:
                ### the following sub ops
                op_idx = op_dict[op_name][step_id]
                if args.combine_method == 'add':
                    traces["traceEvents"][op_idx]["dur"] += event["dur"]
                elif args.combine_method == 'minus':
                    traces["traceEvents"][op_idx]["dur"] = max(traces["traceEvents"][op_idx]["dur"],
                                    event["ts"] + event["dur"] - traces["traceEvents"][op_idx]["ts"])
        else:
            traces["traceEvents"].append(event)
    return traces

class TraceUtil:
    def __init__(self, traces):
        self.traces = traces["traceEvents"] if isinstance(traces, dict) else traces
        self.pid = None
        self.ret_stat()
        
    def _is_ignore_for_sta(self, event):
        ### store the pid for computation
        if event["ph"] == "M" and event["name"] == "process_name":
            if "stream:all Compute" in event["args"]["name"]:
                self.pid = event["pid"]
        if self.pid is not None and event["pid"] != self.pid:
            return True
        if event["ph"] == "M" or event["ph"] == "s" or event["ph"] == "t" \
            or event["name"] == "unknown":
            return True
        if "resnet" in args.model.lower():
            _CANDIDATES = RESNET50_CANDIDATES
        elif "mnist" in args.model.lower():
            _CANDIDATES = MNIST_CANDIDATES
        elif "bert" in args.model.lower():
            _CANDIDATES = None
        elif "dense" in args.model.lower():
            _CANDIDATES = ["_dense", "MatMul", "Mat", "Cast"]
        if _CANDIDATES is None:
            return False
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
                    "cat": event["cat"],
                    "id": len(self.name2sta)
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
        var = []
        for name, statistic in sorted(self.name2sta.items()):        
            nameL.append(name)
            avg.append(statistic["avg"])
            var.append(statistic["var"])
        # print(nameL, avg)
        if args.save_names != "None":
            with open(os.path.join(args.rst_dir, "name.txt"), "a") as f:
                f.write("{}:{}\n".format(args.save_names, str(nameL)))
        with open(os.path.join(args.rst_dir, "avg.txt"), "a") as f:
            f.write(str(avg) + "\n")
            f.write(str(var) + "\n")

    def export2xlsx(self, _stats=None, _dir=None, filename=None, sheet_name=None):
        ''' Export the statitic results to an XLSX file

        Parameters
        ----------
        _stats: list
            A list of statitic results
        _dir: str
            The directory to store the XLSX file
        '''
        _stats = [self.name2sta] if _stats is None else _stats
        filename = 'statistic.xlsx' if filename is None else filename + ".xlsx"
        _dir = "." if _dir is None else _dir
        workbook = xlsxwriter.Workbook(os.path.join(_dir, filename))
        for idx, _stat in enumerate(_stats):
            worksheet = workbook.add_worksheet(sheet_name[idx] if sheet_name is not None else None)
            row = 0
            header = []
            for name, statistic in sorted(_stat.items()):
                if row == 0:
                    # -- Output the header of the sheet
                    col = 0
                    worksheet.write(row, col, "Name")
                    for key in statistic:
                        col += 1
                        header.append(key)
                        worksheet.write(row, col, key)
                row += 1
                col = 0
                worksheet.write(row, col, name)
                for key in header:
                    col += 1
                    worksheet.write(row, col, statistic[key])
        workbook.close()

traces = tf_collect_traces() if args.platform == 'tf' else mx_collect_traces()
trace_util1 = TraceUtil(traces)
if args.option == 'plot':
    trace_util1.used_for_plot()
elif args.option == 'print':
    trace_util1.show_stat()
elif args.option == 'xlsx':
    trace_util1.export2xlsx(_dir=args.rst_dir)
else:
    raise







