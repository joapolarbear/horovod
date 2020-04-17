# append for auto_profiling
import sys, os
from mxnet import profiler
import json
import networkx as nx
import threading
import time

from horovod.mxnet.mpi_ops import size, local_size, rank, local_rank

parameter_index = 0

QueueType = [
  "COORDINATE_REDUCE",
  "REDUCE",
  "COPYD2H",
  "PCIE_REDUCE",
  "COORDINATE_PUSH",
  "PUSH",
  "PULL",
  "COPYH2D",
  "COORDINATE_BROADCAST",
  "BROADCAST",
  "QUEUE_NUM_AND_NOT_A_REAL_QUEUE_TYPE_AND_MUST_BE_THE_LAST"
]

def BYTEPS_TRACE_DEBUG(s, debug=False):
    #! log debug info when debug is True and env HHP_DEBUG is set
    if rank() == 0 and ((debug and os.getenv("HHP_DEBUG", None)) or not debug) :
        print(s)
        sys.stdout.flush()

class Recorder(object):
    #! class used to collect trace info
    def __init__(self, profile_symbolic=True,
                    profile_imperative=False,
                    profile_memory=False,
                    profile_api=False,
                    aggregate_stats=False):
        self.time_dict = {"traceEvents":[]}
        self.idx_dict = {}
        self.gradient_name_list = None
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
        # self.trace_path = self.trace_dir + 'bps_trace_local_rank%s_%dstep.json' % (os.environ.get("BYTEPS_LOCAL_RANK"), self.end_step)

        """config the mxnet profile"""
        profiler.set_config(profile_symbolic=profile_symbolic,
                    profile_imperative=profile_imperative,
                    profile_memory=profile_memory,
                    profile_api=profile_api,
                    # profile_process=False,
                    aggregate_stats=aggregate_stats, 
                    filename=os.path.join(self.trace_dir, 'temp.json'))

        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")
        if self.step_cnt == self.start_step - 1:
            profiler.set_state('run')

        self.dag = None
        self.loss_dag = []

        ### symbol/block, used to get the dependency info, at least one should be given
        self.block = None
        self.symbol = None
        self.loss = None
        ### Used to decide how many weights will be updated in a single updater
        self.opt_aggregate_num = 0

    def scheduler(self, index, _check_stop=False):
        '''A scheduler, manage the counter for each gradient, `self.idx_dict` is 
        used to record the status of each gradient, the fist time a gradinet call 
        this function, register the `index` to self.idx_dict with False; when it
        becomes True, this gradinet is ready to output traces (the communication 

        traces of this gradient have been collected); Output traces only when 
        the status of gradients are True.

        Parameters:
        ----------
        index : int
            The index of the gradient.
        _check_stop : bool
            if the flag is set, add the step_cnt by 1.

        Returns:
        ----------
        bool, whether to collect the communication trace of this gradinet.
        '''
        if self._end_trace:
            return False
        if index not in self.idx_dict:
            self.idx_dict[index] = False
            
        def get_traces(self):
            #! Sleep to wait for all the communication traces have been printed.
            time.sleep(5) 
            self.save_trace()

        if self.idx_dict[index]:
            if False not in self.idx_dict.values():
                """All parameters have been recorded, end profiling"""
                self._end_trace = True 
                #! Inform IO recorder to stop profiling
                os.environ["BYTEPS_TRACE_STATUS"] = "END"
                #! Output mxnet traces and import it
                profiler.set_state('stop')
                #! Create a new thread to process traces
                _t = threading.Thread(target=get_traces, args=(self,))
                _t.start()            
            return False # the communication traces of this parameter have been read

        """ Since each parameter will call this function, to decide when to stop profiling,
            we only focus on one parameter, e.g., the first parameter.
            i.e., only the first parameter can add self.step_cnt by one
        """
        if _check_stop:
            self.step_cnt += 1

        """ Start profiling one step ahead
        """
        if self.step_cnt == self.start_step - 1:
            profiler.set_state('run')
            
        if self.step_cnt >= self.end_step:
            if self.gradient_name_list is None:
                self.gradient_name_list = []
                with open(os.path.join(self.trace_dir, 'arg_namesINpara_names.txt'), 'r') as lines:
                    for line in lines:
                        name = line[:-1]
                        self.gradient_name_list.append(name)
            return True
        else:
            return False            

    def end_trace(self):
        return self._end_trace

    def save_trace(self):
        profiler.dump()
        #! Get the dependency graph, adapt to DistributedOptimizer and DistributedTrainer
        if self.symbol is not None:
            self.dag = self.gen_dag(self.symbol.debug_str(), _main=True)      
        elif self.block is not None:
            symbol = self.block._cached_graph[1]
            self.dag = self.gen_dag(symbol.debug_str(), _main=True)
            self.combine_loss_dag()
        else:
            raise ValueError("A symbol or model/block must be given when defining DistributedOptimizer/DistributedTrainer.")

        #! Output the dag, only containing forward info
        nx.write_gml(self.dag, os.path.join(self.trace_dir, "dag.gml"), lambda x: str(x))
        BYTEPS_TRACE_DEBUG("Stop tracing, output trace: %s" % self.trace_dir)

        ### Record optimizer aggregate num
        with open(os.path.join(self.trace_dir, "info.json"), "w") as f:
            json.dump({"opt_aggregate_num": self.opt_aggregate_num}, f, indent=4)

        if self.gradient_name_list is None:
            return 

        with open(os.path.join(self.trace_dir, "gradient_name_list.txt"), "w") as f:
            for s in self.gradient_name_list:
                f.write(str(s) + "\n")

    def gen_dag(self, s, _str_name="symbol_debug_str", _main=False):
        """Construct a DAG from the mxnet info

        Parameters:
        ----------
        s : str
            Must follow the standard chrome trace format and not None.
        """
        with open(os.path.join(self.trace_dir, _str_name + ".txt"), "w") as f:
            f.write(s)
        _dag = nx.DiGraph()
        blocks = s.split("--------------------\n")
        
        #! 3. FW -> OUTPUT and 4. OUTPUT -> BW
        first_ls = blocks[0].split('\n')
        output_cnt = 0
        for i in range(len(first_ls)):
            if "Variable:" in first_ls[i]:
                break
            if "output[" in first_ls[i]:
                output_node = first_ls[i].split(']=')[1].split('(')[0]
                output_node = output_node.split("_fwd")[0] if "_fwd" in output_node else output_node
                _dag.add_edge("FW." + output_node, "OUTPUT%d"%output_cnt)
                _dag.add_edge("OUTPUT%d"%output_cnt, "BW." + output_node)
                output_cnt += 1

        for i in range(1, len(blocks)):
            prev_block = blocks[i-1]
            var = []
            prev_ls = prev_block.split('\n')
            for l in prev_ls:
                if "Variable" in l:
                    var.append(l.split('Variable:')[1])
            block = blocks[i]
            ls = block.split('\n')
            if 'Name' not in ls[0]:
                continue
            name = ls[0].split('Name=')[1]
            op = ls[0].split(',')[0].split("Op:")[1]
            args = []
            for l in ls:
                if "arg[" in l:
                    arg_name = l.split(']=')[1].split('(')[0]
                    if arg_name not in var:
                        args.append(arg_name)
            if "_fwd" in name:
                name = name.split("_fwd")[0]

            #! --------- construct the graph ----
            _dag.add_node("FW." + name, op=op)
            _dag.add_node("BW." + name, op=op)
            for innode in args:
                innode = innode.split("_fwd")[0] if "_fwd" in innode else innode
                #! 2. FW -> FW and 5. BW -> BW
                _dag.add_edge("FW." + innode, "FW." + name)
                _dag.add_edge("BW." + name, "BW." + innode)
            for _var in var:
                if "data" in _var:
                    _dag.add_edge("I/O", "FW." + name)
                    if _main:
                        #! 1. IO -> FW, 8. BW -> UPDATE -> FW                  
                        _dag.add_edge("BW." + name, "UPDATE")
                        _dag.add_edge("UPDATE", "FW." + name)
                else:
                    #! 7. Comm -> FW and 6. BW -> Comm
                    _dag.add_edge("Comm." + _var, "UPDATE")
                    _dag.add_edge("BW." + name, "Comm." + _var)
        return _dag

    def combine_loss_dag(self):
        if self.loss is None:
            return
        ### if loss DAGs are given, add loss nodes to the graph
        self.loss_dag = [(self.gen_dag(l._cached_graph[1].debug_str(), _str_name="loss%d"%i) if l is not None else None) for i, l in enumerate(self.loss)]
        for idx, ld in enumerate(self.loss_dag):
            if ld is None:
                continue
            output_name = "OUTPUT%d"%idx
            output_node = [u for u, _ in self.dag.in_edges(output_name)][0]
            first_bw_node = list(self.dag.successors(output_name))[0]
            for u, v in ld.edges():
                if "I/O" in u:
                    self.dag.add_edge(output_node, v)
                    self.dag.add_edge("BW." + v.split("FW.")[1], first_bw_node)
                elif "OUTPUT" in u:
                    self.dag.add_edge(output_name, v)
                elif "OUTPUT" in v:
                    self.dag.add_edge(u, output_name)
                else: 
                    self.dag.add_edge(u, v)

        self.loss_dag = None


    def end4index(self, index, tensor, name):
        ''' Offline collect the communication trace results of gradient `index`

        Parameters
        ----------
        index : int
            The index of the gradient.
        tensor: tensor
            A tensor to average and sum.
        name : str
            A name of the reduction operation.
        '''
        if self.end_trace():
            return
        self.idx_dict[index] = True # avoid repeatedly read