#!/bin/bash

### MXNet env
MXNET_CUDNN_AUTOTUNE_DEFAULT=0
MXNET_GPU_WORKER_NTHREADS=1
MXNET_EXEC_BULK_EXEC_TRAIN=0

### Profiling env
BYTEPS_TRACE_ON=1 
BYTEPS_TRACE_DIR=/root/traces/host0

# ---------------------- start to run ----------------------
DATA="/tmp/wiki_en_uncased_data/wiki_en_uncased_0*"
OPTIMIZER="bertadam"
## other evnvironment variables
export DMLC_ROLE="${DMLC_ROLE:-worker}"
# optimizer parameters
export LR=0.00354;   
export OPTIONS=--synthetic_data\ --eval_use_npz; 
export WARMUP_RATIO=0.1;          
export NUMSTEPS=281250;   
export CKPTDIR=ckpt_stage1_lamb_16k-682a361-c5fd6fc-0412-cu90; 
export ACC=1;  
# start
export TRUNCATE_NORM="${TRUNCATE_NORM:-1}"
export LAMB_BULK="${LAMB_BULK:-30}"
export EPS_AFTER_SQRT="${EPS_AFTER_SQRT:-1}"
export NUMSTEPS="${NUMSTEPS:-900000}"
export DTYPE="${DTYPE:-float16}"
export ACC="${ACC:-1}"
export MODEL="${MODEL:-bert_24_1024_16}"
export MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-128}"
export MAX_PREDICTIONS_PER_SEQ="${MAX_PREDICTIONS_PER_SEQ:-20}"
export LR="${LR:-0.000625}"
export LOGINTERVAL="${LOGINTERVAL:-10}"
export CKPTDIR="${CKPTDIR:-ckpt_stage1_lamb}"
export CKPTINTERVAL="${CKPTINTERVAL:-300000000}"
export OPTIMIZER="${OPTIMIZER:-lamb}"
export WARMUP_RATIO="${WARMUP_RATIO:-0.003125}"
export BYTEPS_PARTITION_BYTES="${BYTEPS_PARTITION_BYTES:-4096000}"
export BYTEPS_NCCL_GROUP_SIZE="${BYTEPS_NCCL_GROUP_SIZE:-16}"
# export NVIDIA_VISIBLE_DEVICES="${GPUS:-0,1,2,3,4,5,6,7}"
export DMLC_WORKER_ID="${DMLC_WORKER_ID:-0}"
export DMLC_NUM_WORKER="${DMLC_NUM_WORKER:-1}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_FWD:-120}"
export MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD="${MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD:-120}"
export MXNET_SAFE_ACCUMULATION="${MXNET_SAFE_ACCUMULATION:-1}"
export OPTIONS="${OPTIONS:- }"
export DATA="${DATA:-/data/book-corpus/book-corpus-large-split/*.train,/data/enwiki/enwiki-feb-doc-split/*.train}"
export DATAEVAL="${DATAEVAL:-/data/book-corpus/book-corpus-large-split/*.test,/data/enwiki/enwiki-feb-doc-split/*.test}"
export ROOT_DIR=/root

MPI_PREFIX="mpirun -np 1 -H net-g8:1        \
    -x NCCL_IB_DISABLE=1  \
    -x HOROVOD_FUSION_THRESHOLD=0 \
    -x HOROVOD_CYCLE_TIME=0 \
    -x HOROVOD_TIMELINE=/root/traces \
    -x HOROVOD_LOG_LEVEL=info \
    -x BYTEPS_TRACE_ON \
    -x BYTEPS_TRACE_DIR \
    -x BYTEPS_TRACE_START_STEP=90 \
    -x BYTEPS_TRACE_END_STEP=120 \
    -x HOROVOD_TIMELINE_PRETTY=1 \
    -x NCCL_DEBUG=INFO \
    -x NCCL_DEBUG_SUBSYS=NET \
    -x NCCL_ALGO=Ring \
    -x MXNET_GPU_WORKER_NTHREADS=1 \
    -x MXNET_EXEC_BULK_EXEC_TRAIN=0 \
    -x MXNET_USE_FUSION=0 \
    -x MXNET_FUSION_VERBOSE=1 \
    -x MXNET_EXEC_BULK_EXEC_MAX_NODE_TRAIN_BWD \
    -x MXNET_SAFE_ACCUMULATION \
    -bind-to none -map-by slot -mca plm_rsh_args '-p 12345' \
    -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib --allow-run-as-root"

ARGS_SUFFIX="--data=/tmp/wiki_en_uncased_data/wiki_en_uncased_0* \
        --data_eval='' \
        --optimizer bertadam --warmup_ratio 0.1 \
        --num_steps 200 --dtype float32 \
        --ckpt_dir ckpt_stage1_lamb_16k-682a361-c5fd6fc-0412-cu90 \
        --lr 0.00354 --accumulate 1 --model bert_24_1024_16 \
        --max_seq_length 128 --max_predictions_per_seq 20 --num_data_workers 4 \
        --no_compute_acc --comm_backend horovod --log_interval 10 \
        --gpus 0 --synthetic_data \
        --synthetic_data --eval_use_npz
"

RST_DIR=/root/traces
if [ "$1" == "tf" ]; then
	TRACE_PATH=${RST_DIR}/0/temp.json
	PYTHON_FILE=${HOME}/models/official/resnet/imagenet_main.py
elif [ "$1" == "mx" ]; then
	TRACE_PATH=${RST_DIR}/bps_trace_final.json
	PYTHON_FILE=${HOME}/gluon-nlp/scripts/bert/run_pretraining.py
	BPF_PATH=${HOME}/byteprofile-analysis/analyze.py
fi

MODEL="bert"
### Start to train
if [ ! -d ${RST_DIR} ]; then
	mkdir -p ${RST_DIR}
else
	rm -rf ${RST_DIR}/*
fi

function funcRunAndTest {
	rm $TRACE_PATH
	BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} $@
	python3 ${BPF_PATH} --option collect --path ${RST_DIR} --platform MXNET --sub_option xlsx --force
	echo "huhanpeng fp32: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model ${MODEL} --platform mx

	rm $TRACE_PATH
	TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=True BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} --amp $@
	python3 ${BPF_PATH} --option collect --path ${RST_DIR} --platform MXNET --sub_option xlsx --force
	echo "huhanpeng fp16: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model ${MODEL} --platform mx
}

function funcRunAndTestFirst {
	rm $TRACE_PATH
	BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} $@
	python3 ${BPF_PATH} --option collect --path ${RST_DIR} --platform MXNET --sub_option xlsx --force
	echo "huhanpeng fp32: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model ${MODEL} --save_names fp32 --platform mx

	rm $TRACE_PATH
	TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=True BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} --amp $@
	python3 ${BPF_PATH} --option collect --path ${RST_DIR} --platform MXNET --sub_option xlsx --force
	echo "huhanpeng fp16: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model ${MODEL} --save_names fp16 --platform mx
}


### Run with different batch size
for(( id=1; id <= 128; id*=2 ))
do
	if [ "$1" == "tf" ]; then
		CMD="--batch_size $id --total_batch_size_eval $id ${ARGS_SUFFIX}"
	elif [ "$1" == "mx" ]; then
		CMD="--total_batch_size $id --total_batch_size_eval $id ${ARGS_SUFFIX}"
	fi 
	if [ ${id} = 1 ]; then
		funcRunAndTestFirst ${CMD}
	else
		funcRunAndTest ${CMD}
	fi
done

for(( id=256; id <= 1024; id+=128 ))
do
	if [ "$1" == "tf" ]; then
		CMD="--batch_size $id --total_batch_size_eval $id ${ARGS_SUFFIX}"
	elif [ "$1" == "mx" ]; then
		CMD="--total_batch_size $id --total_batch_size_eval $id ${ARGS_SUFFIX}"
	fi 
	funcRunAndTest ${CMD}
done



