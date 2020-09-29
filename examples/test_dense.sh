#!/bin/bash

### MXNet env
MXNET_CUDNN_AUTOTUNE_DEFAULT=0
MXNET_GPU_WORKER_NTHREADS=1
MXNET_EXEC_BULK_EXEC_TRAIN=0

### Profiling env
BYTEPS_TRACE_ON=1 
BYTEPS_TRACE_DIR=/root/traces/host0

### Model info
DENSE_LAYERS=256x5,512x5,1024x5,1536x5,2048x5,2560x5,3072x5,3584x5,4096x5,3000x5


EXMP_PATH=/root/horovod_examples
RST_DIR=/root/traces
BPF_PATH=${HOME}/byteprofile-analysis/analyze.py
if [ "$1" == "tf" ]; then
	exit
	TRACE_PATH=${BYTEPS_TRACE_DIR}/0/temp.json
	PYTHON_FILE=${HOME}/models/official/resnet/imagenet_main.py
	CLCT_SUFFIX="--platform tf --combine_method minus"
	BPF_CMD=""
elif [ "$1" == "mx" ]; then
	TRACE_PATH=${RST_DIR}/bps_trace_final.json
	PYTHON_FILE=${EXMP_PATH}/mxnet_dense.py
	CLCT_SUFFIX="--platform mx --combine_method minus"
	BPF_CMD="python3 ${BPF_PATH} --option collect --path ${RST_DIR} --platform MXNET --sub_option xlsx --force"
fi

### Start to train
if [ ! -d ${RST_DIR} ]; then
	mkdir -p ${RST_DIR}
else
	rm -rf ${RST_DIR}/*
fi

function funcRunAndTest {
	rm $TRACE_PATH
	BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} $@
	${BPF_CMD}
	echo "huhanpeng fp32: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model dense ${CLCT_SUFFIX}

	rm $TRACE_PATH
	TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=True BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} --amp $@
	${BPF_CMD}
	echo "huhanpeng fp16: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model dense ${CLCT_SUFFIX}
}

function funcRunAndTestFirst {
	rm $TRACE_PATH
	BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} $@
	${BPF_CMD}
	echo "huhanpeng fp32: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model dense --save_names fp32 ${CLCT_SUFFIX}

	rm $TRACE_PATH
	TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=True BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} --amp $@
	${BPF_CMD}
	echo "huhanpeng fp16: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model dense --save_names fp16 ${CLCT_SUFFIX}
}


### Run with different batch size
for(( id=1; id <= 128; id*=2 ))
do
	if [ "$1" == "tf" ]; then
		CMD="--batch_size $id --use_synthetic_data --num_gpus=1 --max_train_steps=200 --train_epochs=1"
	elif [ "$1" == "mx" ]; then
		CMD="--batch-size $id --log-interval 10 --layers=${DENSE_LAYERS}"
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
		CMD="--batch_size $id --use_synthetic_data --num_gpus=1 --max_train_steps=200 --train_epochs=1"
	elif [ "$1" == "mx" ]; then
		CMD="--batch-size $id --log-interval 10 --layers=${DENSE_LAYERS}"
	fi 
	funcRunAndTest ${CMD}
done

for(( id=2048; id <= 8096; id+=1024 ))
do
	if [ "$1" == "tf" ]; then
		CMD="--batch_size $id --use_synthetic_data --num_gpus=1 --max_train_steps=200 --train_epochs=1"
	elif [ "$1" == "mx" ]; then
		CMD="--batch-size $id --log-interval 10 --layers=${DENSE_LAYERS}"
	fi 
	funcRunAndTest ${CMD}
done


