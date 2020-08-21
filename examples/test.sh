#!/bin/bash

### MXNet env
MXNET_CUDNN_AUTOTUNE_DEFAULT=0
BYTEPS_TRACE_ON=1 
BYTEPS_TRACE_DIR=/root/traces

RST_DIR=/root/traces
TRACE_PATH=${RST_DIR}/0/temp.json
CLCT_PATH=/root/horovod_examples
PYTHON_FILE=${HOME}/models/official/resnet/imagenet_main.py

### Start to train
if [ ! -d ${RST_DIR} ]; then
	mkdir ${RST_DIR}
else
	rm -rf ${RST_DIR}/*
fi

function funcRunAndTest {
	rm $TRACE_PATH
	BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} $@
	echo "huhanpeng fp32: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model resnet

	rm $TRACE_PATH
	TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=True BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} --amp $@
	echo "huhanpeng fp16: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model resnet
}

function funcRunAndTestFirst {
	rm $TRACE_PATH
	BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} $@
	echo "huhanpeng fp32: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model resnet --save_names fp32

	rm $TRACE_PATH
	TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=True BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=150 python3 ${PYTHON_FILE} --amp $@
	echo "huhanpeng fp16: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model resnet --save_names fp16
}


for(( id=1; id <= 128; id*=2 ))
do
	CMD="--batch_size $id --use_synthetic_data --num_gpus=1 --max_train_steps=200 --train_epochs=1"
	if [ ${id} = 1 ]; then
		funcRunAndTestFirst ${CMD}
	else
		funcRunAndTest ${CMD}
	fi
done

for(( id=256; id <= 1024; id+=128 ))
do
	CMD="--batch_size $id --use_synthetic_data --num_gpus=1 --max_train_steps=200 --train_epochs=1"
	funcRunAndTest ${CMD}
done



