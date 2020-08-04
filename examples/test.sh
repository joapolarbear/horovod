#!/bin/bash

TRACE_PATH=/root/traces/0/temp.json
RST_DIR=/root/traces/

# function funcRunAndTest() {
# 	python3 tensorflow_mnist.py $@
# 	echo "huhanpeng fp32: $@"
# 	python3 collect_tf.py ${TRACE_PATH} mnist

# 	python3 tensorflow_mnist.py $@ --amp
# 	echo "huhanpeng fp16: $@"
# 	python3 collect_tf.py ${TRACE_PATH} mnist
# }
rm -rf ${RST_DIR}/*
function funcRunAndTest {
	BYTEPS_TRACE_START_STEP=40 BYTEPS_TRACE_END_STEP=90 python3 /root/horovod_examples/ResNet-Tensorflow/main.py $@
	echo "huhanpeng fp32: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model resnet

	TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=True BYTEPS_TRACE_START_STEP=40 BYTEPS_TRACE_END_STEP=90 python3 /root/horovod_examples/ResNet-Tensorflow/main.py --amp $@
	echo "huhanpeng fp16: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model resnet
}

function funcRunAndTestFirst {
	BYTEPS_TRACE_START_STEP=40 BYTEPS_TRACE_END_STEP=90 python3 /root/horovod_examples/ResNet-Tensorflow/main.py $@
	echo "huhanpeng fp32: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model resnet --save_names fp32

	TF_AUTO_MIXED_PRECISION_GRAPH_REWRITE_IGNORE_PERFORMANCE=True BYTEPS_TRACE_START_STEP=40 BYTEPS_TRACE_END_STEP=90 python3 /root/horovod_examples/ResNet-Tensorflow/main.py --amp $@
	echo "huhanpeng fp16: $@" >> ${RST_DIR}/avg.txt
	python3 collect_tf.py --trace_path ${TRACE_PATH} --rst_dir ${RST_DIR} --model resnet --save_names fp16
}

# for(( id=100; id < 1000; id+=100 ))
# do
# 	CMD="--batch_size $id --dense_size 1024 --kernel_size 5"
# 	funcRunAndTest ${CMD}
# done

# for(( id=1000; id < 10000; id+=1000 ))
# do
# 	CMD="--batch_size $id --dense_size 1024 --kernel_size 5"
# 	funcRunAndTest ${CMD}
# done

for(( id=1; id <= 128; id*=2 ))
do
	CMD="--batch_size $id --iteration 100"
	if [ ${id} = 1 ]; then
		funcRunAndTestFirst ${CMD}
	else
		funcRunAndTest ${CMD}
	fi
done

for(( id=256; id <= 1024; id+=128 ))
do
	CMD="--batch_size $id --iteration 100"
	funcRunAndTest ${CMD}
done

