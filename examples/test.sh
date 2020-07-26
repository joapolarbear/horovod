#!/bin/bash

function funcRunAndTest() {
	python3 tensorflow_mnist.py $@
	echo "huhanpeng fp32: $@"
	python3 collect_tf.py /root/traces/0/temp.json mnist

	python3 tensorflow_mnist.py $@ --amp
	echo "huhanpeng fp16: $@"
	python3 collect_tf.py /root/traces/0/temp.json mnist
}

# function funcRunAndTest {
# 	BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=100 python3 tensorflow_synthetic_benchmark.py $@
# 	echo "huhanpeng fp32: $@"
# 	python3 collect_tf.py /root/traces/0/temp.json resnet

# 	BYTEPS_TRACE_START_STEP=50 BYTEPS_TRACE_END_STEP=100 python3 tensorflow_synthetic_benchmark.py $@ --amp
# 	echo "huhanpeng fp16: $@"
# 	python3 collect_tf.py /root/traces/0/temp.json resnet
# }


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

for(( id=3; id < 28; id+=2 ))
do
	CMD="--batch_size 1000 --dense_size 1024 --kernel_size $id"
	funcRunAndTest ${CMD}
done

for(( id=128; id < 8193; id*=2 ))
do
	CMD="--batch_size 1000 --dense_size $id --kernel_size 5"
	funcRunAndTest ${CMD}
done


