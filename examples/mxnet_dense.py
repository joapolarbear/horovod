# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import logging
import math
import os
import time

# from gluoncv.model_zoo import get_model
from mxnet.gluon.model_zoo.vision import get_model
from mxnet.contrib import amp
import horovod.mxnet as hvd
import mxnet as mx
import numpy as np
from mxnet import autograd, gluon, lr_scheduler
from mxnet.io import DataBatch, DataIter


# Training settings
parser = argparse.ArgumentParser(description='MXNet ImageNet Example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data-nthreads', type=int, default=2,
                    help='number of threads for data decoding (default: 2)')
parser.add_argument('--batch-size', type=int, default=128,
                    help='training batch size per device (default: 128)')
parser.add_argument('--dtype', type=str, default='float32',
                    help='data type for training (default: float32)')
parser.add_argument('--num-epochs', type=int, default=90,
                    help='number of training epochs (default: 90)')
parser.add_argument('--lr', type=float, default=0.05,
                    help='learning rate for a single GPU (default: 0.05)')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='momentum value for optimizer (default: 0.9)')
parser.add_argument('--wd', type=float, default=0.0001,
                    help='weight decay rate (default: 0.0001)')
parser.add_argument('--lr-mode', type=str, default='poly',
                    help='learning rate scheduler mode. Options are step, \
                    poly and cosine (default: poly)')
parser.add_argument('--lr-decay', type=float, default=0.1,
                    help='decay rate of learning rate (default: 0.1)')
parser.add_argument('--lr-decay-epoch', type=str, default='40,60',
                    help='epoches at which learning rate decays (default: 40,60)')
parser.add_argument('--warmup-lr', type=float, default=0.0,
                    help='starting warmup learning rate (default: 0.0)')
parser.add_argument('--warmup-epochs', type=int, default=10,
                    help='number of warmup epochs (default: 10)')
parser.add_argument('--last-gamma', action='store_true', default=False,
                    help='whether to init gamma of the last BN layer in \
                    each bottleneck to 0 (default: False)')
parser.add_argument('--use-pretrained', action='store_true', default=False,
                    help='load pretrained model weights (default: False)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training (default: False)')
parser.add_argument('--log-interval', type=int, default=0,
                    help='number of batches to wait before logging (default: 0)')
parser.add_argument('--max-iter', type=int, default=200,
                    help='The maximum iteration number allowed to run')
parser.add_argument('--amp', action='store_true', default=False,
                    help='Use fp16 to train the model if set True')

parser.add_argument('--layers', type=str, default=None,
                    help='Use fp16 to train the model if set True')


args = parser.parse_args()

logging.basicConfig(level=logging.INFO)
logging.info(args)


if args.amp:
    amp.init()

# Horovod: initialize Horovod
hvd.init()
num_workers = hvd.size()
rank = hvd.rank()
local_rank = hvd.local_rank()

num_classes = 1000
num_training_samples = 1281167
batch_size = args.batch_size
epoch_size = \
    int(math.ceil(int(num_training_samples // num_workers) / batch_size))

if args.lr_mode == 'step':
    lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')]
    steps = [epoch_size * x for x in lr_decay_epoch]
    lr_sched = lr_scheduler.MultiFactorScheduler(
        step=steps,
        factor=args.lr_decay,
        base_lr=(args.lr * num_workers),
        warmup_steps=(args.warmup_epochs * epoch_size),
        warmup_begin_lr=args.warmup_lr
    )
elif args.lr_mode == 'poly':
    lr_sched = lr_scheduler.PolyScheduler(
        args.num_epochs * epoch_size,
        base_lr=(args.lr * num_workers),
        pwr=2,
        warmup_steps=(args.warmup_epochs * epoch_size),
        warmup_begin_lr=args.warmup_lr
    )
elif args.lr_mode == 'cosine':
    lr_sched = lr_scheduler.CosineScheduler(
        args.num_epochs * epoch_size,
        base_lr=(args.lr * num_workers),
        warmup_steps=(args.warmup_epochs * epoch_size),
        warmup_begin_lr=args.warmup_lr
    )
else:
    raise ValueError('Invalid lr mode')

# Return data and label from batch data
def get_data_label(batch, ctx):
    data = batch.data[0].as_in_context(ctx)
    label = batch.label[0].as_in_context(ctx)
    return data, label


# Nets
class DenseModel(mx.gluon.nn.HybridBlock):
    r"""dense
    Parameters
    ----------
    layers : list of int
        Numbers of units in each layer
    classes : int, default 1000
        Number of classification classes.
    """
    def __init__(self, layers, classes=1000, **kwargs):
        super(DenseModel, self).__init__(**kwargs)
        with self.name_scope():
            self.features = mx.gluon.nn.HybridSequential(prefix='')
            for units in layers:
                self.features.add(mx.gluon.nn.Dense(units))
            self.output = mx.gluon.nn.Dense(classes)

    def hybrid_forward(self, F, x):
        x = self.features(x)
        x = self.output(x)
        return x


# Create data iterator for synthetic data
class SyntheticDataIter(DataIter):
    def __init__(self, num_classes, data_shape, max_iter, dtype, ctx):
        self.batch_size = data_shape[0]
        self.cur_iter = 0
        self.max_iter = max_iter
        self.dtype = dtype
        label = np.random.randint(0, num_classes, [self.batch_size, ])
        data = np.random.uniform(-1, 1, data_shape)
        self.data = mx.nd.array(data, dtype=self.dtype,
                                ctx=ctx)
        self.label = mx.nd.array(label, dtype=self.dtype,
                                 ctx=ctx)
    def __iter__(self):
        return self
    @property
    def provide_data(self):
        return [mx.io.DataDesc('data', self.data.shape, self.dtype)]
    @property
    def provide_label(self):
        return [mx.io.DataDesc('softmax_label',
                               (self.batch_size,), self.dtype)]
    def next(self):
        self.cur_iter += 1
        if self.cur_iter <= self.max_iter:
            return DataBatch(data=(self.data,),
                             label=(self.label,),
                             pad=0,
                             index=None,
                             provide_data=self.provide_data,
                             provide_label=self.provide_label)
        else:
            raise StopIteration
    def __next__(self):
        return self.next()
    def reset(self):
        self.cur_iter = 0


# Horovod: pin GPU to local rank
context = mx.cpu(local_rank) if args.no_cuda else mx.gpu(local_rank)


# Use synthetic data
data_shape = (batch_size, 128)
train_data = SyntheticDataIter(num_classes, data_shape, epoch_size,
                               np.float32, context)
val_data = None


# Get model from GluonCV model zoo
# https://gluon-cv.mxnet.io/model_zoo/index.html
kwargs = {'ctx': context,
          'pretrained': args.use_pretrained,
          'classes': num_classes}
if args.last_gamma:
    kwargs['last_gamma'] = True

if args.layers is None:
    layers = [128, 256, 512, 1024]
else:
    layers = []
    for layernum in args.layers.split(','):
        spliter = None
        if 'x' in layernum:
            spliter = 'x'
        elif '*' in layernum:
            spliter = '*'
        if spliter is not None:
            tmp = layernum.split(spliter)
            num, cnt = int(tmp[0]), int(tmp[1])
        else:
            cnt = 1
            num = int(layernum)
        for _ in range(cnt):
            layers.append(num)
net = DenseModel(layers, classes=num_classes)
net.cast(args.dtype)

# Create initializer
initializer = mx.init.Xavier(rnd_type='gaussian', factor_type="in",
                             magnitude=2)

def train_gluon():
    # Hybridize and initialize model
    net.hybridize()
    net.initialize(initializer, ctx=context)

    # Horovod: fetch and broadcast parameters
    params = net.collect_params()
    if params is not None:
        hvd.broadcast_parameters(params, root_rank=0)

    # Create optimizer
    optimizer_params = {'wd': args.wd,
                        'momentum': args.momentum,
                        'lr_scheduler': lr_sched}
    if args.dtype == 'float16':
        optimizer_params['multi_precision'] = True
    opt = mx.optimizer.create('sgd', **optimizer_params)

    # Horovod: create DistributedTrainer, a subclass of gluon.Trainer
    trainer = hvd.DistributedTrainer(params, opt, block=net, data_shape=[data_shape])

    if args.amp:
        amp.init_trainer(trainer)

    # Create loss function and train metric
    loss_fn = gluon.loss.SoftmaxCrossEntropyLoss()
    metric = mx.metric.Accuracy()

    # Train model
    for epoch in range(args.num_epochs):
        if epoch * epoch_size > args.max_iter:
            break
        tic = time.time()
        metric.reset()

        btic = time.time()
        for nbatch, batch in enumerate(train_data, start=1):
            if nbatch + epoch * epoch_size > args.max_iter:
                break
            data, label = get_data_label(batch, context)
            with autograd.record():
                output = net(data.astype(args.dtype, copy=False))
                loss = loss_fn(output, label)
            loss.backward()
            trainer.step(batch_size)

            metric.update([label], [output])
            if args.log_interval and nbatch % args.log_interval == 0:
                name, acc = metric.get()
                logging.info('Epoch[%d] Rank[%d] Batch[%d]\t%s=%f\tlr=%f',
                             epoch, rank, nbatch, name, acc, trainer.learning_rate)
                if rank == 0:
                    batch_speed = num_workers * batch_size * args.log_interval / (time.time() - btic)
                    logging.info('Epoch[%d] Batch[%d]\tSpeed: %.2f samples/sec',
                                 epoch, nbatch, batch_speed)
                btic = time.time()

        # Report metrics
        elapsed = time.time() - tic
        _, acc = metric.get()
        logging.info('Epoch[%d] Rank[%d] Batch[%d]\tTime cost=%.2f\tTrain-accuracy=%f',
                     epoch, rank, nbatch, elapsed, acc)
        if rank == 0:
            epoch_speed = num_workers * batch_size * nbatch / elapsed
            logging.info('Epoch[%d]\tSpeed: %.2f samples/sec', epoch, epoch_speed)

if __name__ == '__main__':
    train_gluon()
