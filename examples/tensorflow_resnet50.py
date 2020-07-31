import numpy as np
import tensorflow as tf
from tensorflow import keras
import os, sys
import argparse
import horovod.tensorflow as hvd

parser = argparse.ArgumentParser(description='Tensorflow MNIST Example')
parser.add_argument('--tensorboard', action='store_true', default=False,
                    help='output summary for tensorboard visualization')
parser.add_argument('--amp', action='store_true', default=False,
                    help='use automatically mixed precision')
args = parser.parse_args()


from google.protobuf.json_format import MessageToJson
from tensorflow.python.client import timeline
import json
import networkx as nx
class TimelineSession:
    def __init__(self, sess):
        self.sess = sess
        self.graph = sess.graph
        self.step_cnt = 0

        self.trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(hvd.local_rank()))
        if not os.path.exists(self.trace_dir):
            os.makedirs(self.trace_dir)
        if os.environ.get("BYTEPS_TRACE_ON", "") != '1':
            self._end_trace = True
            return
        self._end_trace = False
        self.end_step = int(os.environ.get("BYTEPS_TRACE_END_STEP", "30"))
        self.start_step = int(os.environ.get("BYTEPS_TRACE_START_STEP", "20"))

        if not self._end_trace and self.start_step < 1:
            raise ValueError("BYTEPS_TRACE_START_STEP must be larger than 1")
        if not self._end_trace and self.end_step <= self.start_step:
            raise ValueError("BYTEPS_TRACE_END_STEP must be larger than BYTEPS_TRACE_START_STEP")   

        ### Timeline configuratoin
        self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        self.run_metadata = tf.RunMetadata()
        self.traces = {"traceEvents":[]}

        self.dag = None

    def run(self, *args_, **kwargs_):
        if self._end_trace:
            ret = self.sess.run(*args_, **kwargs_)
        elif not self._end_trace and self.step_cnt < self.start_step:
            ret = self.sess.run(*args_, **kwargs_)
            self.step_cnt += 1
        elif not self._end_trace and self.step_cnt < self.end_step:
            ret = self.sess.run(*args_, options=self.run_options, run_metadata=self.run_metadata, **kwargs_)
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

        ### Return all fetches
        return ret

    
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

    def should_stop(self):
        return self.sess.should_stop()

hvd.init()
sess = TimelineSession(tf.Session())
from keras import backend as K
K.set_session(sess)

# Load Cifar-10 data-set
(train_im, train_lab), (test_im, test_lab) = tf.keras.datasets.cifar10.load_data()

#### Normalize the images to pixel values (0, 1)
train_im, test_im = train_im/255.0 , test_im/255.0
#### Check the format of the data 
print ("train_im, train_lab types: ", type(train_im), type(train_lab))
#### check the shape of the data
print ("shape of images and labels array: ", train_im.shape, train_lab.shape) 
print ("shape of images and labels array ; test: ", test_im.shape, test_lab.shape)


#### Check the distribution of unique elements 
(unique, counts) = np.unique(train_lab, return_counts=True)
frequencies = np.asarray((unique, counts)).T
print (frequencies)
print (len(unique))


class_types = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck'] # from cifar-10 website

### One hot encoding for labels 
train_lab_categorical = tf.keras.utils.to_categorical(
    train_lab, num_classes=10, dtype='uint8')

test_lab_categorical = tf.keras.utils.to_categorical(
    test_lab, num_classes=10, dtype='uint8')

### Train -test split 
from sklearn.model_selection import train_test_split 
train_im, valid_im, train_lab, valid_lab = train_test_split(train_im, train_lab_categorical, test_size=0.20, 
                                                            stratify=train_lab_categorical, 
                                                            random_state=40, shuffle = True)

print ("train data shape after the split: ", train_im.shape)
print ('new validation data shape: ', valid_im.shape)
print ("validation labels shape: ", valid_lab.shape)


from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, ZeroPadding2D,\
     Flatten, BatchNormalization, AveragePooling2D, Dense, Activation, Add 
from tensorflow.keras.models import Model
from tensorflow.keras import activations
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2

##### Include Little Data Augmentation 
batch_size = 64 # try several values

train_DataGen = tf.keras.preprocessing.image.ImageDataGenerator(zoom_range=0.2, 
                                                                width_shift_range=0.1, 
                                                                height_shift_range = 0.1, 
                                                                horizontal_flip=True)
 
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator()

train_set_conv = train_DataGen.flow(train_im, train_lab, batch_size=batch_size) # train_lab is categorical 
valid_set_conv = valid_datagen.flow(valid_im, valid_lab, batch_size=batch_size) # so as valid_lab

def half_precision(layer_f, input_, *args_, **kwargs_):
    if args.amp:
        input_fp16 = tf.keras.backend.cast(input_, dtype="float16")
        output_fp16 = layer_f(input_fp16, *args_, **kwargs_)
        output_fp32 = tf.keras.backend.cast(output_fp16, dtype="float32")
    else:
        output_fp32 = layer_f(input_, *args_, **kwargs_)
    return output_fp32

def single_precision(layer_f, input_, *args_, **kwargs_):
    output_fp32 = layer_f(input_, *args_, **kwargs_)
    return output_fp32

def res_identity(x, filters): 
  #renet block where dimension doesnot change.
  #The skip connection is just simple identity conncection
  #we will have 3 blocks and then input will be added

  x_skip = x # this will be used for addition with the residual block 
  f1, f2 = filters

  #first block 
  x = Conv2D(f1, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #second block # bottleneck (but size kept same with padding)
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  # third block activation used after adding the input
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  # x = Activation(activations.relu)(x)

  # add the input 
  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x


def res_conv(x, s, filters):
  '''
  here the input size changes''' 
  x_skip = x
  f1, f2 = filters

  # first block
  x = Conv2D(f1, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x)
  # when s = 2 then it is like downsizing the feature map
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  # second block
  x = Conv2D(f1, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)

  #third block
  x = Conv2D(f2, kernel_size=(1, 1), strides=(1, 1), padding='valid', kernel_regularizer=l2(0.001))(x)
  x = BatchNormalization()(x)

  # shortcut 
  x_skip = Conv2D(f2, kernel_size=(1, 1), strides=(s, s), padding='valid', kernel_regularizer=l2(0.001))(x_skip)
  x_skip = BatchNormalization()(x_skip)

  # add 
  x = Add()([x, x_skip])
  x = Activation(activations.relu)(x)

  return x


def resnet50():

  input_im = Input(shape=(train_im.shape[1], train_im.shape[2], train_im.shape[3])) # cifar 10 images size
  x = ZeroPadding2D(padding=(3, 3))(input_im)

  # 1st stage
  # here we perform maxpooling, see the figure above

  x = single_precision(Conv2D, 64, kernel_size=(7, 7), strides=(2, 2))(x)
  x = BatchNormalization()(x)
  x = Activation(activations.relu)(x)
  x = MaxPooling2D((3, 3), strides=(2, 2))(x)

  #2nd stage 
  # frm here on only conv block and identity block, no pooling

  x = single_precision(res_conv, x, s=1, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))
  x = res_identity(x, filters=(64, 256))

  # 3rd stage

  x = single_precision(res_conv, x, s=2, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))
  x = res_identity(x, filters=(128, 512))

  # 4th stage

  x = single_precision(res_conv, x, s=2, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))
  x = res_identity(x, filters=(256, 1024))

  # 5th stage

  x = single_precision(res_conv, x, s=2, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))
  x = res_identity(x, filters=(512, 2048))

  # ends with average pooling and dense connection

  x = AveragePooling2D((2, 2), padding='same')(x)

  x = Flatten()(x)
  x = Dense(len(class_types), activation='softmax', kernel_initializer='he_normal')(x) #multi-class

  # define the model 

  model = Model(inputs=input_im, outputs=x, name='Resnet50')

  return model

### Define some Callbacks
def lrdecay(epoch):
    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    #print('Learning rate: ', lr)
    return lr
  # if epoch < 40:
  #   return 0.01
  # else:
  #   return 0.01 * np.math.exp(0.03 * (40 - epoch))
lrdecay = tf.keras.callbacks.LearningRateScheduler(lrdecay) # learning rate decay 
def earlystop(mode):
  if mode=='acc':
    estop = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=15, mode='max')
  elif mode=='loss':
    estop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, mode='min')
  return estop

resnet50_model = resnet50() 
# resnet50_model.summary()

resnet50_model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-3), 
                       metrics=['acc'])


trace_dir = os.path.join(os.environ.get("BYTEPS_TRACE_DIR", "."), str(0))
if not os.path.exists(trace_dir):
  os.makedirs(trace_dir)
logdir = os.path.join(trace_dir, "board")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

batch_size=batch_size # test with 64, 128, 256

resnet_train = resnet50_model.fit(train_set_conv, 
                                  epochs=160, 
                                  steps_per_epoch=train_im.shape[0]/batch_size, 
                                  validation_steps=valid_im.shape[0]/batch_size, 
                                  validation_data=valid_set_conv, 
                                  callbacks=[lrdecay, tensorboard_callback])










