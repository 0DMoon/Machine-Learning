from __future__ import print_function
import tensorflow.python.keras
import numpy as np
from tensorflow.python.keras.datasets import mnist
from tensorflow.python.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.python.keras.layers import AveragePooling2D, Input, GlobalAveragePooling2D
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers.core import Lambda

from imblearn.over_sampling import BorderlineSMOTE
from data_utils import *

K.set_learning_phase(1)

x_train,y_train=get_data(train=True,corrupt=True)
x_test,y_test=get_data(train=False)
sm=BorderlineSMOTE(random_state=42,kind="borderline-1")
x_train,y_train=sm.fit_resample(x_train,y_train)
# print(x_test.shape)
input_shape=(1,1,x_train.shape[1])
x_train=x_train.reshape(x_train.shape[0],1,1,x_train.shape[1])
# y_train=y_train.reshape(1,1,len(y_train),len(y_train[0]))
x_test=x_test.reshape(x_test.shape[0],1,1,x_test.shape[1])
# y_test=y_test.reshape(1,1,len(y_test),len(y_test[0]))

def abs_backend(inputs):
    return K.abs(inputs)


def expand_dim_backend(inputs):
    return K.expand_dims(K.expand_dims(inputs,1),1)

def sign_backend(inputs):
    return K.sign(inputs)

def pad_backend(inputs, in_channels, out_channels):
    pad_dim = (out_channels - in_channels)//2
    inputs = K.expand_dims(inputs,-1)
    inputs = K.spatial_3d_padding(inputs, ((0,0),(0,0),(pad_dim,pad_dim)), 'channels_last')
    return K.squeeze(inputs, -1)

def residual_shrinkage_block(incoming, nb_blocks, out_channels, downsample=False,
                             downsample_strides=2):
    
    residual = incoming
    in_channels = incoming.get_shape().as_list()[-1]
    
    for i in range(nb_blocks):
        
        identity = residual
        
        if not downsample:
            downsample_strides = 1
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, strides=(downsample_strides, downsample_strides), 
                          padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        residual = BatchNormalization()(residual)
        residual = Activation('relu')(residual)
        residual = Conv2D(out_channels, 3, padding='same', kernel_initializer='he_normal', 
                          kernel_regularizer=l2(1e-4))(residual)
        
        # Calculate global means
        residual_abs = Lambda(abs_backend)(residual)
        abs_mean = GlobalAveragePooling2D()(residual_abs)
        
        # Calculate scaling coefficients
        scales = Dense(out_channels, activation=None, kernel_initializer='he_normal', 
                       kernel_regularizer=l2(1e-4))(abs_mean)
        scales = BatchNormalization()(scales)
        scales = Activation('relu')(scales)
        scales = Dense(out_channels, activation='sigmoid', kernel_regularizer=l2(1e-4))(scales)
        scales = Lambda(expand_dim_backend)(scales)
        
        # Calculate thresholds
        thres = tensorflow.keras.layers.multiply([abs_mean, scales])
        
        # Soft thresholding
        sub = tensorflow.keras.layers.subtract([residual_abs, thres])
        zeros = tensorflow.keras.layers.subtract([sub, sub])
        n_sub = tensorflow.keras.layers.maximum([sub, zeros])
        residual = tensorflow.keras.layers.multiply([Lambda(sign_backend)(residual), n_sub])
        
        # Downsampling using the pooL-size of (1, 1)
        if downsample_strides > 1:
            identity = AveragePooling2D(pool_size=(1,1), strides=(2,2))(identity)
            
        # Zero_padding to match channels
        if in_channels != out_channels:
            identity = Lambda(pad_backend, arguments={'in_channels':in_channels,'out_channels':out_channels})(identity)
        
        residual = tensorflow.keras.layers.add([residual, identity])
    
    return residual


# define and train a model
inputs = Input(shape=input_shape)
net = Conv2D(8, 3, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(inputs)
net = residual_shrinkage_block(net, 1, 8, downsample=True)
net = BatchNormalization()(net)
net = Activation('relu')(net)
net = GlobalAveragePooling2D()(net)
outputs = Dense(units=10, activation='softmax', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))(net)
model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])
# print(y_train.shape)
model.fit(x_train, y_train, batch_size=100, epochs=5, verbose=1, validation_data=(x_test, y_test))

# get results
K.set_learning_phase(0)
DRSN_train_score = model.evaluate(x_train, y_train, batch_size=100, verbose=0)
print('Train loss:', DRSN_train_score[0])
print('Train accuracy:', DRSN_train_score[1])
DRSN_test_score = model.evaluate(x_test, y_test, batch_size=100, verbose=0)
print('Test loss:', DRSN_test_score[0])
print('Test accuracy:', DRSN_test_score[1])
model.summary()