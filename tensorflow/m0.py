#!/usr/bin/env python3

## sys build-in package
import tensorflow
import os
import numpy as np
import logging
# set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## keras-related package
import keras
from keras import backend as K
from keras.models import Model
from keras.layers import Activation, Input, Conv2D, Conv3D, MaxPooling2D, UpSampling2D, concatenate, BatchNormalization
from keras.layers import add, multiply, LeakyReLU, PReLU, ThresholdedReLU
from keras.initializers import he_normal
from keras.initializers import Constant
from keras.engine.topology import Layer


def m0(model = None, img_row=512, img_col=512, img_slice = 5, img_channel=1, layers = 32, features = 64, 
		act_type ='relu', scale_factor = 0.1, init_seed=15213, batch_norm=False,is_pred=False, prow=512, 
		pcol=512, pslice=1, pchannel=1):

	logging.info("Current version is 3D : LR -> HR")
	logging.info("Scaling factor is " + str(scale_factor))
	logging.info("Activation function " + act_type)

	# Initialize the inputs
	if is_pred == False:
		inputs = Input((img_row, img_col, img_slice, img_channel))
	else:
		inputs = Input((prow, pcol, pslice, pchannel))
	# Perform the first convolution and save it for residuals at the end
	x = Conv3D(features, kernel_size = (3,3,3), 
				kernel_initializer = he_normal(seed=init_seed),
				padding = 'same',
				name = ('conv0'))(inputs)
	conv_1 = x

	# Perform convolutions for the residual residual_block
	for blk_num in range(layers):
		x = residual_block_3d(x, features, act_type, scale_factor, blk_num, init_seed)

	# Perform the penultimate convolution and add to the first conv
	x = Conv3D(features, kernel_size = (3,3,3), 
				kernel_initializer = he_normal(seed=init_seed),
				padding = 'same',
				name = ('conv_penultimate'))(x)
	x = add([x, conv_1])

	# Final conv to create the super-res output
	output = Conv3D(1, kernel_size = (3,3,3), 
				kernel_initializer = he_normal(seed=init_seed),
				padding = 'same',
				name = ('conv_final'))(x)

	model = Model(inputs = [inputs], outputs = [output])

	return model


class ConstMultiplierLayer(Layer):
    def __init__(self, val=0.1, **kwargs):
        self.val = val
        super(ConstMultiplierLayer, self).__init__(**kwargs)


    # Create a function to initialize this layer with a specific value
    def build(self, input_shape):
        self.k = self.add_weight(
            name='k',
            shape=(),
            initializer=Constant(value=self.val),
            dtype='float32',
            trainable=True,
        )
        super(ConstMultiplierLayer, self).build(input_shape)

    def call(self, x):
        return K.tf.multiply(self.k, x)

    def compute_output_shape(self, input_shape):
        return input_shape


def residual_block_3d(x, filter_size, act_type, scale_factor, blk_num, init_seed):
	tmp = Conv3D(filter_size, kernel_size = (3,3,3), kernel_initializer = he_normal(seed=init_seed),
				padding = 'same',
				name = ('resblock_%i_conv_1' % blk_num))(x)
	if act_type =="LeakyReLU":
		tmp = LeakyReLU(alpha=0.2)(tmp)
	else:
		tmp = Activation(act_type, name = ('resblock_%i_act' % blk_num))(tmp)

	tmp = Conv3D(filter_size, kernel_size = (3,3,3), kernel_initializer = he_normal(seed=init_seed),
				padding = 'same',
				name = ('resblock_%i_conv_2' % blk_num))(tmp)
	
	tmp = ConstMultiplierLayer(val = scale_factor)(tmp)
	out = keras.layers.Add()([x, tmp])

	return out



