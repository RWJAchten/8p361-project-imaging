
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model

def inverted_residual_block(input, output_dim, expansion_factor=4):

    batch, height, width, channel = input.shape

    # determine the output dimensions and the stride that is necessary to obtain these.
    outheight,outwidth,outchannel=output_dim
    (strx,stry)=(height//outheight,width//outwidth)

    # expand with expand*amount of channels and reduce dimensions with stride
    m = Conv2D(filters=expansion_factor*outchannel, kernel_size=(3,3), strides=(strx,stry), activation=None, padding='same')(input)

    # perform depthwise convolution
    m = DepthwiseConv2D((3,3), activation=None, padding='same', use_bias=False)(m)

    # squeeze back to initial amount of channels
    output = Conv2D(outchannel, (1,1), activation=None, padding='same', use_bias=False)(m)
    output = BatchNormalization()(output)

    output = tf.nn.gelu(output)  

    return output

