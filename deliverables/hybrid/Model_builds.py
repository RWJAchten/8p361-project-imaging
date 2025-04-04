
from ViT import *
from MBConv import *
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import Conv2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dense, LayerNormalization
from tensorflow.keras.models import Model
import tensorflow as tf


#============================================ Simple 1 layer vision transformer ==============================
    
def Vision_transformer_single(embed_dim=256, input_shape=(96,96,3), 
                       num_heads=4, dropout_rate=0.1, num_classes=1, feed_forward_factor=2, patch_size=False, MLP_depth=2048):
    """
    Simple vision transformer network with 1 layer. First patches are created (unless patch_size=False), then the patches
    are passed trough the transformer block and finally the output is classified by the MLP.

    Arguments:
        embed_dim: The embedding dimension for the transformer block.
        input_shape: The shape of the input images.
        num_heads: The number of attention heads in the transformer block.
        dropout_rate: The dropout rate for regularization.
        num_classes: The number of output classes for classification.
        feed_forward_factor: The factor by which the feed forward layer is expanded.
        patch_size: The size of the patches to be created. If False, no patches are created.
        MLP_depth: The depth of the MLP head.
    
    Output:
        A Keras model with the specified architecture.
    """

    inputs = Input(shape=input_shape)

    # patch embedding if desired
    if patch_size != False:
        x = patch_embedding(inputs, patch_size, embed_dim)
    else:
        shape = tf.shape(x)
        batch, height, width, channel = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, (batch, height * width, channel)) 

    x = transformer_block(x, embed_dim, num_heads, dropout_rate, feed_forward_factor)

    mlp_head = Multi_Layer_Perceptron(num_classes, depth=MLP_depth)
    
    outputs = mlp_head(x)

    return Model(inputs, outputs)

#============================================ Simple 2 layer CNN ===================================

def CNN_double(input_shape=(96,96,3), output_dim1=(48,48,8), output_dim2=(24,24,16), expansion_factor=4, num_classes=1):
    """
    CNN model with two MBConv blocks.

    Arguments:
        input_shape: The shape of the input images.
        output_dim1: The output dimensions of the first MBConv block.
        output_dim2: The output dimensions of the second MBConv block.
        expansion_factor: The expansion factor for the MBConv blocks.
        num_classes: The number of output classes for classification.
    
    Output:
        A Keras model with the specified architecture.
    """

    inputs = Input(shape=input_shape)

    x = inverted_residual_block(inputs, output_dim1, expansion_factor)
   
    x = inverted_residual_block(x, output_dim2, expansion_factor)

    # output layer
    x=Conv2D(1, (1,1), activation='sigmoid')(x)
    outputs=GlobalAveragePooling2D()(x)

    return Model(inputs,outputs)

#============================================ 2 layer ViT ==========================================

def Vision_transformer_double(embed_dim1=64, embed_dim2=128, input_shape=(96,96,3), 
                       num_heads=4, dropout_rate=0.2, num_classes=1, feed_forward_factor=2, patch_size=False, MLP_depth=2048):
    """
    Two layer vision transformer. In between transformer blocks the dimensions are reduced by a factor 2, while
    the amount of channels is kept the same.

    Arguments: 
        embed_dim1: The embedding dimension for the first transformer block.
        embed_dim2: The embedding dimension for the second transformer block.
        input_shape: The shape of the input images.
        num_heads: The number of attention heads in the transformer blocks.
        dropout_rate: The dropout rate for regularization.
        num_classes: The number of output classes for classification.
        feed_forward_factor: The factor by which the feed forward layer is expanded.
        patch_size: The size of the patches to be created. If False, no patches are created.
        MLP_depth: The depth of the MLP head.
    
    Output:
        A Keras model with the specified architecture.
    """

    inputs = Input(shape=input_shape)

    # patch embedding if desired
    if patch_size != False:
        x = patch_embedding(inputs, patch_size, embed_dim1)
    else:
        shape = tf.shape(x)
        batch, height, width, channel = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, (batch, height * width, channel)) 

    x = transformer_block(x, embed_dim1, num_heads, dropout_rate, feed_forward_factor)

    # Convolution in between layers to reduce dimensions by factor 2
    x = Conv1D(filters=embed_dim1, kernel_size=3, strides=2, padding='same')(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Project embedding dimension to match embed_dim2
    x = Dense(embed_dim2)(x)

    # Second transformer block
    x = transformer_block(x, embed_dim2, num_heads, dropout_rate, feed_forward_factor)


    mlp_head = Multi_Layer_Perceptron(num_classes, depth=MLP_depth)
    
    outputs = mlp_head(x)

    return Model(inputs, outputs)

# ============================================= 1 MBConv 1 ViT Hybrid ==========================================

def Hybrid_single(input_shape=(96,96,3),
        output_dim_cnn=(48,48,8), expansion_factor=4,
        embed_dim=64, num_heads=4, dropout_rate=0.2, 
        num_classes=1, feed_forward_factor=2, patch_size=False, MLP_depth=2048):
    """
    Hybrid consisting of MBConv block and ViT block (classified by MLP).

    Arguments:
        input_shape: The shape of the input images.
        output_dim_cnn: The output dimensions of the MBConv block.
        expansion_factor: The expansion factor for the MBConv block.
        embed_dim: The embedding dimension for the transformer block.
        num_heads: The number of attention heads in the transformer block.
        dropout_rate: The dropout rate for regularization.
        num_classes: The number of output classes for classification.
        feed_forward_factor: The factor by which the feed forward layer is expanded.
        patch_size: The size of the patches to be created. If False, no patches are created.
        MLP_depth: The depth of the MLP head.

    Output:
        A Keras model with the specified architecture.
    """

    inputs = Input(shape=input_shape)

    x = inverted_residual_block(inputs, output_dim_cnn, expansion_factor)

    # patch embedding if desired
    if patch_size != False:
        x = patch_embedding(x, patch_size, embed_dim)
    else:
        shape = tf.shape(x)
        batch, height, width, channel = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, (batch, height * width, channel)) 

    x = transformer_block(x, embed_dim, num_heads, dropout_rate, feed_forward_factor)

    mlp_head = Multi_Layer_Perceptron(num_classes, depth=MLP_depth)
    
    outputs = mlp_head(x)

    return Model(inputs, outputs)

# ============================================= 2 MBConv 2 ViT Hybrid ==========================================

def Hybrid_double(input_shape=(96,96,3),
        output_dim_cnn1=(48,48,8), output_dim_cnn2=(24,24,16), expansion_factor=4,
        embed_dim1=64, embed_dim2=128, num_heads=4, dropout_rate=0.2, 
        num_classes=1, feed_forward_factor=2, patch_size=False, MLP_depth=2048):
    """
    Hybrid consisting of two MBConv blocks, followed by two transformer blocks.

    Arguments:
        input_shape: The shape of the input images.
        output_dim_cnn1: The output dimensions of the first MBConv block.
        output_dim_cnn2: The output dimensions of the second MBConv block.
        expansion_factor: The expansion factor for the MBConv blocks.
        embed_dim1: The embedding dimension for the first transformer block.
        embed_dim2: The embedding dimension for the second transformer block.
        num_heads: The number of attention heads in the transformer blocks.
        dropout_rate: The dropout rate for regularization.
        num_classes: The number of output classes for classification.
        feed_forward_factor: The factor by which the feed forward layer is expanded.
        patch_size: The size of the patches to be created. If False, no patches are created.
        MLP_depth: The depth of the MLP head.

    Output:
        A Keras model with the specified architecture.
    """

    inputs = Input(shape=input_shape)

    x = inverted_residual_block(inputs, output_dim_cnn1, expansion_factor)

    x = inverted_residual_block(x, output_dim_cnn2, expansion_factor)

    # patch embedding if desired
    if patch_size != False:
        x = patch_embedding(x, patch_size, embed_dim1)
    else:
        shape = tf.shape(x)
        batch, height, width, channel = shape[0], shape[1], shape[2], shape[3]
        x = tf.reshape(x, (batch, height * width, channel)) 

    x = transformer_block(x, embed_dim1, num_heads, dropout_rate, feed_forward_factor)

    # Convolution in between layers to reduce dimensions by factor 2
    x = Conv1D(filters=embed_dim1, kernel_size=3, strides=2, padding='same')(x)
    x = LayerNormalization(epsilon=1e-6)(x)

    # Project embedding dimension to match embed_dim2
    x = Dense(embed_dim2)(x)

    x = transformer_block(x, embed_dim2, num_heads, dropout_rate, feed_forward_factor)

    mlp_head = Multi_Layer_Perceptron(num_classes, depth=MLP_depth)
    
    outputs = mlp_head(x)

    return Model(inputs, outputs)