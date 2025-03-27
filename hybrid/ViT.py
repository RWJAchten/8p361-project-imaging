import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}   

import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LayerNormalization
from tensorflow.keras.layers import MultiHeadAttention


def patch_embedding(x, patch_size, embed_dim):
    """
    Function that transforms the image into a patch embedding space.
    """

    shape = tf.shape(x)
    batch, height, width, channel = shape[0], shape[1], shape[2], shape[3]

    patches = tf.image.extract_patches(
            images=x,
            sizes=[1, patch_size[0], patch_size[1], 1],  # patch size
            strides=[1, patch_size[0], patch_size[1], 1],  # stride size
            rates=[1, 1, 1, 1],
            padding='VALID')
    
    num_patches = (height // patch_size[0]) * (width // patch_size[1])
    patches = tf.reshape(patches, [-1, num_patches, patch_size[0] * patch_size[1] * channel])
    patch_embeddings=Dense(embed_dim)(patches)
    
    return patch_embeddings

def transformer_block(x, embed_dim, num_heads=4, dropout_rate=0.1, feed_forward_factor=2): 
    """
    The transformer block consisting of Multi Head Attention (MHA) and a Feed Forward Network (FFN)
    """

    ff_dim=feed_forward_factor*embed_dim

    # Pre-normalization
    x = LayerNormalization(epsilon=1e-6)(x)
    # Multi-head attention layer
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(x, x)
    attn_output = Dropout(dropout_rate)(attn_output) # dropout to prevent overfitting

    # Post-normalization
    x = LayerNormalization(epsilon=1e-6)(x + attn_output)  
 
    # Feed-forward network 
    x = Dense(ff_dim, activation="gelu")(x) # expands feature dimension and introduces non-linearity (to recognize complex patterns)
    x = Dropout(dropout_rate)(x) # dropout for generalization
  
    x = Dense(embed_dim)(x) # projects back to original size
    ffn_output = Dropout(dropout_rate)(x) # dropout for generalization

    output=LayerNormalization(epsilon=1e-6)(x + ffn_output)

    return output

def Multi_Layer_Perceptron(num_classes=1, depth=2048):
    """ 
    Multi layer perceptron used for classification
    """
    return tf.keras.Sequential([
            LayerNormalization(),
            GlobalAveragePooling1D(),
            Dense(depth, activation='relu'),
            Dense(num_classes, activation='sigmoid')
        ])





