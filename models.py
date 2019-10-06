import tensorflow as tf

keras = tf.keras
K = keras.backend
from tensorflow.keras.layers import *

import numpy as np
from functools import partial
import warnings
warnings.filterwarnings('ignore')

def scaledDotProductAttention(q, k, v, dim_k, bias=None):
    """
    scaled dot product attention from transformer paper
    """
    atts = K.batch_dot(q, k, axes=[2,2])
    atts = atts / np.sqrt(dim_k)
    if not bias is None:
        atts = atts + bias
    atts = K.softmax(atts)
    dpa = K.batch_dot(atts, v, axes=[2, 1])
    return dpa


def multiheadAttention(q, k, v, weights, bias_weights, dim_k, bias=None, heads=1):
    """
    multihead attention from transformer paper
    """
    w_q, w_k, w_v, w_mha = weights
    b_q, b_k, b_v, b_mha = bias_weights
    def _dense(x, w, b):
        return tf.tensordot(x,w,axes=[-1,0]) + b
    
    # initial linear projections
    q = _dense(q, w_q, b_q)
    k = _dense(k, w_k, b_k)
    v = _dense(v, w_v, b_v)
    
    # doing sdpa on each head
    qs = tf.split(q, heads, axis=-1)
    ks = tf.split(k, heads, axis=-1)
    vs = tf.split(v, heads, axis=-1)
    head_outputs = []
    for q, k, v in zip(qs, ks, vs):
        head_outputs.append(scaledDotProductAttention(q, k, v, dim_k, bias))
    mha = tf.concat(head_outputs, -1)
    
    # final linear
    return _dense(mha, w_mha, b_mha)

class MultiheadAttention(keras.layers.Layer):
    """
    multihead attention layer
    """
    def __init__(self, dim, dim_k, dim_v, heads, **kwargs):
        super().__init__(**kwargs)
        self.dim = dim
        self.dim_k = dim_k
        self.dim_v = dim_v
        self.heads = heads
        
    def build(self, input_shapes):
        q_shape, k_shape, v_shape = input_shapes[:3]
        weight_shapes = [[int(q_shape[-1]), self.dim_k*self.heads], 
                         [int(k_shape[-1]), self.dim_k*self.heads],
                         [int(v_shape[-1]), self.dim_v*self.heads],
                         [self.dim_v*self.heads, self.dim]]
        bias_shapes = [[1,1,self.dim_k*self.heads],
                       [1,1,self.dim_k*self.heads],
                       [1,1,self.dim_v*self.heads],
                       [1,1,self.dim]]
        
        self.kernel_weights = [self.add_weight(initializer='glorot_uniform',
                                               shape=ws) for ws in weight_shapes]
        self.bias_weights = [self.add_weight(initializer='zeros', 
                                             shape=bs) for bs in bias_shapes]
        
        self.built = True
        super().build(input_shapes)
        
    def call(self, inputs):
        if len(inputs) == 3:
            q, k, v = inputs
            bias = None
        elif len(inputs) == 4:
            q, k, v, bias = inputs
        else:
            raise ValueError('can only pass 3 or 4 inputs [q,k,v,(bias)]')
        mha = multiheadAttention(q, k, v, self.kernel_weights, self.bias_weights, self.dim_k, bias)
        return mha
    
    def compute_output_shapes(self, input_shapes):
        batch_size = input_shapes[0][0]
        N = input_shapes[0][1]
        return (batch_size, N, self.dim)
            

def encode_block(x, num_heads, dim, bias, block_number):
    """
    single encoder layer from transformer paper
    """
    # mha + add/norm
    mha_in = [x,x,x] if bias is None else [x,x,x,bias]
    _x = MultiheadAttention(dim, dim//2, dim//2, num_heads, name=f'mha_{block_number}')(mha_in)
    x = Add()([x, _x])
    x = LayerNormalization()(x)

    # ffn + add/norm
    _x = Dense(2*dim, activation='relu')(x)
    _x = Dense(dim)(_x)
    x = Add()([x, _x])
    x = LayerNormalization()(x)
    return x
    
def decode_block(x, enc_x, num_heads, dim, bias0, bias1, block_number):
    """
    single decoder layer from transformer paper
    """
    # mha + add/norm
    lt_bias = Lambda(make_temporal_bias)(x)
    if not bias0 is None: lt_bias = Add()([bias0, lt_bias])
    mha_in = [x,x,x,lt_bias]
    _x = MultiheadAttention(dim, dim//2, dim//2, num_heads, name=f'd0_mha_{block_number}')(mha_in)
    x = Add()([x, _x])
    x = LayerNormalization()(x)

    # mha + add/norm with encoder input
    mha_in = [x,enc_x,enc_x] if bias1 is None else [x,enc_x,enc_x,bias1]
    _x = MultiheadAttention(dim, dim//2, dim//2, num_heads, name=f'd1_mha_{block_number}')(mha_in)
    x = Add()([x, _x])
    x = LayerNormalization()(x)

    # ffn + add/norm
    _x = Dense(2*dim, activation='relu')(x)
    _x = Dense(dim)(_x)
    x = Add()([x, _x])
    x = LayerNormalization()(x)
    return x

def pointer(inputs):
    """
    using a pointer-net like head
    """
    pointers, points = inputs
    logits = K.batch_dot(pointers, points, axes=2)
    probits = K.softmax(logits)
    return probits

def make_bias(key, query, meta):
    """
    generates bias based on mask meta
    """
    mask_key = tf.cast(tf.not_equal(key, meta), tf.float32)
    mask_key = tf.expand_dims(mask_key, -1)
    
    mask_query = tf.cast(tf.not_equal(query, meta), tf.float32)
    mask_query = tf.expand_dims(mask_query, -1)
    
    mask = K.batch_dot(mask_query, mask_key, axes=[2, 2])
    mask = 1 - mask
    return -1e9 * mask

def make_temporal_bias(key):
    """
    generates bias based on right temporal masking
    """
    batch_size = tf.shape(key)[0]
    N = tf.shape(key)[1]
    bias =  -1e9 * (1 - tf.matrix_band_part(tf.ones((N, N)), -1, 0))
    bias = tf.tile(tf.expand_dims(bias, 0), (batch_size, 1, 1))
    return bias
    

def pointer_transformer(alphabet_size=28,
                        dim=64,
                        num_heads=6,
                        blocks=4,
                        mask_meta=2):
    """
    pointer-transformer network
    """
    inpt_encoder = Input(shape=(None,), name='inpt_encoder')
    encoder_bias = Lambda(lambda u : make_bias(*u, meta=mask_meta))([inpt_encoder, inpt_encoder])
    
    inpt_decoder = Input(shape=(None,), name='inpt_decoder')
    decoder_bias_0 = Lambda(lambda u : make_bias(*u, meta=mask_meta))([inpt_decoder, inpt_decoder])
    decoder_bias_1 = Lambda(lambda u : make_bias(*u, meta=mask_meta))([inpt_encoder, inpt_decoder])

    

    embedding = Embedding(input_dim=alphabet_size, output_dim=dim, name='embedding')
    ex = embedding(inpt_encoder)
    dx = embedding(inpt_decoder)

    for i in range(blocks):   
        ex = encode_block(ex, num_heads, dim, encoder_bias, i)
        dx = decode_block(dx, ex, num_heads, dim, decoder_bias_0, decoder_bias_1, i)

    ex = Dense(dim, activation='relu')(ex)
    outpt = Lambda(pointer, name='pointer')([dx, ex])
    model = keras.Model([inpt_decoder, inpt_encoder], outpt)
    return model
    
    
    