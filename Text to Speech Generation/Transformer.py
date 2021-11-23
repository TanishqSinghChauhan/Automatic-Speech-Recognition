import tensorflow as tf
from tensorflow import keras
import numpy as np


#########################
# Positional Encoder
#########################

def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])


    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)

#########################
# Mask Calculations
#########################
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, x)


def create_mel_padding_mask(seq):
    seq = tf.reduce_sum(tf.math.abs(seq), axis=-1)
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, x)


def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask
#########################
# Attention
#########################
class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads):
        super(MultiHeadAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "Embedding size needs to be divisible by heads"

        self.wq = tf.keras.layers.Dense(self.head_dim)
        self.wk = tf.keras.layers.Dense(self.head_dim)
        self.wv = tf.keras.layers.Dense(self.head_dim)
        self.dense = tf.keras.layers.Dense(self.embed_size)


    def call(self, v, k, q, mask):
        # Get number of training examples
        batch_size = q.shape[0]

        seq_len_v, seq_len_k, seq_len_q = v.shape[1], k.shape[1], q.shape[1]


        # Split the embedding into self.heads different pieces
        v = tf.reshape(v,(batch_size,seq_len_v, self.heads, self.head_dim))
        k = tf.reshape(k,(batch_size,seq_len_k, self.heads, self.head_dim))
        q = tf.reshape(q,(batch_size,seq_len_q, self.heads, self.head_dim))


        values = self.wv(v)  # (batch_size, value_len, heads, head_dim)
        keys = self.wk(k)  # (batch_size, key_len, heads, head_dim)
        queries = self.wq(q)  # (batch_size, query_len, heads, heads_dim)

        # Einsum does matrix mult. for query*keys for each training example

        attention = tf.einsum("nqhd,nkhd->nhqk", queries, keys)
        # queries shape: (batch_size, seq_len_q, heads, heads_dim),
        # keys shape: (batch_size, seq_len_k, heads, heads_dim)
        # attention: (batch_size, heads, seq_len_q, seq_len_k)

         # scale matmul_qk
        dk = tf.cast(tf.shape(keys)[1], tf.float32)
        scaled_attention_logits = attention / tf.math.sqrt(dk)

        # Mask padded indices so their weights become 0

        if mask is not None:
            scaled_attention_logits += (mask * -1e9)

        # Normalize energy values similarly to seq2seq + attention
        # so that they sum to 1. Also divide by scaling factor for
        # better stability
        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        # attention shape: (batch_size, heads, seq_len_q, seq_len_k)



        out = tf.einsum("nhql,nlhd->nqhd", attention_weights, values)
        out = tf.reshape(out, (batch_size, seq_len_q, -1))
        # attention shape: (batch_size, heads, seq_len_q, key_len)
        # values shape: (batch_size, seq_len_v, heads, heads_dim)
        # out after matrix multiply: (batch_size, seq_len_q, heads, head_dim), then
        # we reshape and flatten the last two dimensions.

        out = self.dense(out)
        # Linear layer doesn't modify the shape, final shape will be
        # (N, query_len, embed_size)

        return out, attention_weights

#########################
# Feed Forward Layer
#########################
def point_wise_feed_forward_network(embed_size, feed_forward):
    return tf.keras.Sequential([
        # (batch_size, seq_len, feed_forward)
        tf.keras.layers.Dense(feed_forward, activation='relu'),
        tf.keras.layers.Dense(embed_size)  # (batch_size, seq_len, embed_size)
    ])

class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, feed_forward, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(embed_size, heads)
        self.ffn = point_wise_feed_forward_network(embed_size, feed_forward)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, x, training, mask):
        # (batch_size, input_seq_len, embed_size)
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        # (batch_size, input_seq_len, embed_size)
        out1 = self.layernorm1(x + attn_output)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, embed_size)
        ffn_output = self.dropout2(ffn_output, training=training)
        # (batch_size, input_seq_len, embed_size)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual connection

        return out2
#########################
# Decoder
#########################

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, embed_size, heads, feed_forward, rate=0.1):
        super(DecoderLayer, self).__init__()

        self.mha1 = MultiHeadAttention(embed_size, heads)
        self.mha2 = MultiHeadAttention(embed_size, heads)

        self.ffn = point_wise_feed_forward_network(embed_size, feed_forward)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.dropout3 = tf.keras.layers.Dropout(rate)

    def call(self, x, enc_output, training,
             look_ahead_mask, padding_mask):
        # enc_output.shape == (batch_size, input_seq_len, embed_size)
        attn1, attn_weights_block1 = self.mha1(
            x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, embed_size)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(attn1 + x) # residual

        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, embed_size)
        attn2 = self.dropout2(attn2, training=training)
        # (batch_size, target_seq_len, embed_size)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, embed_size)
        ffn_output = self.dropout3(ffn_output, training=training)
        # (batch_size, target_seq_len, embed_size)
        out3 = self.layernorm3(ffn_output + out2)

        return out3, attn_weights_block1, attn_weights_block2

class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, embed_size, heads, feed_forward, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.embed_size = embed_size
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, embed_size)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.embed_size)

    self.enc_layers = [EncoderLayer(embed_size, heads, feed_forward, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, embed_size)
    x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]
    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, embed_size)

class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, embed_size, heads, feed_forward, dense_hidden_units,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.embed_size = embed_size
    self.num_layers = num_layers
    self.rate = rate
    self.d1 = tf.keras.layers.Dense(dense_hidden_units,activation='relu')  # (batch_size, seq_len, dense_hidden_units)
    self.d2 = tf.keras.layers.Dense(embed_size, activation='relu')  # (batch_size, seq_len, model_dim)
    self.dropout_1 = tf.keras.layers.Dropout(self.rate)
    self.dropout_2 = tf.keras.layers.Dropout(self.rate)
    
    
    self.pos_encoding = positional_encoding(maximum_position_encoding, embed_size)

    self.dec_layers = [DecoderLayer(embed_size, heads, feed_forward, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.d1(x)
    # use dropout also in inference for positional encoding relevance
    x = self.dropout_1(x, training=training)
    x = self.d2(x)
    x = self.dropout_2(x, training=training)   
    x *= tf.math.sqrt(tf.cast(self.embed_size, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):

      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2
    # x.shape == (batch_size, target_seq_len, embed_size)
    return x, attention_weights


##################################################
# Feature Extraction for spectogram output
##################################################

class WritingFeatureEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_hid=64, target_vocab_size=32):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(filters = num_hid, kernel_size =1, strides=1, padding="same", activation="relu")
        self.Norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv2 = tf.keras.layers.Conv1DTranspose(filters = num_hid, kernel_size =3, strides=2, padding="same", activation="relu")
        self.Norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv3 = tf.keras.layers.Conv1D(filters = num_hid*2, kernel_size =1, strides=2, padding="same", activation="relu")
        self.Norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.conv4 = tf.keras.layers.Conv1D(filters = target_vocab_size, kernel_size =1, strides=1, padding="same", activation="relu")
        self.Norm4 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    def call(self, x):
        x = self.conv1(x)
        x =  self.Norm1(x)
        x = self.conv2(x)
        x =  self.Norm2(x)
        x = self.conv3(x)
        x =  self.Norm3(x)
        x = self.conv4(x)
        x = self.Norm4(x)
        return x

##################################################
# Transformer
##################################################
class Transformer(tf.keras.Model):
  def __init__(self, num_layers, embed_size, heads, feed_forward, input_vocab_size, target_vocab_size,dense_hidden_units, pe_input, pe_target, num_hid, rate=0.1):
    super(Transformer, self).__init__()

    self.tokenizer = Encoder(num_layers, embed_size, heads, feed_forward,
                             input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, embed_size, heads, feed_forward,
                           dense_hidden_units, pe_target, rate)

    self.Pre_final_layer = tf.keras.layers.Dense(dense_hidden_units, name='Pre_final_layer')

    self.Pre_layer = tf.keras.layers.Dense(target_vocab_size, name = 'Pre_layer')
    self.CNNfinal_layer = WritingFeatureEmbedding(num_hid, target_vocab_size)
    self.Final_layer = tf.keras.layers.Dense(target_vocab_size, name='Final_layer')

  def call(self, inp, tar, training, enc_padding_mask,
           look_ahead_mask, dec_padding_mask):

    enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, embed_size)

    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)

    output = self.Pre_final_layer(dec_output)  
    output = self.Pre_layer(output) # (batch_size, tar_seq_len, target_vocab_size)
    output = self.CNNfinal_layer(output)
    mel = self.Final_layer(output)

    return mel, attention_weights
