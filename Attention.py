import tensorflow as tf
import numpy as np
from keras.layers import Layer

# expected hidden_states, [batch, time_steps, hidden_states_units], a list of hidden_state, h_i
# expected fused_vec, [batch,visual&acoustic_fusion_feature_dims]
# e_i for fused_vec = vT *tanh(W_h* h_i + W_f * fused_vec + b_attention)
# attention distribution = softmax(e)

# when init, declare hidden_state_units of lstm, feature dims of fusion acoustic and visual vector
# attention dimensions, it can be set to hidden_state_units, padding mask -> like [ 1 1 1 1 0 0 ]
# the current padding

class attention(Layer):
    def __init__(self, input_dim):
        self.hidden_state_units=input_dim[0]
        self.fusion_vec_dims=input_dim[1]
        self.attention_dims = input_dim[2]
        super(attention, self).__init__()
        # self.W_h = tf.compat.v1.get_variable('W_h', shape = [hidden_state_units,self.attention_dims])
        # self.W_f = tf.compat.v1.get_variable('W_f', shape = [fusion_vec_dims,self.attention_dims])
        # self.b_attn = tf.compat.v1.get_variable("b_attn", [attention_dims])
        # self.vT = self.add_variable("v", [attention_dims,1]) -> delete

    def build(self, input_shape):
        self.W_h = self.add_weight('W_h', shape=[input_shape[0][-1], input_shape[0][-1]],initializer='uniform',trainable=True)
        self.W_f = self.add_weight('W_f', shape=[input_shape[1][-1], input_shape[0][-1]],initializer='uniform',trainable=True)
        self.b_attn = self.add_weight("b_attn", [input_shape[0][-1]],initializer='zeros',trainable=True)
        self.built = True
    # input:
    # bilstm ->(2,batch,time_step,hidden_units)
    # hidden_state: reshape (batch,time_step,2*hidden_units)
    # fusion_vec: acoustic visual concat()-> (batch,a_dims + v_dims)
    # return: [0.1,0.2,...,0.1], (batch, time_steps)
    # batch * [[],[],[]] -> batch* [ ]
    # (batch,time_steps,2*hidden_units) * (batch,time_steps) -> (batch,time_steps,2*hidden_units)
    # weighted_hidden_states = hidden_states * attention
    # context_vector = tf.reduce_sum(weighted_hidden_states,axis = 1)/time_steps ->(batch, 2*hidden_units)
    # d3 = Dense(context_vector)
    # BiLSTM()

    def call(self, inputs):
        hidden_states = inputs[0]
        # print('in attention hidden_states.shape')
        # print(hidden_states.shape)
        fusion_vec = inputs[1]
        padding_mask = inputs[2]
        h_i_s = tf.unstack(hidden_states, axis= 1)
        _ = []
        for h_i in h_i_s: # h_i [batch, hidden_state_units] -> [batch, attention_dims]; fusion_vec [batch, Dims] - > [batch, attention_dims]
            # e = tf.matmul(math_ops.tanh(tf.matmul(h_i,self.W_h) + tf.matmul(fusion_vec, self.W_f) + self.b_attn), self.vT) -> delete
            e = tf.reduce_sum(tf.tanh(tf.matmul(h_i,self.W_h) + tf.matmul(fusion_vec, self.W_f) + self.b_attn),axis= 1)
            # e = tf.reduce_sum(e, axis= 1)
            # print(e.shape)
            _.append(e)

        attention_distribution = tf.stack(_,axis=1)
        # print(attention_distribution.shape)
        # attention_distribution = tf.reshape() -> delete
        attention_distribution = tf.nn.softmax(attention_distribution)
        attention_distribution *= padding_mask
        # print(attention_distribution.shape)
        masked_sum = tf.reduce_sum(attention_distribution, axis=1)
        # print(masked_sum.shape)
        outcome = attention_distribution / tf.reshape(masked_sum,[-1,1])
        return outcome

    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0],input_shape[0][1])
        return tuple(output_shape)

class Weighted_hidden_states(Layer):
    def __init__(self, input_dim):
        super(Weighted_hidden_states, self).__init__()
        self.dim = input_dim

    def build(self, input_shape):
        self.build = True

    # weighted sum of hidden_states of attention
    def call(self, inputs):
        hidden_states = inputs[0]
        attention_distribution = inputs[1]
        temp2 = []
        # for i in range(self.dim):
        #     temp2.append(attention_distribution)
        temp = tf.unstack(hidden_states,axis=2)
        for t in temp:
            temp2.append(attention_distribution)
        atten_dis = tf.stack(temp2, axis=2)
        # print('atten_dist.shape')
        # print(atten_dis.shape)
        outcome = tf.multiply(hidden_states, atten_dis)
        outcome = tf.reduce_sum(outcome,axis=1)
        # print(outcome.shape)
        return outcome
    #
    def compute_output_shape(self, input_shape):
        output_shape = (input_shape[0][0],input_shape[0][-1])
        return tuple(output_shape)

class reduce_sum(Layer):
    def __init__(self, axis=-1):
        super(reduce_sum, self).__init__()
        self.axis = axis

    def build(self, input_shape):
        self.build = True

    def call(self, inputs):
        return tf.reduce_sum(inputs,axis=self.axis)

    def compute_output_shape(self, input_shape):
        return tuple((input_shape[0],input_shape[-1]))


def get_mask(m3):

    # print(m3)
    to_mask = np.zeros(m3.get_shape()[-1])
    temp = tf.unstack(m3, axis=1)
    padding_mask = []
    for t in temp:
        padding_mask.append(1 - tf.cast(tf.reduce_all(tf.equal(t, to_mask), axis=1), tf.float32))
    # print(padding_mask)
    # print(len(temp))
    padding_mask = tf.stack(padding_mask, axis=1)
    return padding_mask

def expand_dims_multiply(inputs):
    a = inputs[0]
    b = inputs[1]
    b = tf.expand_dims(b,axis = 2)
    outcome = tf.multiply(a,b)
    return outcome


if __name__ == '__main__':
    # test code
    Attention= attention([12,10,5])
    print(Attention([tf.zeros([3, 5, 12]),tf.ones([3,10]),tf.ones(5)]))

