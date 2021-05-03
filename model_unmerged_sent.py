import keras
import tensorflow as tf
# from tensorflow import contrib as cb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , Activation, Masking, LSTM, GRU, Multiply,Dot
from keras.layers import Conv2D, MaxPooling2D,Conv1D,MaxPooling1D
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Lambda, Input, Dense, Concatenate ,Conv2DTranspose, TimeDistributed,Layer,Permute

from keras.layers import LeakyReLU,BatchNormalization,AveragePooling2D,Reshape
from keras.layers import UpSampling2D,ZeroPadding2D
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.optimizers import Adagrad
from keras.layers import Lambda,TimeDistributed, Bidirectional
# from Attention import attention
import numpy as np
# tf.keras.layers.Input


from Attention import attention, Weighted_hidden_states, reduce_sum, get_mask, expand_dims_multiply


def CNN_fusion_unmerged():

    input1 = Input((3628,49))
    input2 = Input((1574,23))
    input3 = Input((342,256))

    # AUs
    m1 = Masking(mask_value=0)(input1)
    mask_seq_1 = Lambda(get_mask)(m1)

    x_1 = Lambda(expand_dims_multiply)([input1,mask_seq_1])

    x_1 = Conv1D(100,10,activation = 'tanh')(x_1)
    x_1 = MaxPooling1D()(x_1)

    x_1 = Conv1D(50,5,activation = 'tanh')(x_1)
    x_1 = MaxPooling1D()(x_1)

    x_1 = Conv1D(20, 3, activation='tanh')(x_1)
    x_1 = MaxPooling1D()(x_1)

    l = Permute((2,1))(x_1)
    l = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(l) # (batch, 20, 64)

    x_1 = Permute((2,1))(l)
    atten_temp1 = Dense(20, activation = 'softmax')(x_1) # (batch, 64, 20)
    atten_1 = Permute((2,1))(atten_temp1)

    l = Multiply()([l, atten_1])
    l = reduce_sum(axis = 1)(l)

    d1 = Dense(32,activation='relu')(l)


    #eGeMAPS
    m2 = Masking(mask_value=0)(input2)
    mask_seq_2 = Lambda(get_mask)(m2)

    x_2 = Lambda(expand_dims_multiply)([input2, mask_seq_2])

    x_2 = Conv1D(100, 10, activation='tanh')(x_2)
    x_2 = MaxPooling1D()(x_2)

    x_2 = Conv1D(50, 5, activation='tanh')(x_2)
    x_2 = MaxPooling1D()(x_2)

    x_2 = Conv1D(20, 3, activation='tanh')(x_2)
    x_2 = MaxPooling1D()(x_2)

    l2 = Permute((2, 1))(x_2)
    l2 = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(l2)  # (batch, 20, 64)

    x_2 = Permute((2, 1))(l2)
    atten_temp2 = Dense(20, activation='softmax')(x_2)  # (batch, 64, 20)
    atten_2 = Permute((2, 1))(atten_temp2)

    l2 = Multiply()([l2, atten_2])
    l2 = reduce_sum(axis=1)(l2)

    d2 = Dense(32,activation="relu")(l2)

    #text

    m3 = Masking(mask_value=0)(input3)
    padding_mask = Lambda(get_mask)(m3)

    hidden_states = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(input3)

    fusion_vec = Concatenate(axis=1)([l,l2])

    attention_distribution = attention([128,96,128])([hidden_states,fusion_vec,padding_mask])

    context_vector = Weighted_hidden_states(64)([hidden_states,attention_distribution])


    d3 = Dense(32)(context_vector)

    d4 = Concatenate()([d1,d2,d3])
    d4 = Dense(16)(d4)

    o2 = Dense(1,name="regress_1")(d4)

    model = Model([input1, input2, input3],o2)

    return model

def CNN_fusion_merged():

    input1 = Input((3628,49))
    input2 = Input((1574,23))
    input3 = Input((128,256))

    # AUs
    m1 = Masking(mask_value=0)(input1)
    mask_seq_1 = Lambda(get_mask)(m1)

    x_1 = Lambda(expand_dims_multiply)([input1,mask_seq_1])

    x_1 = Conv1D(100,10,activation = 'tanh')(x_1)
    x_1 = MaxPooling1D()(x_1)

    x_1 = Conv1D(50,5,activation = 'tanh')(x_1)
    x_1 = MaxPooling1D()(x_1)

    x_1 = Conv1D(20, 3, activation='tanh')(x_1)
    x_1 = MaxPooling1D()(x_1)

    l = Permute((2,1))(x_1)
    l = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(l) # (batch, 20, 64)

    x_1 = Permute((2,1))(l)
    atten_temp1 = Dense(20, activation = 'softmax')(x_1) # (batch, 64, 20)
    atten_1 = Permute((2,1))(atten_temp1)

    l = Multiply()([l, atten_1])
    l = reduce_sum(axis = 1)(l)

    d1 = Dense(32,activation='relu')(l)


    #eGeMAPS
    m2 = Masking(mask_value=0)(input2)
    mask_seq_2 = Lambda(get_mask)(m2)

    x_2 = Lambda(expand_dims_multiply)([input2, mask_seq_2])

    x_2 = Conv1D(100, 10, activation='tanh')(x_2)
    x_2 = MaxPooling1D()(x_2)

    x_2 = Conv1D(50, 5, activation='tanh')(x_2)
    x_2 = MaxPooling1D()(x_2)

    x_2 = Conv1D(20, 3, activation='tanh')(x_2)
    x_2 = MaxPooling1D()(x_2)

    l2 = Permute((2, 1))(x_2)
    l2 = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(l2)  # (batch, 20, 64)

    x_2 = Permute((2, 1))(l2)
    atten_temp2 = Dense(20, activation='softmax')(x_2)  # (batch, 64, 20)
    atten_2 = Permute((2, 1))(atten_temp2)

    l2 = Multiply()([l2, atten_2])
    l2 = reduce_sum(axis=1)(l2)

    d2 = Dense(32,activation="relu")(l2)

    #text

    m3 = Masking(mask_value=0)(input3)
    padding_mask = Lambda(get_mask)(m3)

    hidden_states = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(input3)

    fusion_vec = Concatenate(axis=1)([l,l2])

    attention_distribution = attention([128,96,128])([hidden_states,fusion_vec,padding_mask])

    context_vector = Weighted_hidden_states(64)([hidden_states,attention_distribution])


    d3 = Dense(32)(context_vector)

    d4 = Concatenate()([d1,d2,d3])
    d4 = Dense(16)(d4)

    o2 = Dense(1,name="regress_1")(d4)

    model = Model([input1, input2, input3],o2)

    return model

# exploding parameters
def fusion_attention_timestep_attention():

    input1 = Input((3628,49))
    input2 = Input((1574,23))
    input3 = Input((342,256))

    # AUs
    m1 = Masking(mask_value=0)(input1)
    mask_seq_1 = Lambda(get_mask)(m1)

    x_1 = Lambda(expand_dims_multiply)([input1,mask_seq_1])


    l = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(x_1) # (batch, 20, 64)

    x_1 = Permute((2,1))(l)
    atten_temp1 = Dense(3628, activation = 'softmax')(x_1) # (batch, 64, 20)
    atten_1 = Permute((2,1))(atten_temp1)

    l = Multiply()([l, atten_1])
    l = reduce_sum(axis = 1)(l)

    d1 = Dense(32,activation='relu')(l)


    #eGeMAPS
    m2 = Masking(mask_value=0)(input2)
    mask_seq_2 = Lambda(get_mask)(m2)

    x_2 = Lambda(expand_dims_multiply)([input2, mask_seq_2])


    l2 = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))(x_2)

    x_2 = Permute((2, 1))(l2)
    atten_temp2 = Dense(1574, activation='softmax')(x_2)
    atten_2 = Permute((2, 1))(atten_temp2)

    l2 = Multiply()([l2, atten_2])
    l2 = reduce_sum(axis=1)(l2)

    d2 = Dense(32,activation="relu")(l2)

    #text

    m3 = Masking(mask_value=0)(input3)
    padding_mask = Lambda(get_mask)(m3)

    hidden_states = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(input3)

    fusion_vec = Concatenate(axis=1)([l,l2])

    attention_distribution = attention([128,96,128])([hidden_states,fusion_vec,padding_mask])

    context_vector = Weighted_hidden_states(64)([hidden_states,attention_distribution])


    d3 = Dense(32)(context_vector)

    d4 = Concatenate()([d1,d2,d3])
    d4 = Dense(16)(d4)

    o2 = Dense(1,name="regress_1")(d4)

    model = Model([input1, input2, input3],o2)

    return model

if __name__ == '__main__':
    model = CNN_fusion_merged()
    # model = CNN_fusion_unmerged()
    # model = CNN_fusion_lastlinear_improved_attention()
    adagrad = Adagrad(0.0001)
    model.compile(optimizer=adagrad, loss=["binary_crossentropy"])
    model.summary()
    # m = Model(model.input,)