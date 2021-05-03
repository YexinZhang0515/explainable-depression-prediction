import keras
import tensorflow as tf
# from tensorflow import contrib as cb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten , Activation, Masking, LSTM, GRU, Multiply,Dot
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
from keras.callbacks import Callback
from keras.layers import Lambda, Input, Dense, Concatenate ,Conv2DTranspose, TimeDistributed,Layer

from keras.layers import LeakyReLU,BatchNormalization,AveragePooling2D,Reshape
from keras.layers import UpSampling2D,ZeroPadding2D
from keras.losses import mse, binary_crossentropy
from keras.models import Model
from keras.layers import Lambda,TimeDistributed, Bidirectional
from keras.optimizers import Adagrad
from Attention import attention, Weighted_hidden_states, reduce_sum, get_mask, expand_dims_multiply


def fusion_attention():

    input1 = Input((3628,49))
    input2 = Input((1574,23))
    input3 = Input((128,256))

    m1 = Masking(mask_value=0)(input1)

    attention_probs_1 = Dense(49, activation='softmax', name='attention_vec_1')(m1)
    attention_mul_1 = Multiply(name='attention_mul_1')([m1, attention_probs_1])

    l = Bidirectional(LSTM(32, dropout=0.2, recurrent_dropout=0.2))(attention_mul_1)
    d1 = Dense(32,activation='relu')(l)

    m2 = Masking(mask_value=0)(input2)

    attention_probs_2 = Dense(23, activation='softmax', name='attention_vec_2')(m2)
    attention_mul_2 = Multiply(name='attention_mul_2')([m2, attention_probs_2])

    l2 = Bidirectional(LSTM(16, dropout=0.2, recurrent_dropout=0.2))(attention_mul_2)
    d2 = Dense(32,activation="relu")(l2)

    #text
    m3 = Masking(mask_value=0)(input3)
    padding_mask = Lambda(get_mask)(m3)

    hidden_states = Bidirectional(LSTM(64, dropout=0.2, recurrent_dropout=0.2,return_sequences=True))(input3)

    fusion_vec = Concatenate(axis=1)([l,l2])

    attention_distribution = attention([128,96,128])([hidden_states,fusion_vec,padding_mask])

    context_vector = Weighted_hidden_states(64)([hidden_states,attention_distribution])

    d3 = Dense(32)(context_vector)
    d3 = Concatenate()([d1,d2,d3])
    d4 = Dense( 16)(d3)
    o2 = Dense(1,name="regress_1")(d4)

    model = Model([input1, input2, input3],o2)

    return model

def fusion_attention_binary_PHQ():
    model = fusion_attention()
    saved_weights = "weights/fusion_attention_mse_LR4_ep400.hdf5"
    model.load_weights(saved_weights)
    input1 = Input((3628, 49))
    input2 = Input((1574, 23))
    input3 = Input((128, 256))

    m = Model(input=model.input, output=model.get_layer('concatenate_2').output,trainable= False)
    x = m([input1, input2, input3])
    x = Dense(16)(x)
    x = Dense(1, name='last', activation='sigmoid')(x)
    M = Model([input1, input2, input3], output=x)
    return M

def fusion_attention_binary_PHQ_train():
    model = fusion_attention()
    input1 = Input((3628, 49))
    input2 = Input((1574, 23))
    input3 = Input((128, 256))

    m = Model(input=model.input, output=model.get_layer('concatenate_2').output,trainable= True)
    x = m([input1, input2, input3])
    x = Dense(16)(x)
    x = Dense(1, name='last', activation='sigmoid')(x)
    M = Model([input1, input2, input3], output=x)
    return M
    # M.compile(optimizer=adagrad, loss=['binary_crossentropy'])
    # M.summary()

if __name__ == '__main__':
    adagrad = Adagrad(0.0001)
    model = fusion_attention()
    model.compile(optimizer=adagrad, loss=['mse'])
    model.summary()

    # M = fusion_attention_binary_PHQ()
    # M.compile(optimizer=adagrad, loss=['binary_crossentropy'])
    # M.summary()
