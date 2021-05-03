import pandas as pd
import numpy as np
import os
from glob import glob
from sklearn.metrics import f1_score, recall_score, precision_score
import keras
from keras.layers import Masking, Embedding, LSTM, Dense, Activation
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam,Adagrad

from model import fusion_attention
from model_unmerged_sent import CNN_fusion_merged
from generator import Generator_Net_3feature_1regress
from loss import ccc_loss

LR = 0.0001
BATCH_SIZE = 32
EPOCH = 400

#AU: minLen: 12447; 3628*49
#eGeMAPS: 1574*23
label_path="data/train_split.csv"
data_path="padding_average_data/"
text_path = "mini_n/"
trainLostList = [314, 385, 387, 436, 443, 445, 446]

#label_path="../data/train_split.csv"
#data_path="../padding_average_data/"
#text_path = "../mini_n/"


prefix = "CNN_Fusion"

# model = fusion_attention()
model = CNN_fusion_merged()
#model.load_weights(saved_weights)

adam = Adam(LR)
adagrad = Adagrad(LR)

#model.compile(optimizer=adam, loss=["binary_crossentropy","mse","binary_crossentropy","mse"])
model.compile(optimizer=adagrad, loss=["mse"])
model.summary()

ith = prefix +'_ep' + str(EPOCH)
log_name = 'logs/'+ ith + '.log'
csv_logger = keras.callbacks.CSVLogger(log_name)

ckpt_filepath = 'weights/'+ prefix +'_ep{epoch:02d}.hdf5'
model_ckpt = keras.callbacks.ModelCheckpoint(ckpt_filepath,period = 40)

callbacks = [csv_logger,model_ckpt,keras.callbacks.EarlyStopping(monitor='val_loss')]

train_gen = Generator_Net_3feature_1regress(label_path=label_path,data_path=data_path, text_path=text_path, trainLosList=trainLostList, output_type="PHQ", batch_size=BATCH_SIZE,shuffle=True)

model.fit_generator(generator=train_gen, epochs=EPOCH, callbacks=callbacks)

