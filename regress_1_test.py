import pandas as pd
import numpy as np
import os
from glob import glob
import keras
from sklearn.metrics import f1_score, recall_score, precision_score, mean_squared_error
from keras import Model
import matplotlib.pyplot as plt
import gc
from model_unmerged_sent import CNN_fusion_merged
from model import fusion_attention

from model import fusion_attention,fusion_attention_binary_PHQ,fusion_attention_binary_PHQ_train

def ccc(x,y):
    ''' Concordance Correlation Coefficient'''
    sxy = np.sum((x - x.mean())*(y - y.mean()))/x.shape[0]
    rhoc = 2*sxy / (np.var(x) + np.var(y) + (x.mean() - y.mean())**2)
    return rhoc


def concordance_correlation_coefficient(y_true, y_pred):
    # s_xy = np.sum((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred)) / y_true.shape[0]
    # rho_c = 2 * s_xy / (np.var(y_true) + np.var(y_pred) + (np.mean(y_true) - np.mean(y_pred)) ** 2)

    rho = np.corrcoef(y_true, y_pred)[0][1]
    y_true_mean = np.mean(y_true)
    y_pred_mean = np.mean(y_pred)
    y_true_var = np.var(y_true)
    y_pred_var = np.var(y_pred)
    y_true_std = np.std(y_true)
    y_pred_std = np.std(y_pred)

    numerator = 2 * rho * y_true_std * y_pred_std
    denominator = y_true_var + y_pred_var + (y_true_mean - y_pred_mean) ** 2

    return numerator / denominator

test_path="data/test_split.csv"
df = pd.read_csv(test_path)
df = df.dropna(axis=0,how="any")
df = df.reset_index(drop=True)
#print(df)

data_path = "padding_average_data/"


saved_weights2 = "weights/fusion_attention_mse_LR4_ep400.hdf5"
# model = Net_3feature_1regress_merged()
model = fusion_attention()
model.load_weights(saved_weights2)


# Initialization
batch_size = len(df['Participant_ID'])
#batch_size = 5

X1 = np.empty((batch_size, *(3628, 49)))
X2 = np.empty((batch_size, *(1574, 23)))
X3 = np.empty((batch_size,*(128,256)))

y1 = []
y2 = []
y3 = []
y4 = []
original_y_index =[]
labels = {}
for i in range(0, len(df['Participant_ID'])):
    yLis = []
    yLis.append(df['PHQ_Binary'][i])
    yLis.append(df['PHQ_Score'][i])
    yLis.append(df['PCL-C (PTSD)'][i])
    yLis.append(df['PTSD Severity'][i])

    labels[df['Participant_ID'][i]] = yLis.copy()

for i,ID in enumerate(df['Participant_ID']):

    print("ID:",ID)

    # AU
    AUpath = data_path + "AUs/"
    eGePath = data_path + "eGeMAPS_filtered/"
    text_path = "mini_n/"

    au_raw = np.load(AUpath + str(ID) + '.npy')
    ege_raw = np.load(eGePath + str(ID) + '.npy')
    text_raw = np.load(text_path + str(ID) + '.npy')

    X1[i,] = au_raw
    X2[i,] = ege_raw
    X3[i,] = text_raw

    # Store class
    label = labels[ID]
    y1.append(label[0])
    y2.append(label[1])
    y3.append(label[2])
    y4.append(label[3])
    original_y_index.append(ID)

y_predicted = model.predict([X1,X2,X3])
#print(y_predicted)
y2_p = y_predicted

for i in range(0,len(y2_p)):
    print('y1:',y1[i],'y2:',y2[i],' y2_p:',y2_p[i])
def convert(x):
    outcome = []
    for ele in x:
        if ele > 10:
            outcome.append(1)
        else:
            outcome.append(0)
    outcome = np.array(outcome)
    return outcome
print(np.array(y2).shape)
y2_p = np.array(y2_p)
y2_p = np.reshape(y2_p,54)
print(y1)
print(np.round(y2_p))
print(f1_score(y1, convert(y2_p), average='micro'))
y2_ccc = concordance_correlation_coefficient(np.array(y2),np.array(y2_p))
# print(type(y2))
y2_RMSE = mean_squared_error(y2, y2_p)**0.5
print('y2_ccc:',y2_ccc)
print('y2_RMSE:',y2_RMSE)
#
# attention_layer_model = Model(input=model.input,
#                                  output=model.get_layer('attention_1').output,trainable= False)
# attention_vector = attention_layer_model.predict([X1,X2,X3])
# # print(attention_vector)
# # print(attention_vector.shape)
# # print(np.argmax(attention_vector,axis=1))
# print('---')
# for ele in attention_vector:
#     print(ele)
# print('---')
# index = np.argmax(attention_vector,axis=1).astype(np.int)
#
# x = [e for e in range(128)]
# file_path = 'split/'
# for i in range(attention_vector.shape[0]):
#     # if original_y_index[i] == 602:
#
#     y = attention_vector[i]
#     plt.bar(x,y,width=1)
#     # plt.legend()
#     plt.title(str(original_y_index[i])+' '+ str(y1[i]))
#     for a, b in zip(x, y):
#         if b >=0.1:
#             plt.text(a + 0.02, b + 0.001, a, ha='center', va='bottom')
#     # plt.show()
#
#     f = open(file_path + str(original_y_index[i]) + '.txt', 'r')
#     data = f.readlines()
#     print(original_y_index[i], index[i])
#     print(data[index[i]])
#     # plt.savefig('visualization/'+ str(original_y_index[i])+'.png')
#     plt.close()
#     f.close()
#
#
#
#
# # file_path = 'split/'
# #
# # for i in range(len(original_y_index)):
# #     f = open(file_path+str(original_y_index[i])+'.txt','r')
# #     data = f.readlines()
# #     print(original_y_index[i],index[i])
# #     print(data[index[i]])
# #     f.close()