import numpy as np
import keras
import pandas as pd


class Generator_Net_2LSTM(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, label_path, data_path, batch_size=32, shuffle=True):

        'Initialization'

        self.batch_size = batch_size
        self.data_path = data_path

        dF = pd.read_csv(label_path)

        labels = {}
        list_IDs = []

        for i in range(0,len(dF['Participant_ID'])):
            yLis = []
            yLis.append(dF['PHQ_Binary'][i])
            yLis.append(dF['PHQ_Score'][i])
            yLis.append(dF['PCL-C (PTSD)'][i])
            yLis.append(dF['PTSD Severity'][i])

            labels[dF['Participant_ID'][i]] = yLis.copy()

            list_IDs.append(dF['Participant_ID'][i])

        self.labels = labels
        self.list_IDs = list_IDs

        #self.n_classes = n_classes

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *(400,49)))
        X2 = np.empty((self.batch_size, *(400,23)))

        #y = np.empty((self.batch_size), dtype=int)
        y1 = []
        y2 = []
        y3 = []
        y4 = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            AUpath = self.data_path + "AUs/"
            eGePath = self.data_path + "eGeMAPS/"
            au_raw = np.load(AUpath + str(ID) + '.npy')
            ege_raw = np.load(eGePath + str(ID) + '.npy')

            X1[i,] = au_raw
            X2[i,] = ege_raw

            # Store class
            label = self.labels[ID]
            y1.append(label[0])
            y2.append(label[1])
            y3.append(label[2])
            y4.append(label[3])

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        return [X1,X2], [keras.utils.to_categorical(np.array(y1),num_classes=2), np.array(y2,dtype=np.float64),
                         keras.utils.to_categorical(np.array(y3),num_classes=2), np.array(y4,dtype=np.float64)]




class Generator_Net_3feature_1regress(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, label_path, data_path, text_path, trainLosList, output_type, batch_size=32, shuffle=True):

        'Initialization'

        self.batch_size = batch_size
        self.data_path = data_path
        self.text_path = text_path
        self.trainLosList = trainLosList
        self.output_type = output_type

        dF = pd.read_csv(label_path)

        labels = {}
        list_IDs = []

        for i in range(0,len(dF['Participant_ID'])):
            if dF['Participant_ID'][i] not in self.trainLosList:
                yLis = []
                yLis.append(dF['PHQ_Binary'][i])
                yLis.append(dF['PHQ_Score'][i])
                yLis.append(dF['PCL-C (PTSD)'][i])
                yLis.append(dF['PTSD Severity'][i])

                labels[dF['Participant_ID'][i]] = yLis.copy()

                list_IDs.append(dF['Participant_ID'][i])

        self.labels = labels
        self.list_IDs = list_IDs

        #self.n_classes = n_classes

        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X1 = np.empty((self.batch_size, *(3628,49)))
        X2 = np.empty((self.batch_size, *(1574,23)))
        X3 = np.empty((self.batch_size,*(128,256)))

        #y = np.empty((self.batch_size), dtype=int)
        y1 = []
        y2 = []
        y3 = []
        y4 = []

        # Generate data
        for i, ID in enumerate(list_IDs_temp):

            # Store sample

            AUpath = self.data_path + "AUs/"
            eGePath = self.data_path + "eGeMAPS_filtered/"
            au_raw = np.load(AUpath + str(ID) + '.npy')
            ege_raw = np.load(eGePath + str(ID) + '.npy')
            text_raw = np.load(self.text_path + str(ID) + '.npy')

            X1[i,] = au_raw
            X2[i,] = ege_raw
            X3[i,] = text_raw

            # Store class
            label = self.labels[ID]
            y1.append(label[0])
            y2.append(label[1])
            y3.append(label[2])
            y4.append(label[3])

        #return X, keras.utils.to_categorical(y, num_classes=self.n_classes)
        if self.output_type=="PHQ":
            return [X1,X2,X3], np.array(y2,dtype=np.float64)
        elif self.output_type=="PHQ_Binary":
            return [X1,X2,X3], np.array(y1,dtype=np.float64)
        elif self.output_type=="PTSD_Binary":
            return [X1,X2,X3], np.array(y3,dtype=np.float64)
        else:
            return [X1,X2,X3], np.array(y4,dtype=np.float64)



