
# Importing Required Packages
import wfdb
import keras
from keras import layers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
from tensorflow.keras.utils import to_categorical
import random
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import copy
import seaborn as sns
from pylab import rcParams
from matplotlib import rc
from sklearn.model_selection import train_test_split
import tensorflow as tf
from sklearn.metrics import precision_score,recall_score,accuracy_score
from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
import scipy.io
from scipy.io import savemat
# Random Initialization
random.seed(42)

import os
print(os.listdir("/Users/vijaykumar/Desktop/PROJECT/final/REPORT/detection_datasets/"))


data= '/Users/vijaykumar/Desktop/PROJECT/final/REPORT/detection_datasets/mit-bih-arryhmia-dataset/'
# List of Patients
patients = ['100','101','102','103','104','105','106','107',
           '108','109','111','112','113','114','115','116',
           '117','118','119','121','122','123','124','200',
           '201','202','203','205','207','208','209','210',
           '212','213','214','215','217','219','220','221',
           '222','223','228','230','231','232','233','234']

# Creating a Empty Dataframe
dataframe = pd.DataFrame()

# Reading all .atr files
for pts in patients:
    # Generating filepath for all .atr file names
    file = data + pts

    # Saving annotation object
    annotation = wfdb.rdann(file, 'atr')

    # Extracting symbols from the object
    sym = annotation.symbol

    # Saving value counts
    values, counts = np.unique(sym, return_counts=True)

    # Writing data points into dataframe
    df_sub = pd.DataFrame({'symbol':values, 'Counts':counts, 'Patient Number':[pts]*len(counts)})

    # Concatenating all data points
    dataframe = pd.concat([dataframe, df_sub],axis = 0)


ax = sns.countplot(dataframe.symbol)


dataframe

# Non Beat Symbols
nonbeat = ['[','!',']','x','(',')','p','t','u','`',
           '\'','^','|','~','+','s','T','*','D','=','"','@','Q','?']

# Abnormal Beat Symbols
abnormal = ['L','R','V','/','A','f','F','j','a','E','J','e','S']

# Normal Beat Symbols
normal = ['N']


# Classifying normal, abnormal or nonbeat
dataframe['category'] = -1
dataframe.loc[dataframe.symbol == 'N','category'] = 0
dataframe.loc[dataframe.symbol.isin(abnormal), 'category'] = 1

dataframe.groupby('category').Counts.sum()

#removing the non-beat from dataframe
dataframe = dataframe.loc[~((dataframe['category']==-1))]
dataframe.groupby('category').Counts.sum()

def load_ecg(file):
    # load the ecg
    record = wfdb.rdrecord(file)
    # load the annotation
    annotation = wfdb.rdann(file, 'atr')

    # extracting the signal
    p_signal = record.p_signal

    # extracting symbols and annotation index
    atr_sym = annotation.symbol
    atr_sample = annotation.sample

    return p_signal, atr_sym, atr_sample

def build_XY(p_signal, df_ann, num_cols, normal):
    # this function builds the X,Y matrices for each beat
    # it also returns the original symbols for Y

    num_rows = len(df_ann)

    X = np.zeros((num_rows, num_cols))
    Y = np.zeros((num_rows,1))
    sym = []

    # keep track of rows
    max_row = 0

    for atr_sample, atr_sym in zip(df_ann.atr_sample.values,df_ann.atr_sym.values):

        left = max([0,(atr_sample - num_sec*fs) ])
        right = min([len(p_signal),(atr_sample + num_sec*fs) ])
        x = p_signal[left: right]
        if len(x) == num_cols:
            X[max_row,:] = x
            Y[max_row,:] = int(atr_sym in normal)
            sym.append(atr_sym)
            max_row += 1
    X = X[:max_row,:]
    Y = Y[:max_row,:]
    return X,Y,sym

    # creating abnormal beat dataset

def make_dataset(pts, num_sec, fs, abnormal):
    # function for making dataset ignoring non-beats
    # input:
    #   pts - list of patients
    #   num_sec = number of seconds to include before and after the beat
    #   fs = frequency
    # output:
    #   X_all = signal (nbeats , num_sec * fs columns)
    #   Y_all = binary is abnormal (nbeats, 1)
    #   sym_all = beat annotation symbol (nbeats,1)

    # initialize numpy arrays
    num_cols = 2*num_sec * fs
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    sym_all = []

    # list to keep track of number of beats across patients
    max_rows = []

    for pt in pts:
        file = data + pt

        p_signal, atr_sym, atr_sample = load_ecg(file)

        # grab the first signal
        p_signal = p_signal[:,0]

        # make df to exclude the nonbeats
        df_ann = pd.DataFrame({'atr_sym':atr_sym,
                              'atr_sample':atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal)]

        X,Y,sym = build_XY(p_signal,df_ann, num_cols, abnormal)
        sym_all = sym_all+sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all,X,axis = 0)
        Y_all = np.append(Y_all,Y,axis = 0)

    # drop the first zero row
    X_all = X_all[1:,:]
    Y_all = Y_all[1:,:]

    return X_all, Y_all, sym_all


# Parameter Values
num_sec = 1
fs = 360

X_abnormal, Y_abnormal, sym_abnormal = make_dataset(patients, num_sec, fs, abnormal)


#normal ecg data

data= '/Users/vijaykumar/Desktop/PROJECT/final/REPORT/detection_datasets/mit-bih-normal-sinus-rhythm-database/'

patients = ["16265","16272"]

# creating normal beat dataset
def make_dataset(pts, num_sec, fs, normal):
    # function for making dataset ignoring non-beats
    # input:
    #   pts - list of patients
    #   num_sec = number of seconds to include before and after the beat
    #   fs = frequency
    # output:
    #   X_all = signal (nbeats , num_sec * fs columns)
    #   Y_all = binary is abnormal (nbeats, 1)
    #   sym_all = beat annotation symbol (nbeats,1)

    # initialize numpy arrays
    num_cols = 2*num_sec * fs
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    sym_all = []

    # list to keep track of number of beats across patients
    max_rows = []

    for pt in pts:
        file = data + pt

        p_signal, atr_sym, atr_sample = load_ecg(file)

        # grab the first signal
        p_signal = p_signal[:,0]

        # make df to exclude the nonbeats
        df_ann = pd.DataFrame({'atr_sym':atr_sym,
                              'atr_sample':atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(normal)]

        X,Y,sym = build_XY(p_signal,df_ann, num_cols, normal)
        sym_all = sym_all+sym
        max_rows.append(X.shape[0])
        X_all = np.append(X_all,X,axis = 0)
        Y_all = np.append(Y_all,Y,axis = 0)

    # drop the first zero row
    X_all = X_all[1:,:]
    Y_all = Y_all[1:,:]

    return X_all, Y_all, sym_all

# Parameter Values
num_sec = 1
fs = 360

X_normal, Y_normal, sym_normal = make_dataset(patients, num_sec, fs, normal)

X_normal = X_normal[0:34376,:]

Y_normal = np.zeros((34376, 1))


X = np.append(X_normal,X_abnormal,axis=0)
Y = np.append(Y_normal,Y_abnormal,axis=0)
X = X[:,0:140]

raw_data= np.append(X, Y, axis=1)
raw_data = pd.DataFrame(raw_data)

# The last element contains the labels
labels = raw_data.iloc[: , -1]
labels= labels.values
# The other data points are the electrocadriogram data
data = raw_data.iloc[:, 0:-1]
data= data.values


train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(train_data)
test_data = scaler.transform(test_data)
train_data = scaler.transform(train_data)


train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[~train_labels]
normal_test_data = test_data[~test_labels]

anomalous_train_data = train_data[train_labels]
anomalous_test_data = test_data[test_labels]

val_df, test_df = train_test_split(
  test_data,
  test_size=0.2,
  random_state=42)

test_labels =  ~test_labels


#normal ecg plotting

plot.grid()
plot.plot(np.arange(140), normal_train_data[0])
plot.title("normal train data")
plot.show()

plot.grid()
plot.plot(np.arange(140), normal_test_data[543])
plot.title("normal test data")
plot.show()


#plot abnormal ecg

plot.grid()
plot.plot(np.arange(140), anomalous_train_data[0])
plot.title("anomalous train data")
plot.show()

plot.grid()
plot.plot(np.arange(140), anomalous_test_data[0])
plot.title("anomalous test data")
plot.show()



#build model

class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(128, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
])
    self.decoder = tf.keras.Sequential([

      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(64, activation="relu"),
      layers.Dense(128, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

autoencoder = AnomalyDetector()


autoencoder.compile("adam", loss="mean_absolute_error")


# Result at the 1500 epoch and 128 batch size, where the model achieve consistently strong results.


# history = autoencoder.fit(normal_train_data, normal_train_data,epochs=1500, batch_size=128,
#                           validation_data=(normal_test_data, normal_test_data), shuffle=True)

# plot.plot(history.history["loss"], label="Training Loss")
# plot.plot(history.history["val_loss"], label="Validation Loss")
# plot.legend()



#  If the reconstruction error is larger than one standard deviation from normal training samples, we will shortly label an ECG as abnormal. Let's start with a normal ECG from the training set, then the reconstruction after the autoencoder has encoded and decoded it, also the reconstruction error.

encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plot.plot(normal_test_data[0], 'b')
plot.plot(decoded_data[0], 'r')
plot.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
plot.legend(labels=["Input", "Reconstruction", "Error"])
plot.show()


# Calculate whether the reconstruction loss is larger than the defined threshold to detect abnormalities. In this section, you'll calculate the mean average error for normal cases in the training set, and then classify future examples as abnormal if the reconstruction error exceeds the training dataset's standard deviation. Plot the reconstruction error on the training set's normal ECGs.



encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plot.plot(anomalous_test_data[0], 'b')
plot.plot(decoded_data[0], 'r')
plot.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plot.legend(labels=["Input", "Reconstruction", "Error"])
plot.show()


reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

plot.hist(train_loss[None,:], bins=50)
plot.xlabel("Train loss")
plot.ylabel("No of examples")
plot.show()

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)


#When we look at the reconstruction error for the anomalous examples in the test set, we can see that the majority have a higher reconstruction error than the threshold. We may improve your classifier's precision and recall by adjusting the threshold.

reconstructions = autoencoder.predict(anomalous_test_data)
test_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)

plot.hist(test_loss[None, :], bins=50)
plot.xlabel("Test loss")
plot.ylabel("No of examples")
plot.show()


def predict(model, data, threshold):
  reconstructions = model(data)
  loss = tf.keras.losses.mae(reconstructions, data)
  return tf.math.less(loss, threshold)

def print_stats(predictions, labels):
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  print("Precision = {}".format(precision_score(labels, predictions)))
  print("Recall = {}".format(recall_score(labels, predictions)))
#test_data = test_data. numpy()


preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)



reconstructions = autoencoder.predict(anomalous_test_data)
train_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
sns.distplot(train_loss, bins=50, kde=True);


# Furthermore, we can count the number of examples that exceed the threshold (we will consider as anomalies)



reconstructions = autoencoder.predict(normal_test_data)
pred_loss = tf.keras.losses.mae(reconstructions, normal_test_data)

pred_loss = pred_loss.numpy()
correct = sum(l <= threshold for l in pred_loss)
print(f'Correct normal predictions: {correct}/{len(normal_test_data)}')


reconstructions = autoencoder.predict(anomalous_test_data)
train_loss = tf.keras.losses.mae(reconstructions, anomalous_test_data)
train_loss = train_loss.numpy()
correct = sum(l > threshold for l in train_loss)
print(f'Correct anomaly predictions: {correct}/{len(anomalous_test_data)}')



def plot_prediction_normal(i,data, model, title, ax):
    encoded_data = autoencoder.encoder(data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
    ax.axis([0, 140, 0, 1])

    ax.plot(data[i], label='true')
    ax.plot(decoded_data[i], label='reconstructed')
    ax.set_title(f'{title} (loss: {np.around(1000*pred_loss[i], 2)})')
    ax.legend()

def plot_prediction_anomaly(i,data, model, title, ax):
    encoded_data = autoencoder.encoder(data).numpy()
    decoded_data = autoencoder.decoder(encoded_data).numpy()
    ax.axis([0, 140, 0, 1])

    ax.plot(data[i], label='true')
    ax.plot(decoded_data[i], label='reconstructed')
    ax.set_title(f'{title} (loss: {np.around(1000*train_loss[i], 2)})')
    ax.legend()
fig, axs = plot.subplots(
  nrows=2,
  ncols=5,
  figsize=(22, 8)
)
for i in range(5) :
  plot_prediction_normal(i,normal_test_data, autoencoder, title='Normal', ax=axs[0, i])

for i in range(5) :
  plot_prediction_anomaly(i,anomalous_test_data, autoencoder, title='Anomaly', ax=axs[1, i])
fig.tight_layout();



# This project provides the results of an unsupervised anomaly detection for ECG data which is performed on MIT-BIH Arrhythmia Dataset and MIT-BIH Normal Sinus Rhythm Dataset. Although the findings are promising, there is certainly potential for improvement. Future projects might include: more complex model and different type of wavelet and more data to detect anomalies.
