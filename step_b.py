from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import drive
from sklearn.metrics import  precision_score, recall_score, f1_score, multilabel_confusion_matrix

import numpy as np
import pandas as pd

drive.mount('/content/gdrive', force_remount=True)


#Read CSVs
test_data = pd.read_csv('/content/gdrive/MyDrive/InputTest.csv')
train_data = pd.read_csv('/content/gdrive/MyDrive/InputTrain.csv')
train_dish = pd.read_csv('/content/gdrive/MyDrive/StepTwo_LabelTrain_Dishwasher.csv')
train_kettle = pd.read_csv('/content/gdrive/MyDrive/StepTwo_LabelTrain_Kettle.csv')
train_micro = pd.read_csv('/content/gdrive/MyDrive/StepTwo_LabelTrain_Microwave.csv')
train_tumble = pd.read_csv('/content/gdrive/MyDrive/StepTwo_LabelTrain_TumbleDryer.csv')
train_washing = pd.read_csv('/content/gdrive/MyDrive/StepTwo_LabelTrain_WashingMachine.csv')

#Remove House_id and Index columns
test_data = test_data.drop('House_id', axis=1).drop('Index', axis=1).astype(float)
train_data = train_data.drop('House_id', axis=1).drop('Index', axis=1).astype(float)
train_dish = train_dish.drop('House_id', axis=1).drop('Index', axis=1)
train_kettle = train_kettle.drop('House_id', axis=1).drop('Index', axis=1)
train_micro = train_micro.drop('House_id', axis=1).drop('Index', axis=1)
train_tumble = train_tumble.drop('House_id', axis=1).drop('Index', axis=1)
train_washing = train_washing.drop('House_id', axis=1).drop('Index', axis=1)

#Convert CSVs to numpy
X_train = train_data.to_numpy()
X_test = test_data.to_numpy()
y_dish = train_dish.to_numpy()
y_kettle = train_kettle.to_numpy()
y_micro = train_micro.to_numpy()
y_tumble = train_tumble.to_numpy()
y_wash = train_washing.to_numpy()

#Normalization of both X_train and X_test
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

#Reshape for training and combining every labels into one ndarray
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_t = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
y_dish = y_dish.reshape((y_dish.shape[0], y_dish.shape[1], 1))
y_kettle = y_kettle.reshape((y_kettle.shape[0], y_kettle.shape[1], 1))
y_micro = y_micro.reshape((y_micro.shape[0], y_micro.shape[1], 1))
y_tumble = y_tumble.reshape((y_tumble.shape[0], y_tumble.shape[1], 1))
y_wash = y_wash.reshape((y_wash.shape[0], y_wash.shape[1], 1))

y = np.dstack([y_dish, y_kettle, y_micro, y_tumble, y_wash])


#Split into train and test
X_train, X_test, y_train, y_test = train_test_split(X_train, y, test_size=0.2)#y

#Slice the rows into smalle row of length 120
X_train_sliding = np.reshape(X_train, (8336*2160//120, 120, 1))
y_train_sliding = np.reshape(y_train, (8336*2160//120, 120, 5))
X_test_sliding = np.reshape(X_test, (2085*2160//120, 120, 1))
y_test_sliding = np.reshape(y_test, (2085*2160//120, 120, 5))
X_t_sliding = np.reshape(X_t, (2488*2160//120, 120, 1))

#LSTM model
model = Sequential()
model.add(LSTM(700, input_shape=(120, 1), return_sequences=True))
model.add(LSTM(800, return_sequences=True))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(TimeDistributed(Dense(5, activation='sigmoid')))

optimizer = Adam(learning_rate=0.00001)
model.compile(optimizer=optimizer, metrics=['accuracy'], loss='binary_crossentropy')

callback = EarlyStopping(monitor='val_loss', patience=2)#stops model if val_loss decreases for two epochs
history = model.fit(X_train_sliding, y_train_sliding, epochs=100, batch_size=128, validation_data=(X_test_sliding, y_test_sliding), callbacks=[callback])

y_pred_sliding = model.predict(X_t_sliding)

#Reshape results and saving to CSV
final = y_pred_sliding.reshape((2488, 2160, y_pred_sliding.shape[2]))
arr = np.round(final).astype(np.uint8)
arr2 = arr.reshape((-1, arr.shape[2]))
cols = ['Washing Machine', 'Dishwasher', 'Tumble Dryer', 'Microwave', 'Kettle']
df = pd.DataFrame(arr2, columns=cols)
df.index.name = 'Index'
df.to_csv('res.csv', index=True)






####### Evalutation ########

#Reshape data to use sklearn's evalutation metrics
y_pred_sliding = model.predict(X_test_sliding)
y_pred_sliding = y_pred_sliding.reshape((2085, 2160, y_pred_sliding.shape[2]))
y_pred_reshaped = y_pred_sliding.reshape(-1, 5)
y_pred_reshaped = (y_pred_reshaped >= 0.5).astype(int)

y_test_sliding = y_test_sliding.reshape((2085, 2160, y_test_sliding.shape[2]))
y_test_sliding = y_test_sliding.reshape(-1, 5)
y_test_reshaped = y_test_sliding.reshape(-1, 5)

#Metrics for each appliances
f1_scores = f1_score(y_test_reshaped, y_pred_reshaped, average=None)
recalls = recall_score(y_test_reshaped, y_pred_reshaped, average=None)
precisions = precision_score(y_test_reshaped, y_pred_reshaped, average=None)

#Average metrics
print(f1_score(y_test_reshaped, y_pred_reshaped, average='macro'))
print(recall_score(y_test_reshaped, y_pred_reshaped, average='macro'))
print(precision_score(y_test_reshaped, y_pred_reshaped, average='macro'))

print(f1_scores)
print(recalls)
print(precisions)

#Saving plots
classes = ['Dishwasher', 'Kettle', 'Microwave', 'Tumble Dry.', 'Washing mach.']

plt.bar(classes, f1_scores)
plt.xlabel('Class')
plt.ylabel('F1-score')
plt.title('F1-scores')
plt.savefig('f1_scores.png')
plt.close()

plt.bar(classes, recalls)
plt.xlabel('Class')
plt.ylabel('Recall')
plt.title('Recalls')
plt.savefig('recalls.png')
plt.close()

plt.bar(classes, precisions)
plt.xlabel('Class')
plt.ylabel('Precision')
plt.title('Precisions')
plt.savefig('precisions.png')
plt.close()

#Confusion matrix
ml_cm = multilabel_confusion_matrix(y_test_reshaped, y_pred_reshaped)
for i, cm in enumerate(ml_cm):
    fig, ax = plt.subplots(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_title(classes[i])
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    plt.tight_layout()

    plt.savefig(classes[i])
    plt.close(fig)
