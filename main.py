import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier as Model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

SLICE_INDEX = 1000

all_data = pd.read_csv('./data/train.csv', header=0)

train_data, validation_data = train_test_split(all_data, test_size=0.2)

y_train_data = np.asarray(train_data['label'])[:SLICE_INDEX]
x_train_data = np.asarray(train_data.drop('label', axis=1))[:SLICE_INDEX]

y_validation_data = np.asarray(validation_data['label'])[:SLICE_INDEX]
x_validation_data = np.asarray(validation_data.drop('label', axis=1))[:SLICE_INDEX]

x_train_data = x_train_data / 255
x_validation_data = x_validation_data / 255

model = Model(hidden_layer_sizes=(100), verbose=True)
model.fit(x_train_data, y_train_data)

print('Accuracy:', model.score(x_validation_data, y_validation_data))

# test_data = pd.read_csv('./data/test.csv', header=0)
# prediction = model.predict(test_data)
# dataset = pd.DataFrame({'ImageID': np.arange(1, len(prediction) + 1), 'Label': prediction})
# dataset.to_csv('./prediction/results.csv', index=False)
