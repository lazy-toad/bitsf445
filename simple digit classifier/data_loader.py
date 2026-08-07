import numpy as np
import pandas as pd


def load_data():
    data = pd.read_csv("./archive/mnist_train.csv")
    data = np.array(data)
    labels = data[:,0]
    features = data[:,1:]
    features = features/255.0

    #print(features.shape)
    #print(features.min())
    #print(features.max())


    encoded_labels = np.zeros((60000,10))
    encoded_labels[np.arange(60000),labels] = 1

    return encoded_labels, features
