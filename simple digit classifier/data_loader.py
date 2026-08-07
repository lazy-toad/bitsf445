import numpy as np
import pandas as pd

data = pd.read_csv("./archive/mnist_train.csv")
data = np.array(data)
labels = data[:,0]
features = data[:,1:]
features = features/255.0

print(features.shape)
print(features.min())
print(features.max())
