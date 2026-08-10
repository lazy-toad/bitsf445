import numpy as np
from data_loader import load_data

encoded_labels, features = load_data()

#print(features.shape)
#print(encoded_labels.shape)

w1 = np.random.randn(784,30)*0.1
b1 = np.zeros(30)
print(b1.shape)

w2 = np.random.randn(30,10)*0.1
b2 = np.zeros(10)

def sigmoid(z):
    return 1 / (1+np.exp(-z))

def softmax(x):
    exp_x = np.exp(x)
    return exp_x/np.sum(exp_x,axis = 1, keepdims = True)


epochs = 200
alpha = 0.5
m_total = 60000
batch_size = 128


for i in range(epochs):

    #mini batch
    for j in range(0,m_total,batch_size):

        batch_features = features[j:j+batch_size]
        batch_labels = encoded_labels[j:j+batch_size]
        m = batch_features.shape[0]

        #forward pass
        z1 = np.dot(batch_features,w1) + b1
        a1 = sigmoid(z1)

        #2nd pass
        z2 = np.dot(a1,w2) + b2
        a2 = softmax(z2)

        #bacckprop
        dz2 = a2 - batch_labels

        dw2 = (1/m)*np.dot(a1.T,dz2)
        db2 = (1/m)*np.sum(dz2, axis=0)

        dz1 = np.dot(dz2,w2.T)*(a1*(1-a1))

        dw1 = (1/m)*np.dot(batch_features.T,dz1)
        db1 = (1/m)*np.sum(dz1,axis = 0)


        w1 = w1 - alpha * dw1
        b1 = b1 - alpha * db1
        w2 = w2 - alpha * dw2
        b2 = b2 - alpha * db2


    if i%5==0:
        test_z1 = np.dot(features, w1) + b1
        test_a1 = sigmoid(test_z1)
        test_z2 = np.dot(test_a1, w2) + b2
        test_a2 = softmax(test_z2)

        predictions = np.argmax(test_a2, axis=1)
        true_labels = np.argmax(encoded_labels, axis=1)

        accuracy = np.mean(predictions == true_labels) * 100
        print(f"Epoch {i} - Accuracy: {accuracy:.2f}%")

print("Training done")


np.savez("trained_weights.npz", w1=w1, b1=b1, w2=w2, b2=b2)
print("saved successfully to trained_weights.npz")
