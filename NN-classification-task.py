import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D, Input
import numpy as np
from sklearn.datasets import make_classification
from tensorflow.keras.callbacks import TensorBoard
import time
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from tensorflow.keras.utils import to_categorical, normalize

X, y = make_classification(n_samples=50000, n_features=15, n_informative=8, n_classes=3, random_state=149)
X = normalize(X, axis=1)
y = to_categorical(y)

layers = [3, 4, 5]
nodes = [32, 64, 128]
dropouts = [0, 1, 2]
drop_rates = [0.1, 0.3, 0.5]

for n_layers in layers:
    for n_nodes in nodes:
        for drop_rate in drop_rates:
            NAME = "Basic-NN-{}-X-{}-nodes-{}-drop_rate={}".format(n_layers, n_nodes, int(time.time()), drop_rate)
            tensorboard = TensorBoard(log_dir=".\logs\\{}".format(NAME))
            model = Sequential()
            model.add(Input((X.shape[1:]))) # Input layer -> number of nodes will be number of features
            for i in range(n_layers):
                model.add(Dense(n_nodes, activation="relu")) # Layer loop
            model.add(Dropout(drop_rate))
            model.add(Dense(3, activation="sigmoid"))
            model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy'])
            model.fit(X,y,epochs=10, validation_split=0.2, callbacks=[tensorboard])

##### VISUALISATION: -----------------------------------------------------------
def vis(X,y):
    y = y.reshape(-1,1)
    print(y.shape)
    pca = PCA(n_components=3, random_state=42)
    pca = pca.fit_transform(X, y)
    pca1 = pca[:, 0].reshape(-1,1)
    pca2 = pca[:, 1].reshape(-1,1)
    pca3 = pca[:, 2].reshape(-1,1)
    ax = plt.subplot(111, projection="3d")
    ax.plot(pca1[y==0], pca2[y==0], pca3[y==0], "r.", label="Class 0")
    ax.plot(pca1[y==1], pca2[y==1], pca3[y==1], "b.", label="Class 1")
    ax.plot(pca1[y==2], pca2[y==2], pca3[y==2], "g.", label="Class 2")
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.set_zlabel("PCA3")
    plt.legend()
    plt.figure()
    plt.show()
#vis(X,y)
