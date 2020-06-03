import tensorflow as tf
import keras as kr
import matplotlib.pyplot as plt
import mpld3
import numpy as np
import torch as tor
from sklearn.model_selection import train_test_split
import pandas as pd
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation





dataset = pd.read_csv('C:/Users/athanasis/Desktop/Semester_B/DEEP_LEARNING/digit_recognizer_dataset.csv', sep=',')
label = dataset.iloc[:,0]

features =  dataset.iloc[:,1:]
X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.30)

print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize to range 0-1
X_train = X_train / 255.0
X_test = X_test / 255.0
# return normalized images


def first_part(X_train, X_test, y_train, y_test):

        model = Sequential()
        model.add(Dense(10, input_shape=(784,)))
        model.add(Activation('relu'))


        model.add(Dense(10))
        model.add(Activation('relu'))

        model.add(Dense(10))
        model.add(Activation('softmax'))



        batch_size = 64
        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                      loss="sparse_categorical_crossentropy",
                      metrics=['accuracy'])

        model.fit(X_train,y_train, batch_size=batch_size, epochs=10)



        loss, accuracy = model.evaluate(X_test,y_test)
        print("Accuracy for the starting part ",accuracy)
        return (loss, accuracy)


def second_part(X_train, X_test, y_train, y_test):
    model = Sequential()
    model.add(Dense(1024, input_shape=(784,)))
    model.add(Activation('relu'))


    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(512))
    model.add(Activation('relu'))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    batch_size = 64
    model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                  loss="sparse_categorical_crossentropy",
                  metrics=['accuracy'])

    model.fit(X_train, y_train, batch_size=batch_size, epochs=18)

    loss, accuracy = model.evaluate(X_test, y_test)
    print("Accuracy for the Hyperparameter part ", accuracy)
    return (loss,accuracy)





bef_loss, bef_acc =  first_part(X_train, X_test, y_train, y_test)
hyp_loss, hyp_acc = second_part(X_train, X_test, y_train, y_test)
print('/n')
print('======== Before =======')
print('Loss & Accuracy', bef_loss, bef_acc )
print('======== After Parameter Tuning =======')
print('Loss & Accuracy', hyp_loss, hyp_acc )