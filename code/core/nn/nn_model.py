from tensorflow import keras
import numpy as np
import argparse
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from sklearn import metrics
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--x_train", help="path of x_train", type=str)
parser.add_argument("--y_train", help="path of y_train", type=str)
parser.add_argument("--x_test", help="path of x_test", type=str)
parser.add_argument("--y_test", help="path of y_test", type=str)
args = parser.parse_args()

x_train_path = args.x_train
y_train_path = args.y_train
x_test_path = args.x_test
y_test_path = args.y_test

def main():
    x_train, y_train, x_test, y_test = np.load(x_train_path, allow_pickle=True), np.load(y_train_path, allow_pickle=True), \
                                    np.load(x_test_path, allow_pickle=True), np.load(y_test_path, allow_pickle=True)

    def simple_model():
        input_layer = Input(shape=x_train[0].shape)
        # bm_layer = BatchNormalization()(input_layer)
        dense_layer = Dense(100, activation="relu")(input_layer)
        # bm_layer = BatchNormalization()(dense_layer)
        dense_layer = Dense(10, activation="relu")(dense_layer)
        # bm_layer = BatchNormalization()(dense_layer)
        output_layer = Dense(1, activation="sigmoid")(dense_layer)
        
        model = Model(input_layer, output_layer)
        model.summary()

        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(0.0001), metrics=["acc"])
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=2) 

        return model

    
    def bm_model():
        input_layer = Input(shape=x_train[0].shape)
        bm_layer = BatchNormalization()(input_layer)
        dense_layer = Dense(1000, activation="relu")(bm_layer)
        bm_layer = BatchNormalization()(dense_layer)
        dense_layer = Dense(100, activation="relu")(bm_layer)
        bm_layer = BatchNormalization()(dense_layer)
        output_layer = Dense(1, activation="sigmoid")(bm_layer)

        model = Model(input_layer, output_layer)
        model.summary()

        model.compile(loss=keras.losses.binary_crossentropy, optimizer=keras.optimizers.Adam(0.0001), metrics=["acc"])
        model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=2) 

        return model

    model = simple_model()
    y_score = np.squeeze(model.predict(x_test))
    print(y_test.shape, y_score.shape)
    print(y_test, y_score)
    draw(y_score, y_test, "simple")
    del model
    model = bm_model()
    y_score = np.squeeze(model.predict(x_test))
    draw(y_score, y_test, "bm")

def draw(y_score, y_true, msg):
    fpr, tpr, _ = metrics.roc_curve(y_true, y_score)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('1-Specificity')
    plt.ylabel('Sensitivity')
    plt.savefig(msg+"_nn_auc.png")


if __name__ == "__main__":
    main()   


