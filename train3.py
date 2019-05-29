# set the matplotlib backend so figures can be saved in the background
import matplotlib

matplotlib.use("Agg")

# import the necessary packages
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
from imutils import paths
import random
import cv2
import os
import gc
import sys
from keras import backend as K
import tensorflow as tf

sys.path.append('..')

from net.lenet import LeNet
from net.VGG16 import VGG_16
from net.Unet import unet
from net.resnet import ResnetBuilder
from net.inceptionV4 import inception_v4


def args_parse():
    # construct the argument parse and parse the arguments  /home/tang/Desktop/imageclass/data
    ap = argparse.ArgumentParser()
    ap.add_argument("-dtest", "--dataset_test", default="data/data/test",
                    help="path to input dataset_test")
    ap.add_argument("-dtrain", "--dataset_train", default="data/data/train",
                    help="path to input dataset_train")
    ap.add_argument("-m", "--model", default="./model.h5",
                    help="path to output model")
    ap.add_argument("-p", "--plot", type=str, default="plot.png",
                    help="path to output accuracy/loss plot")
    args = vars(ap.parse_args())
    return args


# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 100
INIT_LR = 1e-3
BS = 32
CLASS_NUM = 258  # remember to add 1
norm_size = 224  # 224 299


def get_img(imagePaths):
    data = []
    labels = []
    # loop over the input images
    for imagePath in imagePaths:
        image = cv2.imread(imagePath)
        image = cv2.resize(image, (norm_size, norm_size))
        image = img_to_array(image)
        data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = int((imagePath.split(os.path.sep)[-2]).split('.')[0])
        labels.append(label)
        gc.collect()

    # scale the raw pixel intensities to the range [0, 1]
    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    # convert the labels from integers to vectors
    labels = to_categorical(labels, num_classes=CLASS_NUM)
    return data, labels


def load_data(path):
    imagePaths = sorted(list(paths.list_images(path)))
    random.seed(42)
    random.shuffle(imagePaths)
    while 1:
        for i in range(0, len(imagePaths), BS):
            data, labels = get_img(imagePaths[i:i + BS])
            yield data, labels


def train(args):
    # initialize the model
    print("[INFO] compiling model...")

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    model = ResnetBuilder.build_resnet_18((3, norm_size, norm_size), CLASS_NUM)

    if os.path.exists('my_model_weights.h5'):
        print("[INFO] import model weights...")
        model.load_weights('my_model_weights.h5')

    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt,
                  metrics=["accuracy"])

    # train the network
    print("[INFO] training network...")

    H = model.fit_generator(load_data(args["dataset_train"]),
     validation_data=load_data(args["dataset_test"]),  steps_per_epoch=1000,
                            epochs=EPOCHS, verbose=1, validation_steps=80)

    # save the model to disk
    print("[INFO] serializing network...")
    model.save(args["model"])
    model.save_weights('my_model_weights.h5')

    # plot the training loss and accuracy
    plt.style.use("ggplot")
    plt.figure()
    N = EPOCHS
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["acc"], label="train_acc")
    plt.plot(np.arange(0, N), H.history["val_acc"], label="val_acc")
    plt.title("Training Loss and Accuracy on image classifier")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(args["plot"])


# python train.py --dataset_train ./data/train --dataset_test ./data/test --model traffic_sign.model
if __name__ == '__main__':
    args = args_parse()
    # construct the image generator for data augmentation
    train(args)
