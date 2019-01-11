import sys
import os
import io
from functools import wraps
import time
import pickle
import argparse

import numpy as np
import pandas as pd
from shutil import copy
from keras import optimizers, callbacks
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPool2D,
    Activation
)
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from keras.utils import plot_model


CONFIGS = {
    'setup_path': 'Dataset',
    "percent": 0.8,
    'common': {
        'image': {
            'width': 200,
            'height': 200,
        },
    },
    'model': {
        'path': './models/model.h5',
        'weights_path': './models/weights.h5',
    },
    'prediction': {
        'path': 'Prediction'
    },
    'test': {
        'path': 'Dataset/test'
    },
    'train': {
        'path': 'Dataset/train',
    },
    "tensor_board_log_dir": "./tf-log/",
}


def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = f(*args, **kwargs)
        end = time.time()
        duration = end - start

        print("Execution Time: ")
        if duration < 60:
            print("{} seconds".format(duration))
        elif duration > 60 and duration < 3600:
            duration = duration / 60
            print("{} minutes".format(duration))
        else:
            duration = duration / (60 * 60)
            print("{} hours".format(duration))

        return result

    return wrapper

def create_directory_if_not_exists(dirpath):
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
        return False

    return True


def save_model_as_pickle(history, filename="history.pckl"):
    with io.open(filename, "wb") as f:
        pickle.dump(history.history, f)


def save_model(
    model,
    model_path=CONFIGS["model"]["path"],
    model_weights_path=CONFIGS["model"]["weights_path"],
):
    """
    saving models for predictions for different images (not in the training and test set)
    """

    check_model_files(model_path, model_weights_path)

    model.save(model_path)
    model.save_weights(model_weights_path)


def check_model_files(model_path, model_weights_path):
    dirpath = os.path.dirname(model_path)
    create_directory_if_not_exists(dirpath)

    dirpath = os.path.dirname(model_weights_path)
    create_directory_if_not_exists(dirpath)


def check_path_exists(dirpath):
    pwd = os.getcwd()
    abs_path = os.path.join(pwd, dirpath)
    if not os.path.exists(abs_path):
        print("You must create {} path".format(abs_path))
        sys.exit(1)

    return abs_path

@timing
def training(
    epochs=16,
    img_width=CONFIGS["common"]["image"]["width"],
    img_height=CONFIGS["common"]["image"]["height"],
    batch_size=32,
    steps_per_epoch=32,
    nb_filters1=8,
    nb_filters2=16,
    pool_size=2,
    classes_num=10,
    lr=0.0004,
):
    dataset_abs_path = check_path_exists(CONFIGS["setup_path"])
    check_path_exists(CONFIGS["train"]["path"])
    check_path_exists(CONFIGS["test"]["path"])

    # conv => max pool => dropout => conv => max pool => dropout => fully connected (2 layer)
    # Dropout: Dropout is a technique where randomly selected neurons are ignored during training
    model = Sequential()
    #
    model.add(
        Conv2D(
            filters=8,
            kernel_size=(5, 5),
            padding="Same",
            activation="relu",
            input_shape=(img_width, img_height, 3),
        )
    )
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # there is no need to write input_shape as parameter bc we are working on same model as the last step
    model.add(Conv2D(filters=16, kernel_size=(3, 3), padding="Same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))
    # fully connected
    # to perform a ANN action , input data needs to be flattened
    model.add(Flatten())
    # adding hidden layer
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    # adding output layer
    model.add(Dense(classes_num, activation="softmax"))

    # Compile the model
    # categorical_crossentropy is like binary entropy but its more than 2 classes(binary 0-1)
    # loss parameters performs update of weights so the error rate balances.
    model.compile(
        loss="categorical_crossentropy",
        optimizer=optimizers.RMSprop(lr=lr),
        metrics=["accuracy"],
    )

    # Data Augmentation
    # To avoid overfitting problem, we need to expand artificially our dataset
    # Alter the training data with small transformations to reproduce the variations of images.
    # For example, the car is not centered The scale is not the same. The image is rotated.
    # ImageDataGenerator's parameters are hyperparameters. It can be set according to data and the results
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    target_size = (img_height, img_width)

    # images in dictionaries are flowing. Images are classifying as dictionary names(for training set)
    train_generator = train_datagen.flow_from_directory(
        CONFIGS["train"]["path"],
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )
    # for test set(as validation)
    validation_generator = test_datagen.flow_from_directory(
        CONFIGS["test"]["path"],
        target_size=target_size,
        batch_size=batch_size,
        class_mode="categorical",
    )

    """
    Tensorboard log
    """
    tb_cb = callbacks.TensorBoard(
        log_dir=CONFIGS["tensor_board_log_dir"], histogram_freq=0
    )
    cbks = [tb_cb]

    # Fit the model
    # We'll create our model and train
    # It'll validate the training data with validation_data
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        callbacks=cbks,
    )

    save_model_as_pickle(history)

    return model

 
@timing
def prediction():
    check_path_exists(CONFIGS["model"]["path"])
    check_path_exists(CONFIGS["model"]["weights_path"])


    check = create_directory_if_not_exists(CONFIGS['prediction']['path'])

    if not check:
        print("You have to add prediction images to '{}' folder".format(CONFIGS['prediction']['path']))
        sys.exit(1)

    img_width = CONFIGS["common"]["image"]["width"]
    img_height = CONFIGS["common"]["image"]["height"]
    target_size = (img_width, img_height)

    # Load the pre-trained models
    model = load_model(CONFIGS["model"]["path"])
    model.load_weights(CONFIGS["model"]["weights_path"])

    labels = [
        "airplanes",
        "butterfly",
        "car_side",
        "cellphone",
        "cup",
        "dolphin",
        "headphone",
        "laptop",
        "Motorbikes",
        "pizza",
    ]

    # Prediction Function
    def predict(file):
        x = load_img(file, target_size=target_size)
        x = img_to_array(x)
        x = np.expand_dims(x, axis=0)

        array = model.predict(x)
        result = array[0]
        answer = np.argmax(result)
        print("Answer: ", answer)
        print("Predicted: {}".format(labels[answer]))

        y_prob = model.predict(x)
        y_classes = y_prob.argmax(axis=-1)
        predicted_label = labels[answer]
        print(predicted_label)
        return answer

    # Walk the directory for every image
    for ret in os.walk(CONFIGS["prediction"]['path']):
        image_dir = ret[0]
        image_name = ret[2]
        for filename in filter(lambda f: not f.startswith("."), image_name):
            filepath = "{}/{}".format(image_dir, filename)
            print(filepath)
            result = predict(filepath)

def copy_files(path, dirs, files, mode):
    mode_abs_path = os.path.join(path, mode)
    print(mode_abs_path)
    if not os.path.exists(mode_abs_path):
        os.makedirs(mode_abs_path)
        for directory in dirs:
            __temp_path = os.path.join(path, directory)
            path_sub, dirs_sub, files_sub = next(os.walk(__temp_path))
            files_count = len(files_sub)
            indexLimit = int(files_count * CONFIGS["percent"])

            if mode == "train":
                limited_files = range(0, indexLimit)
            elif mode =="test":
                limited_files = range(indexLimit, files_count)
            else:
                print("Wrong Mode!!!")
                sys.exit(1)

            __temp_path_2 = os.path.join(mode_abs_path, directory)
            if not os.path.exists(__temp_path_2):
                os.makedirs(__temp_path_2)
                for i in limited_files:
                    __temp_path_3 = os.path.join(path_sub, files_sub[i])
                    copy(__temp_path_3, __temp_path_2)


def split_train_and_test_directories():
    dataset_abs_path = check_path_exists(CONFIGS["setup_path"])

    path, dirs, files = next(os.walk(dataset_abs_path))

    copy_files(path=path, dirs=dirs, files=files, mode="train")
    copy_files(path=path, dirs=dirs, files=files, mode="test")


def read_model_as_pickle(filepath="history.pckl"):
    with io.open(filepath, "rb") as f:
        history = pickle.load(f)
        return history

def plotting():
    #plot_model(model, to_file='model.png')
    # retrieve:    
    history = read_model_as_pickle()

    # Plot training & validation accuracy values
    plt.plot(history['acc'])
    plt.plot(history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history['loss'])
    plt.plot(history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Image Processing Project')
    subparsers = parser.add_subparsers(dest='action')

    # Parser Training
    parser_training = subparsers.add_parser('training')
    parser_training.add_argument('--epochs', type=int,default=16)
    parser_training.add_argument('--batch-size', type=int, default=32)
    parser_training.add_argument('--steps-per-epoch', type=int, default=32)

    # Parser Prediction
    parser_prediction = subparsers.add_parser('prediction')

    # Parser Splitting
    parser_splitting = subparsers.add_parser('splitting')

    # Parser Plotting
    parser_plotting = subparsers.add_parser('plotting')
    
    # Parse
    args = parser.parse_args()
   
    # Control
    print(args.action)
    if args.action == "training":
        model = training(
            epochs=args.epochs,
            batch_size=args.batch_size,
            steps_per_epoch=args.steps_per_epoch,
        )
        save_model(model)
    elif args.action == "prediction":
        prediction()
    elif args.action == "splitting":
        split_train_and_test_directories()
    elif args.action == "plotting":
        plotting()
    else:
        print("Please entry valid action! e.g training, prediction, splitting, plotting")

    # python3 main.py training --epochs 16 --batch_size 32 --steps_per_epoch 32
    # python3 main.py splitting
    # python3 main.py prediction
    
    

if __name__ == "__main__":
    main()