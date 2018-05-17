from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization, TimeDistributed
from keras.layers import Conv3DTranspose, Reshape, UpSampling3D, Input, Lambda, ZeroPadding3D, Cropping3D
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras import regularizers
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import class_weight
import nibabel as nib
import numpy as np
import h5py
import pickle
import matplotlib as mpl
import keras.backend as K
import os
import random

mpl.use('Agg')
import matplotlib.pyplot as plt

workdir = os.path.expanduser('~/pac2018_root/')

data_file = 'pac2018.hdf5'

image_size = (152, 152, 152)     # (121, 145, 121) is original size, resized in make_dataset.py
input_size = (152, 152, 152, 1)  # Input size to classifier

cae_loss_function = 'binary_crossentropy'
cae_optimizer = 'adam'
activation_function = 'relu'


def gmd_classifier():
    inputs = Input(shape=input_size)

    padding = 'same'
    strides = 2
    filters = 8
    cs = (4, 4, 4)
    dropout = 0.2

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(inputs)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(x)

    encoder = Flatten(name='encoded')(x)

    x = Dense(50, activation=activation_function)(encoder)

    x = Dropout(0.5)(x)

    x = Dense(10, activation=activation_function)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def batch(indices, f):
    images = f['GMD']
    labels = f['one_hot_label']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            label = labels[index]

            yield (np.reshape(images[index, ...], input_size)[np.newaxis, ...], label[np.newaxis, ...])


def test_gmd_classifier(model, test_indices, f):
    images = f['GMD']
    labels = f['one_hot_label']

    for i in test_indices:
        img = np.reshape(images[i, ...], input_size)[np.newaxis, ...]
        label = labels[i]

        output = model.predict(img)[0]
        #print(output)
        pred = np.argmax(output, axis=-1)  # axis=-1, last axis

        prediction = [1, 0]
        if pred == 1:
            prediction = [0, 1]

        if np.array_equal(prediction, label):
            print("%i C " % i, label, prediction)
        else:
            print("%i I " % i, label, prediction)


def setup_experiment(workdir):
    try:
        experiment_number = pickle.load(open(workdir + 'experiment_number.pkl', 'rb'))
        experiment_number += 1
    except:
        print('Couldnt find the file to load experiment number')
        experiment_number = 0

    print('This is experiment number:', experiment_number)

    results_dir = workdir + '/experiment-' + str(experiment_number) + '/'
    os.makedirs(results_dir)

    pickle.dump(experiment_number, open(workdir + 'experiment_number.pkl', 'wb'))

    return results_dir, experiment_number


if __name__ == "__main__":
    results_dir, experiment_number = setup_experiment(workdir)

    f = h5py.File(workdir + data_file, 'r')
    images = f['GMD']
    labels = f['label'].value
    one_hot_labels = f['one_hot_label'].value

    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    pac2018_indices = list(range(len(images)))

    print('number of samples in dataset:', images.shape[0])

    train_indices = pac2018_indices
    train_labels = labels

    sss_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    train_indices, validation_indices, test_indices = None, None, None

    for train_index, validation_index in sss_validation.split(np.zeros(len(labels)), labels):
        train_indices = train_index
        validation_indices = validation_index

    labs = labels[validation_indices, ...]

    for validation_index, test_index in sss_test.split(np.zeros(len(labs)), labs):
        validation_indices = validation_index
        test_indices = test_index

    print('train:', train_indices)
    print('val:', validation_indices)
    print('test:', test_indices)

    model = gmd_classifier()
    best_model_filename = results_dir + 'best_stacked_model.hdf5'
    model_filename = results_dir + 'stacked_model.hdf5'
    metrics_filename = results_dir + 'test_metrics_stacked'
    batch_func = batch
    monitor = 'val_categorical_accuracy'
    test_function = test_gmd_classifier
    zero_indices = np.where((one_hot_labels == (1, 0)).all(axis=1))[0]
    steps_per_epoch = len(train_indices)

    # print summary of model
    model.summary()

    num_epochs = 1200

    model_checkpoint = ModelCheckpoint(best_model_filename,
                                       monitor=monitor,
                                       save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs/' + str(experiment_number), histogram_freq=0, write_graph=True,
                              write_grads=True,
                              write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5,
                                  patience=10, min_lr=0.0000001)

    hist = model.fit_generator(
        batch_func(train_indices, f),
        steps_per_epoch,
        epochs=num_epochs,
        callbacks=[model_checkpoint, tensorboard, reduce_lr],
        validation_data=batch_func(validation_indices, f),
        validation_steps=len(validation_indices), class_weight=class_weights
    )

    model.load_weights(best_model_filename)
    model.save(model_filename)

    metrics = model.evaluate_generator(batch_func(test_indices, f), len(test_indices))

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(metrics_filename, 'wb'))

    test_function(model, test_indices, f)

    print('This experiment brought to you by the number:', experiment_number)
