from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization
from keras.layers import Conv3DTranspose, Reshape, UpSampling3D
from keras.callbacks import ModelCheckpoint
from keras.optimizers import SGD, Adam

import numpy as np
import h5py
import pickle

import keras.backend as K
import os

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit

from custom_loss import sensitivity, specificity

workdir = '/data1/users/jmorse/pac2018/'
data_file = 'pac2018.hdf5'

image_size = (152, 152, 152)#(121, 145, 121)


def pac2018_3d_cae_model():
    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    model = Sequential()

    crop_size = (152, 152, 152)

    # Convolutional encoding
    model.add(Conv3D(8, conv_size, activation='relu', input_shape=(crop_size[0], crop_size[1], crop_size[2], 1)))
    model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(8, conv_size, activation='relu'))
    model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(8, conv_size, activation='relu'))

    # Dense layers
    #model.add(Flatten())

    #model.add(Dense(32*32*32, activation='relu'))

 #   model.add(Dense(125, activation='relu'))
 #   model.summary()
    #model.add(Reshape((32, 32, 32, 1)))

    # Deconvolutional decoding
    model.add(Conv3DTranspose(8, conv_size, activation='relu'))

    model.add(UpSampling3D(pool_size))
    model.add(Conv3DTranspose(8, conv_size, activation='relu'))

    model.add(UpSampling3D(pool_size))
    model.add(Conv3DTranspose(8, conv_size, activation='relu'))

    model.add(Conv3DTranspose(1, conv_size, activation='relu'))

    model.summary()
    # Reshape to original size
   # model.add(Reshape(crop_size))

    model.compile(loss='mean_absolute_error',
                  optimizer='adam',
                  metrics=["accuracy"])

    return model


def pac2018_3d_model():
    nb_classes = 2

    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    model = Sequential()

    model.add(Conv3D(4, conv_size, activation='relu', input_shape=(image_size[0], image_size[1], image_size[2], 1)))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Conv3D(8, conv_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(16, conv_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Conv3D(16, conv_size, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(32, conv_size, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Conv3D(32, conv_size, activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(32, conv_size, activation='relu'))
    model.add(Dropout(0.4))
    model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(64, conv_size, activation='relu'))
    model.add(Dropout(0.4))
    # model.add(MaxPooling3D(pool_size=pool_size))

    model.add(Conv3D(8, (1, 1, 1), activation='relu'))

    model.add(Flatten())
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.5))

    # model.add(Conv3D(256, (1, 1, 1), activation=('relu')))
    # model.add(Dropout(0.5))
    # model.add(Conv3D(nb_classes, (1, 1, 1), activation=('relu')))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["accuracy", sensitivity, specificity])

    return model


def batch(indices, f):
    images = f['MRI']
    labels = f['label']    #already in one-hot

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                # print(images[index, ...][np.newaxis, ...].shape)
                yield (np.reshape(images[index, ...], image_size + (1,))[np.newaxis, ...], labels[index, ...][np.newaxis, ...])
            except:
                yield (np.reshape(images[index, ...], image_size + (1,))[np.newaxis, ...])


def batch_cae(indices, f):
    images = f['MRI']
    labels = f['MRI']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                # print(images[index, ...][np.newaxis, ...].shape)
                yield (np.reshape(images[index, ...], image_size + (1,))[np.newaxis, ...],
                       np.reshape(labels[index, ...], image_size + (1,))[np.newaxis, ...])
            except:
                yield (np.reshape(images[index, ...], image_size + (1,))[np.newaxis, ...])


def plot_metrics(hist, results_dir):
    epoch_num = range(len(hist.history['acc']))
    train_error = np.subtract(1, np.array(hist.history['acc']))
    test_error  = np.subtract(1, np.array(hist.history['val_acc']))

    plt.clf()
    plt.plot(epoch_num, np.array(hist.history['acc']), label='Training Accuracy')
    plt.plot(epoch_num, np.array(hist.history['val_acc']), label="Validation Error")
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Error")
    plt.savefig(results_dir + 'results.png')
    plt.close()


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

    pac2018_indices = pickle.load(open(workdir + 'pac2018_indices.pkl', 'rb'))

    f = h5py.File(workdir + data_file, 'r')
    images = f['MRI']

    # 3D CAE
    print('number of samples in dataset:', images.shape[0])

    train_indices = pac2018_indices

    train_labels = np.zeros((len(train_indices), 2))
    print('labels shape:', train_labels.shape)

    good_subject_index = 0
    for index in train_indices:
        label = f['label'][index, ...]
        train_labels[good_subject_index, ...] = label
        good_subject_index += 1

    skf = StratifiedShuffleSplit(n_splits=1, test_size = 0.2)

    for train, other in skf.split(train_indices, train_labels):
        train_indices = train
        validation_indices = other[::2] # even
        test_indices = other[1::2] # odd

    print('train:', train_indices)
    print('test:', test_indices)

    # define model
    model = pac2018_3d_cae_model()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

    # print summary of model
    model.summary()

    num_epochs = 1

    model_checkpoint = ModelCheckpoint(workdir + 'best_3d_cae_model.hdf5',
                                       monitor="val_acc",
                                       save_best_only=True)

    hist = model.fit_generator(
        batch_cae(train_indices, f),
        len(train_indices),
        epochs=num_epochs,
        callbacks=[model_checkpoint],
        validation_data=batch_cae(validation_indices, f),
        validation_steps=len(validation_indices)
    )

    model.load_weights(results_dir + 'best_3d_cae_model.hdf5')
    model.save(results_dir + '3d_cae_model.hdf5')

    metrics = model.evaluate_generator(batch_cae(test_indices, f), len(test_indices))

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(results_dir + 'test_metrics', 'wb'))

    plot_metrics(hist, results_dir)

    print('This experiment brought to you by the number:', experiment_number)

    # CNN

    print('number of samples in dataset:', images.shape[0])

    train_indices = pac2018_indices

    train_labels = np.zeros((len(train_indices), 2))
    print('labels shape:', train_labels.shape)

    good_subject_index = 0
    for index in train_indices:
        label = f['label'][index, ...]
        train_labels[good_subject_index, ...] = label
        good_subject_index += 1

    skf = StratifiedShuffleSplit(n_splits=1, test_size = 0.2)

    for train, other in skf.split(train_indices, train_labels):
        train_indices = train
        validation_indices = other[::2] # even
        test_indices = other[1::2] # odd

    print('train:', train_indices)
    print('test:', test_indices)

    # define model
    model = pac2018_3d_model()

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)

    # print summary of model
    model.summary()

    num_epochs = 300

    model_checkpoint = ModelCheckpoint( workdir + 'best_3d_cnn_model.hdf5',
                                        monitor="val_acc",
                                        save_best_only=True)

    hist = model.fit_generator(
        batch(train_indices, f),
        len(train_indices),
        epochs=num_epochs,
        callbacks=[model_checkpoint],
        validation_data=batch(validation_indices, f),
        validation_steps=len(validation_indices)
    )

    model.load_weights(results_dir + 'best_3d_cnn_model.hdf5')
    model.save(results_dir + '3d_cnn_model.hdf5')

    metrics = model.evaluate_generator(batch(test_indices, f), len(test_indices))

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(results_dir + 'test_metrics', 'wb'))

    plot_metrics(hist, results_dir)

    print('This experiment brought to you by the number:', experiment_number)