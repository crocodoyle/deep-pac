from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization
from keras.layers import Conv3DTranspose, Reshape, UpSampling3D, Input
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

workdir = os.path.expanduser('~/pac2018_root/')
data_file = 'pac2018.hdf5'

image_size = (152, 152, 152)  # (121, 145, 121)
input_size = (152, 152, 152, 1)


def pac2018_3d_cae_model():
    conv_size = (3, 3, 3)
    pool_size = (2, 2, 2)

    # 3D Convolutional Auto-Encoder
    inputs = Input(shape=input_size)

    x = Conv3D(8, conv_size, activation='relu')(inputs)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(8, conv_size, activation='relu')(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    encoder = Conv3D(8, conv_size, activation='relu')(x)

    print("Shape of encoder", K.int_shape(encoder))

    # 3D Convolutional Auto-Decoder
    x = Conv3DTranspose(8, conv_size, activation='relu')(encoder)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(8, conv_size, activation='relu')(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(8, conv_size, activation='relu')(x)

    decoder = Conv3DTranspose(1, conv_size, activation='sigmoid')(x)

    print("Shape of decoder", K.int_shape(decoder))

    autoencoder = Model(inputs, decoder)

    autoencoder.compile(loss='binary_crossentropy', optimizer='adam', metrics=["accuracy"])

    return autoencoder


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
                  metrics=["accuracy"])

    return model


def batch(indices, f):
    images = f['GMD']
    labels = f['label']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                label = labels[index]
                if label == 1:
                    one_hot_label = [1, 0]
                elif label == 2:
                    one_hot_label = [0, 1]

                yield (np.reshape(images[index, ...], input_size)[np.newaxis, ...], one_hot_label[np.newaxis, ...])
            except:
                yield (np.reshape(images[index, ...], input_size)[np.newaxis, ...])


def batch_cae(indices, f):
    images = f['GMD']
    labels = f['GMD']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                yield (np.reshape(images[index, ...], input_size)[np.newaxis, ...],
                       np.reshape(labels[index, ...], input_size)[np.newaxis, ...])
            except:
                yield (np.reshape(images[index, ...], input_size)[np.newaxis, ...])


def plot_metrics(hist, results_dir):
    epoch_num = range(len(hist.history['acc']))
    # train_error = np.subtract(1, np.array(hist.history['acc']))
    # test_error = np.subtract(1, np.array(hist.history['val_acc']))

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

    f = h5py.File(workdir + data_file, 'r')
    images = f['GMD']
    labels = f['labels']

    pac2018_indices = list(range(len(images)))

    # 3D CAE
    print('number of samples in dataset:', images.shape[0])

    train_indices = pac2018_indices
    train_labels = labels

    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train, other in skf.split(train_indices, train_labels):
        train_indices = train
        validation_indices = other[::2]  # even
        test_indices = other[1::2]  # odd

    print('train:', train_indices)
    print('val:', validation_indices)
    print('test:', test_indices)

    # define model
    model = pac2018_3d_cae_model()

    # print summary of model
    model.summary()

    num_epochs = 10

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

    model.load_weights(workdir + 'best_3d_cae_model.hdf5')
    model.save(results_dir + '3d_cae_model.hdf5')

    metrics = model.evaluate_generator(batch_cae(test_indices, f), len(test_indices))

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(results_dir + 'test_metrics', 'wb'))

    plot_metrics(hist, results_dir)

    print('This experiment brought to you by the number:', experiment_number)