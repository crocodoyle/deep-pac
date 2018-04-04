from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization
from keras.layers import Conv3DTranspose, Reshape, UpSampling3D, Input, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
from keras.optimizers import SGD, Adam

import nibabel as nib
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

image_size = (152, 152, 152)    # (121, 145, 121) is original size, resized in make_dataset.py
input_size = (152, 152, 152, 1) # Input size to 3D CAE

conv_size = (3, 3, 3)
pool_size = (2, 2, 2)

cae_loss_function = 'binary_crossentropy'
cae_optimizer = 'adam'
cae_metrics = ['accuracy']


def cae_encoder(trainable=True):
    # 3D Convolutional Auto-Encoder
    inputs = Input(shape=input_size)

    x = Conv3D(8, conv_size, activation='relu', trainable=trainable)(inputs)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(8, conv_size, activation='relu', trainable=trainable)(x)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(8, conv_size, activation='relu', trainable=trainable)(x)

    encoder = Flatten(name='encoded', trainable=trainable)(x)

    model = Model(inputs=inputs, outputs=encoder)

    return model


def cae_decoder():
    # 3D Convolutional Auto-Decoder
    inputs = Input(shape=(34*34*34*8, 1))

    x = Reshape((34, 34, 34, 8))(inputs)

    x = Conv3DTranspose(8, conv_size, activation='relu')(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(8, conv_size, activation='relu')(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(8, conv_size, activation='relu')(x)

    decoder = Conv3DTranspose(1, conv_size, activation='sigmoid', name='decoded')(x)

    model = Model(inputs=inputs, output=decoder)

    return model


def cae_model():

    encoder = cae_encoder()
    decoder = cae_decoder()

    autoencoder = Model(encoder.input, decoder(encoder.output))

    autoencoder.compile(loss=cae_loss_function, optimizer=cae_optimizer, metrics=cae_metrics)

    return autoencoder


def cnn_model():
    nb_classes = 2

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


def visualize_cae(results_dir, model, indices, f):
    images = f['GMD']

    acc_index = 0
    k = 0
    for mn in model.metrics_names:
        if mn == 'acc':
            acc_index = k
        k += 1

    min_acc = 1.01
    min_id = -1
    max_acc = 0.00
    max_id = -1
    min_img = []
    max_img = []

    # Find best and worst images
    for index in indices:
        img = np.reshape(images[index, ...], input_size)[np.newaxis, ...]
        eval = model.evaluate(img, img)
        pred = model.predict(img)

        acc = eval[acc_index]

        if acc > max_acc:
            max_acc = acc
            max_id = index
            max_img = pred
        if acc < min_acc:
            min_acc = acc
            min_id = index
            min_img = pred

    worst_img = nib.Nifti1Image(min_img, affine=None)
    nib.nifti1.save(worst_img, results_dir + 'worst_img.nii')
    best_img = nib.Nifti1Image(max_img, affine=None)
    nib.nifti1.save(best_img, results_dir + 'best_img.nii')


def plot_metrics(hist, results_dir, fn):
    epoch_num = range(len(hist.history['acc']))
    # train_error = np.subtract(1, np.array(hist.history['acc']))
    # test_error = np.subtract(1, np.array(hist.history['val_acc']))

    plt.clf()
    plt.plot(epoch_num, np.array(hist.history['acc']), label='Training Accuracy')
    plt.plot(epoch_num, np.array(hist.history['val_acc']), label="Validation Error")
    plt.legend(shadow=True)
    plt.xlabel("Training Epoch Number")
    plt.ylabel("Error")
    plt.savefig(results_dir + fn)
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


def load_cae_test(path):
    # Load Encoder with model weights
    encoder = cae_encoder(False)
    encoder.load_weights(path, by_name=True)
    encoder.compile(loss=cae_loss_function, optimizer=cae_optimizer, metrics=cae_metrics)
    encoder.summary()


if __name__ == "__main__":
    results_dir, experiment_number = setup_experiment(workdir)

    # load_cae_test(workdir + 'experiment-2/3d_cae_model.hdf5')

    # quit()

    f = h5py.File(workdir + data_file, 'r')
    images = f['GMD']
    labels = f['label']

    pac2018_indices = list(range(len(images)))

    # 3D CAE
    print('number of samples in dataset:', images.shape[0])

    train_indices = pac2018_indices
    train_labels = labels

    skf = StratifiedShuffleSplit(n_splits=1, test_size=0.2)

    for train, other in skf.split(train_indices, train_labels):
        train_indices = train
        validation_indices = other[::2]  # even
        test_indices = other[1::2]       # odd

    print('train:', train_indices)
    print('val:', validation_indices)
    print('test:', test_indices)

    # define model
    model = cae_model()

    # print summary of model
    model.summary()

    num_epochs = 1

    model_checkpoint = ModelCheckpoint(workdir + 'best_3d_cae_model.hdf5',
                                       monitor="val_acc",
                                       save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_acc', patience=2)

    hist = model.fit_generator(
        batch_cae(train_indices, f),
        len(train_indices),
        epochs=num_epochs,
        callbacks=[model_checkpoint, early_stopping],
        validation_data=batch_cae(validation_indices, f),
        validation_steps=len(validation_indices)
    )

    model.load_weights(workdir + 'best_3d_cae_model.hdf5')
    model.save(results_dir + '3d_cae_model.hdf5')

    metrics = model.evaluate_generator(batch_cae(test_indices, f), len(test_indices))

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(results_dir + 'test_metrics_3d_cae', 'wb'))

    plot_metrics(hist, results_dir, 'results_3d_cae.png')

    visualize_cae(results_dir, model, test_indices, f)

    print('This experiment brought to you by the number:', experiment_number)
