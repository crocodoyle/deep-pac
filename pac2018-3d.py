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
cae_model_file = workdir + 'experiment-1/3d_cae_model.hdf5'
data_file = 'pac2018.hdf5'

image_size = (152, 152, 152)    # (121, 145, 121) is original size, resized in make_dataset.py
input_size = (152, 152, 152, 1) # Input size to 3D CAE

conv_size = (3, 3, 3)
pool_size = (2, 2, 2)

cae_loss_function = 'binary_crossentropy'
cae_optimizer = 'adam'
cae_metrics = ['accuracy']
activation_function = 'relu'

train_stacked_model = True


def cae_encoder(trainable=True):
    # 3D Convolutional Auto-Encoder
    inputs = Input(shape=input_size)

    x = Conv3D(8, conv_size, activation=activation_function, trainable=trainable)(inputs)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(8, conv_size, activation=activation_function, trainable=trainable)(x)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(8, conv_size, activation=activation_function, trainable=trainable)(x)

    encoder = Flatten(name='encoded', trainable=trainable)(x)

    model = Model(inputs=inputs, outputs=encoder)

    return model


def cae_decoder():
    # 3D Convolutional Auto-Decoder
    inputs = Input(shape=(34*34*34*8, 1))

    x = Reshape((34, 34, 34, 8))(inputs)

    x = Conv3DTranspose(8, conv_size, activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(8, conv_size, activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(8, conv_size, activation=activation_function)(x)

    decoder = Conv3DTranspose(1, conv_size, activation='sigmoid', name='decoded')(x)

    model = Model(inputs=inputs, output=decoder)

    return model


def cae_model():
    encoder = cae_encoder()
    decoder = cae_decoder()

    autoencoder = Model(encoder.input, decoder(encoder.output))

    autoencoder.compile(loss=cae_loss_function, optimizer=cae_optimizer, metrics=cae_metrics)

    return autoencoder


def top_level_classifier():
    inputs = Input(shape=(34*34*34*8, 1))
    x = Flatten()(inputs)

    x = Dense(20, activation=activation_function)(x)
    x = Dense(10, activation=activation_function)(x)
    x = Dense(1, activation='softmax')(x)

    model = Model(inputs=inputs, output=x)

    return model


def load_cae(path):
    # Load Encoder with model weights
    encoder = cae_encoder(False)
    encoder.load_weights(path, by_name=True)
    encoder.compile(loss=cae_loss_function, optimizer=cae_optimizer, metrics=cae_metrics)
    return encoder


def cae_classifier_model():
    encoder = load_cae(cae_model_file)
    top_level = top_level_classifier()

    model = Model(encoder.input, top_level(encoder.output))

    model.compile(loss='binary_crossentropy',
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

                #if label == 2:
                #    label = 0

                # one_hot_label = [0, 1]
                # if label == 1:
                #    one_hot_label = [1, 0]

                yield (np.reshape(images[index, ...], input_size)[np.newaxis, ...], label[np.newaxis, ...])
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


def get_nifti_data():
    gmd_filepath = workdir + 'PAC2018_0001.nii'
    gmd_img = nib.load(gmd_filepath)

    return gmd_img.affine, gmd_img.header, gmd_img.get_data()


def visualize_cae(results_dir, model, indices, f):
    from make_dataset import resize_image_with_crop_or_pad
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
    min_orig = []
    max_img = []
    max_orig = []

    # Find best and worst images
    for index in indices:
        img = np.reshape(images[index, ...], input_size)[np.newaxis, ...]
        eval = model.evaluate(img, img)
        pred = model.predict(img)

        acc = eval[acc_index]

        if acc > max_acc:
            max_acc = acc
            max_id = index
            max_img = np.reshape(pred, image_size)
            max_orig = np.reshape(img, image_size)
        if acc < min_acc:
            min_acc = acc
            min_id = index
            min_img = np.reshape(pred, image_size)
            min_orig = np.reshape(img, image_size)

    aff = np.eye(4)

    img_data = get_nifti_data()
    test_img = nib.Nifti1Image(resize_image_with_crop_or_pad(img_data[2], image_size, mode='constant'), affine=aff)
    nib.save(test_img, results_dir + 'test_img.nii')

    worst_img = nib.Nifti1Image(min_img, affine=aff)
    print("Sum of worst image is %f" % np.sum(min_img))
    nib.nifti1.save(worst_img, results_dir + 'worst_img.nii')
    worst_img = nib.Nifti1Image(min_orig, affine=aff)
    print("Sum of worst image original is %f" % np.sum(min_orig))

    nib.nifti1.save(worst_img, results_dir + 'worst_img_orig.nii')
    best_img = nib.Nifti1Image(max_img, affine=aff)
    print("Sum of best image is %f" % np.sum(max_img))
    nib.nifti1.save(best_img, results_dir + 'best_img.nii')
    best_img = nib.Nifti1Image(max_orig, affine=aff)
    nib.nifti1.save(best_img, results_dir + 'best_img_orig.nii')
    print("Sum of best image original is %f" % np.sum(max_orig))
    print("Original size is ", max_orig.shape)
    print("NIFTI size is ", test_img.shape)


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


if __name__ == "__main__":
    results_dir, experiment_number = setup_experiment(workdir)

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

    if train_stacked_model:
        print("Training stacked classifier model")
        model = cae_classifier_model()
        best_model_filename = workdir + 'best_stacked_model.hdf5'
        model_filename = results_dir + 'stacked_model.hdf5'
        metrics_filename = results_dir + 'test_metrics_stacked'
        results_plot_filename = 'results_stacked.png'
        batch_func = batch
    else:
        print("Training 3D convolutional autoencoder")
        model = cae_model()
        best_model_filename = workdir + 'best_3d_cae_model.hdf5'
        model_filename = results_dir + '3d_cae_model.hdf5'
        metrics_filename = results_dir + 'test_metrics_3d_cae'
        results_plot_filename = 'results_3d_cae.png'
        batch_func = batch_cae

    # print summary of model
    model.summary()

    num_epochs = 100

    early_stopping = EarlyStopping(monitor='val_acc', patience=2)
    model_checkpoint = ModelCheckpoint(best_model_filename,
                                       monitor="val_acc",
                                       save_best_only=True)

    hist = model.fit_generator(
        batch_func(train_indices, f),
        len(train_indices),
        epochs=num_epochs,
        callbacks=[model_checkpoint, early_stopping],
        validation_data=batch_func(validation_indices, f),
        validation_steps=len(validation_indices)
    )

    model.load_weights(best_model_filename)
    model.save(model_filename)

    metrics = model.evaluate_generator(batch_func(test_indices, f), len(test_indices))

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(metrics_filename, 'wb'))

    plot_metrics(hist, results_dir, results_plot_filename)

    visualize_cae(results_dir, model, test_indices, f)

    print('This experiment brought to you by the number:', experiment_number)
