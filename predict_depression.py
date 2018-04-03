import keras

import sklearn

import os, csv, pickle

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import h5py, pickle, csv, os, h5py

import nibabel as nib

from keras.utils import to_categorical

from models import convnet

data_dir = '/data1/users/adoyle/PAC2018/'
image_size = (256, 256, 256)


def batch_gen(f, labels, features):
    images = f['MRI']

    indices = range(len(labels))

    while True:
        np.random.shuffle(indices)

        for index in indices:
            yield (images[index:index+1, ...], labels[index:index+1, ...])

if __name__ == '__main__':
    print('PAC 2018')



    example = data_dir + subjects[0]['id'] + '.nii'
    image_size = nib.load(example).shape

    print('Image size:', image_size)

    features = np.zeros((len(subjects), 4))
    labels = np.zeros((len(subjects), 2))

    for i, sub in enumerate(subjects):
        features[i, 0] = sub['tiv']
        features[i, 1] = sub['site']
        features[i, 2] = sub['gender']
        features[i, 3] = sub['age']

        if sub['label'] == 1:
            labels[i, :] = [1, 0]
        else:
            labels[i, :] = [0, 1]

    f = hdf5_smash(subjects)

    model = convnet(image_size)
    model.fit_generator(batch_gen(f, labels, features), steps_per_epoch=None, epochs=10, validation_data=0.1, shuffle=False)