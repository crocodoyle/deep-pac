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

data_dir = 'E:/brains/PAC2018/'
image_size = (256, 256, 256)


def parse_covariates():
    file_name = 'PAC2018_Covariates_Upload.csv'

    covariates_reader = csv.reader(open(data_dir + file_name))

    lines = list(covariates_reader)[1:]

    subjects = []

    for line in lines:
        subject = {}
        pac_id = line[0]
        label = int(line[1])

        age = int(line[2])
        gender = int(line[3])
        tiv = float(line[4])

        subject['id'] = pac_id
        subject['label'] = label
        subject['age'] = age
        subject['gender'] = gender
        subject['tiv'] = tiv

        subjects.append(subject)

    return subjects

def parse_sites(subjects):
    file_name = 'PAC2018_Sites.csv'
    sites_reader = csv.reader(open(data_dir + file_name))

    lines = list(sites_reader)[1:]

    subjects_with_site = []
    for subject in subjects:
        for line in lines:
            if subject['id'] in line[0]:
                subject['site'] = int(line[1])
                subjects_with_site.append(subject)

    return subjects_with_site

def hdf5_smash(subjects):
    f = h5py.File(data_dir + 'PAC.hdf5')
    f.create_dataset('images', (len(subjects), image_size[0], image_size[1], image_size[2], 1), dtype='float32')

    for i, subject in enumerate(subjects):
        f['images'] = nib.load(data_dir + subject['id'] + '.nii').get_data()

    return f

def batch_gen(f, labels, features):
    images = f['MRI']

    indices = range(len(labels))

    while True:
        np.random.shuffle(indices)

        for index in indices:
            yield (images[index:index+1, ...], labels[index:index+1, ...])

if __name__ == '__main__':
    print('PAC 2018')

    subjects = parse_covariates()
    subjects = parse_sites(subjects)

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