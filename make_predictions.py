import csv, os

from keras.models import load_model
from keras.utils import to_categorical

import nibabel as nib
import numpy as np


data_dir = os.path.expanduser('~/pac2018_root/')
model_filename = 'pac2018_model.hdf5'


def parse_covariates(data_dir):
    file_name = 'PAC2018_Covariates_Testset.csv'

    covariates_reader = csv.reader(open(data_dir + file_name))

    lines = list(covariates_reader)[1:]

    subjects = []

    for line in lines:
        subject = {}
        pac_id = line[0]
        # label = int(line[1])

        site = int(line[1])
        age = int(line[2])
        gender = int(line[3])
        tiv = float(line[4])

        subject['id'] = pac_id
        # subject['label'] = label
        subject['site'] = site
        subject['age'] = age
        subject['gender'] = gender
        subject['tiv'] = tiv

        subjects.append(subject)

    return subjects

if __name__ == '__main__':
    subjects = parse_covariates(data_dir)

    model = load_model(data_dir + model_filename)
    model.summary()

    pacwriter = csv.writer(open(data_dir + 'predictions.csv', 'w'))

    pacwriter.writerow(['PAC_ID', 'Prediction'])
    for subj in subjects:

        img = nib.load(data_dir + subj['id'] + '.nii').get_data()
        meta = np.empty((1, 7))
        meta[0, 0:3] = to_categorical(subj['site'] - 1, num_classes=3)
        meta[0, 3] = subj['age'] / 100
        meta[0, 4:6] = to_categorical(subj['gender'] - 1, num_classes=2)
        meta[0, 6] = subj['tiv'] / 2000

        print(img.shape)

        prediction = model.predict([img[np.newaxis, ..., np.newaxis], meta])[0]

        pacwriter.writerow([subj['id'], np.argmax(prediction)])


