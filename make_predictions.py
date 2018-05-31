import csv, os

from keras.models import load_model
from keras.utils import to_categorical
from keras import backend as K

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

    num_models = 20

    pac_file = open(data_dir + 'predictions.csv', 'w')
    pacwriter = csv.writer(pac_file)

    pacwriter.writerow(['PAC_ID', 'Prediction'])

    predictions = np.empty((num_models, len(subjects)))

    for model_num in range(num_models):
        model_filename = 'pac_model_' + str(model_num) + '.hdf5'
        model = load_model(data_dir + model_filename)

        print('model', model_num)
        for j, subj in enumerate(subjects):

            img = nib.load(data_dir + subj['id'] + '.nii').get_data()
            meta = np.empty((1, 7))
            meta[0, 0:3] = to_categorical(subj['site'] - 1, num_classes=3)
            meta[0, 3] = subj['age'] / 100
            meta[0, 4:6] = to_categorical(subj['gender'] - 1, num_classes=2)
            meta[0, 6] = subj['tiv'] / 2000

            prediction = model.predict([img[np.newaxis, ..., np.newaxis], meta])[0]
            predictions[model_num, j] = prediction[1]

        K.clear_session()

    for i in range(len(subjects)):
        subj_pred = predictions[:, i]

        print('Predictions:', subj_pred, np.sum(subj_pred) / num_models)

        if np.sum(subj_pred) / num_models >= 0.5:
            final_prediction = 1
        else:
            final_prediction = 0

        pacwriter.writerow([subjects[i]['id'], final_prediction])

    pac_file.close()
