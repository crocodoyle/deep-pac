import numpy as np
from scipy.spatial.distance import euclidean

import os, sys, time, csv, subprocess, pickle, h5py

from sklearn.neighbors import KDTree

import nibabel as nib
from nibabel.processing import resample_from_to
from skimage.exposure import equalize_hist

data_dir = os.path.expanduser('~/pac2018_root/')
output_dir = os.path.expanduser('~/pac2018_root/')

output_file = output_dir + 'pac2018.hdf5'

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#image_size = (121, 145, 121)
target_size = (152, 152, 152)

# taken from DLTK
def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float32)
    ret = (image - np.min(image))
    ret /= (np.max(image) + 0.000001)
    return ret

# taken from DLTK
def resize_image_with_crop_or_pad(image, img_size=(64, 64, 64), **kwargs):
    """Image resizing. Resizes image by cropping or padding dimension
     to fit specified size.
    Args:
        image (np.ndarray): image to be resized
        img_size (list or tuple): new image size
        kwargs (): additional arguments to be passed to np.pad
    Returns:
        np.ndarray: resized image
    """

    assert isinstance(image, (np.ndarray, np.generic))
    assert (image.ndim - 1 == len(img_size) or image.ndim == len(img_size)), \
        'Example size doesnt fit image size'

    # Get the image dimensionality
    rank = len(img_size)

    # Create placeholders for the new shape
    from_indices = [[0, image.shape[dim]] for dim in range(rank)]
    to_padding = [[0, 0] for dim in range(rank)]

    slicer = [slice(None)] * rank

    # For each dimensions find whether it is supposed to be cropped or padded
    for i in range(rank):
        if image.shape[i] < img_size[i]:
            to_padding[i][0] = (img_size[i] - image.shape[i]) // 2
            to_padding[i][1] = img_size[i] - image.shape[i] - to_padding[i][0]
        else:
            from_indices[i][0] = int(np.floor((image.shape[i] - img_size[i]) / 2.))
            from_indices[i][1] = from_indices[i][0] + img_size[i]

        # Create slicer object to crop or leave each dimension
        slicer[i] = slice(from_indices[i][0], from_indices[i][1])

    # Pad the cropped image to extend the missing dimension
    return np.pad(image[slicer], to_padding, **kwargs)


def count_pac2018(input_path, label_file):
    with open(os.path.join(input_path, label_file)) as label_file:
        pac_reader = csv.reader(label_file)
        pac_reader.__next__()

        num_subjects = 0
        for line in list(pac_reader):
            try:
                label = line[1]
                if len(label) < 1:
                    raise Exception()
                t1_filename = line[0][:] + '.nii'
                t1_data = nib.load(input_path + t1_filename)
                num_subjects += 1
            except:
                print('missing', t1_filename)

    return num_subjects


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

def parse_regional_GMD(subjects):
    file_name = 'PAC2018Covariates_and_regional_GMD.csv'
    region_reader = csv.reader(open(data_dir + file_name))

    lines = list(region_reader)[1:]

    subjects_with_regional_GMD = []

    for subject in subjects:
        for line in lines:
            if subject['id'] in line[0]:
                subject['regional_GMD_mean'] = np.asarray(line[5:251], dtype='float32')
                subject['regional_GMD_var'] = np.asarray(line[252:], dtype='float32')
                subjects_with_regional_GMD.append(subject)

    return subjects_with_regional_GMD

if __name__ == "__main__":

    n_pac2018 = count_pac2018(data_dir, 'PAC2018Covariates_and_regional_GMD.csv')
    n_rois = 246

    print('Subjects with labels:')
    print('PAC2018:', n_pac2018)

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('id', (n_pac2018,), dtype='int32')
        f.create_dataset('GMD', (n_pac2018, target_size[0], target_size[1], target_size[2]), dtype='float32')

        f.create_dataset('regional_GMD_mean', (n_pac2018, n_rois), dtype='float32')
        f.create_dataset('regional_GMD_var', (n_pac2018, n_rois), dtype='float32')

        f.create_dataset('label', (n_pac2018,), dtype='uint8')
        f.create_dataset('orig_label', (n_pac2018,), dtype='uint8')
        f.create_dataset('one_hot_label', (n_pac2018, 2), dtype='uint8')

        f.create_dataset('site', (n_pac2018,), dtype='uint8')

        f.create_dataset('age', (n_pac2018,), dtype='uint8')
        f.create_dataset('gender', (n_pac2018,), dtype='uint8')

        f.create_dataset('tiv', (n_pac2018,), dtype='float32')

        subjects = parse_covariates()
        subjects = parse_sites(subjects)
        subjects = parse_regional_GMD(subjects)

        for i, subject in enumerate(subjects):
            gmd_filepath = data_dir + subject['id'] + '.nii'
            gmd_img = nib.load(gmd_filepath).get_data()

            gmd_img = resize_image_with_crop_or_pad(gmd_img, target_size, mode='constant')

            f['id'][i] = int(subject['id'][8:])
            f['GMD'][i, ...] = gmd_img
            f['regional_GMD_mean'][i, ...] = subject['regional_GMD_mean']
            f['regional_GMD_var'][i, ...] = subject['regional_GMD_var']
            if subject['label'] == 2:
                f['label'][i] = 1
                f['one_hot_label'][i] = [0, 1]
            else:
                f['label'][i] = 0
                f['one_hot_label'][i] = [1, 0]
            f['orig_label'][i] = subject['label']
            f['site'][i] = subject['site']
            f['age'][i] = subject['age']
            f['gender'][i] = subject['gender']
            f['tiv'][i] = subject['tiv']
