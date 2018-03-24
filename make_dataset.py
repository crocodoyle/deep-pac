import numpy as np
from scipy.spatial.distance import euclidean

import os, sys, time, csv, subprocess, pickle, h5py

from sklearn.neighbors import KDTree

import nibabel as nib
from nibabel.processing import resample_from_to
from skimage.exposure import equalize_hist

data_dir = '/data1/users/jmorse/'
output_dir = '/data1/users/jmorse/pac2018/'

output_file = output_dir + 'pac2018.hdf5'

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#image_size = (121, 145, 121)
target_size = (152, 152, 152)

# taken from DLTK
def normalise_zero_one(image):
    """Image normalisation. Normalises image to fit [0, 1] range."""

    image = image.astype(np.float16)
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
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        num_subjects = 0
        for line in list(qc_reader):
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


def make_pac2018(input_path, f, label_file, subject_index):
    n = count_pac2018(input_path, label_file)

    with open(os.path.join(input_path, label_file)) as label_file:
        qc_reader = csv.reader(label_file)
        qc_reader.__next__()

        lines = list(qc_reader)
        input_paths = [input_path] * len(lines)

        index = subject_index
        for line, input_path in zip(lines, input_paths):
            print('Processing subject %i of %i' % (index + 1, n))

            returned_index = make_pac2018_subject(line, index, input_path, f)
            if not returned_index == -1:
                index += 1

    return index


def make_pac2018_subject(line, subject_index, input_path, f):
    try:
        t1_filename = line[0][:] + '.nii'

        label = int(line[1])  # 0, 1, or 2

        one_hot = [0, 0]

        if label >= 1:
            one_hot = [0, 1]
        else:
            one_hot = [1, 0]

        f['label'][subject_index, :] = one_hot

        t1_data = nib.load(input_path + t1_filename).get_data()

        if not t1_data.shape == target_size:
            t1_data = resize_image_with_crop_or_pad(t1_data, img_size=target_size, mode='constant')

        #t1_data = equalize_hist(t1_data, mask=mask)

        f['MRI'][subject_index, ...] = normalise_zero_one(t1_data)
        f['dataset'][subject_index] = 'PAC2018'
        f['filename'][subject_index] = t1_filename

        # plt.imshow(t1_data[96, ...])
        # plt.axis('off')
        # plt.savefig(output_dir + t1_filename[:] + '.png', bbox_inches='tight', cmap='gray')

        return subject_index
    except Exception as e:
        print('Error:', e)
        return -1


def check_datasets():
    mri_sites = ['PAC2018']

    histograms = {}

    for site in mri_sites:
        histograms[site] = np.zeros(256, dtype='float32')

    bins = np.linspace(0.0, 1.0, 257)

    images = f['MRI']
    datasets = f['dataset']
    filenames = f['filename']

    for i, (img, dataset, filename) in enumerate(zip(images, datasets, filenames)):
        dataset = dataset.decode('UTF-8')
        filename = filename.decode('UTF-8')
        img = np.asarray(img, dtype='float32')

        try:
            histo = np.histogram(img, bins=bins)
            histograms[dataset] += histo[0]
            print(filename, dataset, np.mean(histo[0]), np.var(histo[0]))
        except:
            print('Error for', filename, 'in dataset', dataset)

    fig, axes = plt.subplots(len(mri_sites), 1, sharex=True, figsize=(4, 24))

    for i, site in enumerate(mri_sites):
        try:
            histograms[site] = np.divide(histograms[site], np.sum(histograms[site]))
            axes[i].plot(bins[:-1], histograms[site], lw=2, label=site)

            # axes[i].set_xlim([0, 1])

            axes[i].set_ylabel(site, fontsize=16)
            axes[i].set_xscale('log')
            axes[i].set_yscale('log')
        except:
            print('Problem normalizing histogram for', site)

    plt.tight_layout()
    plt.savefig(output_dir + 'dataset_histograms.png', bbox_inches='tight')

if __name__ == "__main__":
    n_pac2018 = count_pac2018(data_dir + '/pac2018/', 'PAC2018Covariates_and_regional_GMD.csv')

    print('Subjects with labels:')
    print('PAC2018:', n_pac2018)

    total_subjects = n_pac2018

    with h5py.File(output_file, 'w') as f:
        f.create_dataset('MRI', (total_subjects, target_size[0], target_size[1], target_size[2]), dtype='float32')
        f.create_dataset('label', (total_subjects, 2), dtype='uint8')
        dt = h5py.special_dtype(vlen=bytes)
        f.create_dataset('filename', (total_subjects,), dtype=dt)
        f.create_dataset('dataset', (total_subjects,), dtype=dt)

        print('Starting PAC2018...')
        next_index = make_pac2018(data_dir + '/pac2018/', f, 'PAC2018Covariates_and_regional_GMD.csv', 0)
        pac2018_indices = range(0, next_index)
        print('Last PAC2018 index:', next_index - 1)

        pickle.dump(pac2018_indices, open(output_dir + 'pac2018_indices.pkl', 'wb'))

        check_datasets()
