from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization, TimeDistributed
from keras.layers import Conv3DTranspose, Reshape, UpSampling3D, Input, Lambda, ZeroPadding3D, Cropping3D
from keras.layers import Conv2D, MaxPooling2D, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
from keras import regularizers
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.utils import class_weight
import nibabel as nib
import numpy as np
import h5py
import pickle
import matplotlib as mpl
import keras.backend as K
import os
import random

mpl.use('Agg')
import matplotlib.pyplot as plt

# Set seed for experiment replicability
#random_seed = 38493
#seed(random_seed)
#set_random_seed(random_seed)

workdir = os.path.expanduser('~/pac2018_root/')
cae_model_file = workdir + 'experiment-268/best_3d_cae_model.hdf5'  # 'experiment-1/3d_cae_model.hdf5' 62 is deep, 1 is shallow
strided_gmd_classifier = workdir + 'experiment-442/best_stacked_model.hdf5' # 350 2 mil, 348 8 mil
# 442 29000
data_file = 'pac2018.hdf5'
average_image_file = workdir + 'average_nifti.nii'

image_size = (152, 152, 152)     # (121, 145, 121) is original size, resized in make_dataset.py
input_size = (152, 152, 152, 1)  # Input size to 3D CAE

mean_input_size = (246, )
single_input_size = (1, )

conv_size = (3, 3, 3)
pool_size = (2, 2, 2)

cae_loss_function = 'binary_crossentropy'  # 'mean_squared_error' has 0 output for deepest networks
# bce worked when 19, 19, 19
cae_optimizer = 'adam'
activation_function = 'relu'

train_stacked_model = True

cae_filter_count = 8
cae_output_shape = (5, 5, 5, cae_filter_count*2)  # (34, 34, 34, 8)
cae_output_count = cae_output_shape[0]*cae_output_shape[1]*cae_output_shape[2]*cae_output_shape[3]

activity_reg = None
kernel_reg = None
bias_reg = None  # regularizers.l2(0.01)

layers_to_watch = ['classifier_input', 'output']

# custom adam optimizer and gmd_classifier: 0.89 test accuracy
# next step: add gmd mean/var and site/gender/age/tiv to gmd_classifier and retrain

# below are numbers for CAE

# 19, 19, 19, 96, bce (240): 0.0786 test loss

# 10, 10, 10, 64, bce (245): 0.0712 test loss (requires a zero padding layer)
# 5, 5, 5, 32, bce (246): 0.0755 test loss (requires a zero padding layer)

# 5, 5, 5, 64, bce (247): 0.0710 test loss (requires zero padding layer)
# 5, 5, 5, 64, bce (261): 0.0705 test loss (requires zero padding layer)
# 5, 5, 5, 70, bce (252): 0.0720 test loss (requires zero padding layer)
# 5, 5, 5, 8/16/32/64/128/8, bce (256): 0.0751 test loss (requires zero padding layer)

# 5, 5, 5, 16/32/64/128/256/16, bce (257): 0.0721 test loss (requires zero padding layer)

# 5, 5, 5, 96, bce (249): does not converge (requires zero padding layer)
# 5, 5, 5, 80, bce (250): does not converge
# 5, 5, 5, 75, bce (251): does not converge

# 19, 19, 19, 96, mse (241): had 0.0395 test loss BUT EMPTY reconstructed images


def mean_classifier():
    inputs1 = Input(shape=mean_input_size)
    inputs2 = Input(shape=mean_input_size)

    x1 = Dense(50, activation=activation_function)(inputs1)
    x2 = Dense(50, activation=activation_function)(inputs2)

    x = concatenate([x1, x2])
    x = Dense(50, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(10, activation=activation_function)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs1, inputs2], outputs=x)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=True)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def classifer_25D():
    input = Input(shape=input_size)
    base_cnn_model = InceptionV3(include_top=False)
    temporal_analysis = TimeDistributed(base_cnn_model)(input)
    conv3d_analysis = Conv3D(8, 3, 3, 3)(temporal_analysis)
    conv3d_analysis = Conv3D(8, 3, 3, 3)(conv3d_analysis)
    output = Flatten()(conv3d_analysis)
    output = Dense(2, activation="softmax")(output)

    model = Model(inputs=input, outputs=output)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def cae_encoder(trainable=True):
    # 3D Convolutional Auto-Encoder
    inputs = Input(shape=input_size)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function, padding='same', trainable=trainable)(inputs)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(cae_filter_count*2, conv_size, activation=activation_function, padding='same', trainable=trainable)(x)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(cae_filter_count*4, conv_size, activation=activation_function, padding='same', trainable=trainable)(x)
    x = ZeroPadding3D(padding=(1, 1, 1))(x)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(cae_filter_count*8, conv_size, activation=activation_function, padding='same', trainable=trainable)(x)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(cae_filter_count*16, conv_size, activation=activation_function, padding='same', trainable=trainable)(x)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(cae_filter_count*2, conv_size, activation=activation_function, padding='same', trainable=trainable)(x)

    encoder = Flatten(name='encoded', trainable=trainable)(x)

    model = Model(inputs=inputs, outputs=encoder)

    model.summary()

    return model


def cae_decoder():
    # 3D Convolutional Auto-Decoder
    inputs = Input(shape=(cae_output_count, 1))

    x = Reshape(cae_output_shape)(inputs)

    x = Conv3DTranspose(cae_filter_count*2, conv_size, padding='same', activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(cae_filter_count*16, conv_size, padding='same', activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(cae_filter_count*8, conv_size, padding='same', activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(cae_filter_count*4, conv_size, padding='same', activation=activation_function)(x)

    x = Cropping3D()(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(cae_filter_count*2, conv_size, padding='same', activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(cae_filter_count, conv_size, padding='same', activation=activation_function)(x)

    decoder = Conv3DTranspose(1, conv_size, padding='same', activation='sigmoid', name='decoded')(x)

    model = Model(inputs=inputs, outputs=decoder)

    model.summary()

    return model


def cae_model():
    encoder = cae_encoder()
    decoder = cae_decoder()

    autoencoder = Model(encoder.input, decoder(encoder.output))

    autoencoder.compile(loss=cae_loss_function, optimizer=cae_optimizer)

    return autoencoder


def gmd_classifier():
    inputs = Input(shape=input_size)

    padding = 'same'
    strides = 2
    filters = 8
    cs = (4, 4, 4)
    dropout = 0.3

  #  x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(inputs)
   # x = MaxPooling3D(pool_size=pool_size)(x)

   # x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(inputs)
   # x = MaxPooling3D(pool_size=pool_size)(x)

  #  x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(inputs)
   # x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function,
               activity_regularizer=activity_reg, bias_regularizer=bias_reg, kernel_regularizer=kernel_reg)(inputs)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function,
               activity_regularizer=activity_reg, bias_regularizer=bias_reg, kernel_regularizer=kernel_reg)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function,
               activity_regularizer=activity_reg, bias_regularizer=bias_reg, kernel_regularizer=kernel_reg)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function,
               activity_regularizer=activity_reg, bias_regularizer=bias_reg, kernel_regularizer=kernel_reg)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function,
               activity_regularizer=activity_reg, bias_regularizer=bias_reg, kernel_regularizer=kernel_reg)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function,
               activity_regularizer=activity_reg, bias_regularizer=bias_reg, kernel_regularizer=kernel_reg)(x)

    x = Dropout(dropout)(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function,
               activity_regularizer=activity_reg, bias_regularizer=bias_reg, kernel_regularizer=kernel_reg)(x)

    encoder = Flatten(name='encoded')(x)

    x = Dense(50, activation=activation_function)(encoder)

    x = Dropout(0.5)(x)

    x = Dense(10, activation=activation_function)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def no_gmd_classifier():
    inputs_age = Input(shape=single_input_size)
    inputs_site = Input(shape=single_input_size)
    inputs_gender = Input(shape=single_input_size)
    inputs_tiv = Input(shape=single_input_size)
    inputs_mean = Input(shape=mean_input_size)
    inputs_var = Input(shape=mean_input_size)

    x_mean = Dense(50, activation=activation_function)(inputs_mean)
    x_var = Dense(50, activation=activation_function)(inputs_var)

    single_concat = concatenate([inputs_age, inputs_site, inputs_gender, inputs_tiv])
    dense_single = Dense(4, activation=activation_function)(single_concat)

    x = concatenate([x_mean, x_var, dense_single])

    x = Dense(50, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(10, activation=activation_function)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs_age, inputs_site, inputs_gender, inputs_tiv, inputs_mean, inputs_var],
                  outputs=x)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def merged_gmd_classifier():
    inputs_gmd = Input(shape=input_size)
    inputs_age = Input(shape=single_input_size)
    inputs_site = Input(shape=single_input_size)
    inputs_gender = Input(shape=single_input_size)
    inputs_tiv = Input(shape=single_input_size)
    inputs_mean = Input(shape=mean_input_size)
    inputs_var = Input(shape=mean_input_size)

    padding = 'valid'
    strides = 2
    filters = 8
    cs = (4, 4, 4)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(inputs_gmd)
    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(x)
    x = Conv3D(filters, cs, padding=padding, strides=strides, activation=activation_function)(x)

    x_gmd = Flatten(name='encoded')(x)

    x_mean = Dense(50, activation=activation_function)(inputs_mean)
    x_var = Dense(50, activation=activation_function)(inputs_var)

    single_concat = concatenate([inputs_age, inputs_site, inputs_gender, inputs_tiv])
    dense_single = Dense(4, activation=activation_function)(single_concat)

    x = concatenate([x_gmd, x_mean, x_var, dense_single])

    x = Dense(50, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(10, activation=activation_function)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs_gmd, inputs_age, inputs_site, inputs_gender, inputs_tiv, inputs_mean, inputs_var],
                  outputs=x)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def strided_merged_classifier():
    inputs_gmd = Input(shape=input_size)
    inputs_age = Input(shape=single_input_size)
    inputs_site = Input(shape=single_input_size)
    inputs_gender = Input(shape=single_input_size)
    inputs_tiv = Input(shape=single_input_size)
    inputs_mean = Input(shape=mean_input_size)
    inputs_var = Input(shape=mean_input_size)

    saved_gmd_classifier = load_model(strided_gmd_classifier)
    cutoff_layer_name = 'dense_1'
    gmd_model = Model(inputs=saved_gmd_classifier.input,
                      outputs=saved_gmd_classifier.get_layer(cutoff_layer_name).output)

    for layer in gmd_model.layers:
        layer.trainable = False

    x_mean = Dense(50, activation=activation_function)(inputs_mean)
    x_var = Dense(50, activation=activation_function)(inputs_var)

    single_concat = concatenate([inputs_age, inputs_site, inputs_gender, inputs_tiv])
    dense_single = Dense(4, activation=activation_function)(single_concat)

    x = concatenate([gmd_model(inputs_gmd), x_mean, x_var, dense_single])

    x = Dense(50, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(10, activation=activation_function)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs_gmd, inputs_age, inputs_site, inputs_gender, inputs_tiv, inputs_mean, inputs_var],
                  outputs=x)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def top_level_classifier():
    inputs = Input(shape=(cae_output_count, 1))
    x = Flatten(name='classifier_input')(inputs)

    x = Dense(400, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(100, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(50, activation=activation_function)(x)
    x = Dropout(0.5)(x)

    x = Dense(2, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=x)

    model.summary()

    return model


def load_cae(path, trainable=False):
    # Load Encoder with model weights
    encoder = cae_encoder(trainable)
    encoder.load_weights(path, by_name=True)
    encoder.compile(loss=cae_loss_function, optimizer=cae_optimizer)
    return encoder


def cae_classifier_one_hot_model():
    encoder = load_cae(cae_model_file)
    top_level = top_level_classifier()

    model = Model(encoder.input, top_level(encoder.output))

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def cae_classifier_model():
    encoder = load_cae(cae_model_file)
    top_level = top_level_classifier()

    model = Model(encoder.input, top_level(encoder.output))

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def merged_classifier():
    inputs_gmd = Input(shape=input_size)
    inputs_age = Input(shape=single_input_size)
    inputs_site = Input(shape=single_input_size)
    inputs_gender = Input(shape=single_input_size)
    inputs_tiv = Input(shape=single_input_size)
    inputs_mean = Input(shape=mean_input_size)
    inputs_var = Input(shape=mean_input_size)

    encoder = load_cae(cae_model_file, trainable=True)

    # x = Conv3D(cae_filter_count, conv_size, activation=activation_function, padding='same')(inputs_gmd)
    # x = MaxPooling3D(pool_size=pool_size)(x)
    #
    # x = Conv3D(cae_filter_count*2, conv_size, activation=activation_function, padding='same')(x)
    # x = MaxPooling3D(pool_size=pool_size)(x)
    #
    # x = Conv3D(cae_filter_count*4, conv_size, activation=activation_function, padding='same')(x)
    # x = ZeroPadding3D(padding=(1, 1, 1))(x)
    # x = MaxPooling3D(pool_size=pool_size)(x)
    #
    # x = Conv3D(cae_filter_count*8, conv_size, activation=activation_function, padding='same')(x)
    # x = MaxPooling3D(pool_size=pool_size)(x)
    #
    # x = Conv3D(cae_filter_count*16, conv_size, activation=activation_function, padding='same')(x)
    # x = MaxPooling3D(pool_size=pool_size)(x)
    #
    # x = Conv3D(cae_filter_count*2, conv_size, activation=activation_function, padding='same')(x)
    #
    # x = Flatten()(x)

    x1 = Dense(100, activation=activation_function)(inputs_mean)
    x2 = Dense(100, activation=activation_function)(inputs_var)

    single_concat = concatenate([inputs_age, inputs_site, inputs_gender, inputs_tiv])
    dense_single = Dense(4, activation=activation_function)(single_concat)

    x = concatenate([x1, x2, dense_single, encoder(inputs_gmd)])

    x = Dropout(0.5)(x)

    x = Dense(500, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(200, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(100, activation=activation_function)(x)

    x = Dropout(0.5)(x)

    x = Dense(50, activation=activation_function)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[inputs_gmd, inputs_age, inputs_site, inputs_gender, inputs_tiv, inputs_mean, inputs_var],
                  outputs=x)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def batch_mean_var(indices, f):
    mean = f['regional_GMD_mean']
    var = f['regional_GMD_var']
    labels = f['one_hot_label']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            label = labels[index]

            yield ([mean[index, ...][np.newaxis, ...],
                    var[index, ...][np.newaxis, ...]],
                   label[np.newaxis, ...])


def batch_all(indices, f):
    mean = f['regional_GMD_mean']
    var = f['regional_GMD_var']
    images = f['GMD']
    labels = f['one_hot_label']
    age = f['age']
    site = f['site']
    gender = f['gender']
    tiv = f['tiv']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            label = labels[index]

            yield ([np.reshape(images[index, ...], input_size)[np.newaxis, ...],
                    age[index, ...][np.newaxis, ...],
                    site[index, ...][np.newaxis, ...],
                    gender[index, ...][np.newaxis, ...],
                    tiv[index, ...][np.newaxis, ...],
                    mean[index, ...][np.newaxis, ...],
                    var[index, ...][np.newaxis, ...]],
                   label[np.newaxis, ...])


def batch_no_gmd(indices, f):
    mean = f['regional_GMD_mean']
    var = f['regional_GMD_var']
    labels = f['one_hot_label']
    age = f['age']
    site = f['site']
    gender = f['gender']
    tiv = f['tiv']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            label = labels[index]

            yield ([age[index, ...][np.newaxis, ...],
                    site[index, ...][np.newaxis, ...],
                    gender[index, ...][np.newaxis, ...],
                    tiv[index, ...][np.newaxis, ...],
                    mean[index, ...][np.newaxis, ...],
                    var[index, ...][np.newaxis, ...]],
                   label[np.newaxis, ...])


def batch(indices, f):
    images = f['GMD']
    labels = f['one_hot_label']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            label = labels[index]

            yield (np.reshape(images[index, ...], input_size)[np.newaxis, ...], label[np.newaxis, ...])


def balanced_batch(indices, f):
    images = f['GMD']
    labels = f['one_hot_label'].value

    zero_indices = np.where((labels == (1, 0)).all(axis=1))[0]
    one_indices = np.where((labels == (0, 1)).all(axis=1))[0]

    while True:
        np.random.shuffle(indices)

        zi = -1
        oi = -1

        for index in zero_indices:
            label = labels[index]
            other_index = 0
            if np.array_equal(label, [1, 0]):
                oi += 1
                other_index = one_indices[oi]
            else:
                zi += 1
                other_index = zero_indices[zi]

            if zi >= len(zero_indices) - 1:
                zi = -1
            if oi >= len(one_indices) - 1:
                oi = -1

            yield (np.array([np.reshape(images[index, ...], input_size),
                            np.reshape(images[other_index, ...], input_size)]),
                   np.array([label,
                            labels[other_index]]))


def batch_cae(indices, f):
    images = f['GMD']
    labels = f['GMD']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            yield (np.reshape(images[index, ...], input_size)[np.newaxis, ...],
                   np.reshape(labels[index, ...], input_size)[np.newaxis, ...])


def get_nifti_data():
    gmd_filepath = workdir + 'PAC2018_0001.nii'
    gmd_img = nib.load(gmd_filepath)

    return gmd_img.affine, gmd_img.header, gmd_img.get_data()


def visualize_cae(results_dir, model, indices, f):
    from make_dataset import resize_image_with_crop_or_pad
    images = f['GMD']

    min_loss = 1.01
    min_id = -1
    max_loss = 0.00
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

        loss = eval

        if loss > max_loss:
            max_loss = loss
            max_id = index
            max_img = np.reshape(pred, image_size)
            max_orig = np.reshape(img, image_size)
        if loss < min_loss:
            min_loss = loss
            min_id = index
            min_img = np.reshape(pred, image_size)
            min_orig = np.reshape(img, image_size)

    aff = np.eye(4)

    img_data = get_nifti_data()
    test_img = nib.Nifti1Image(resize_image_with_crop_or_pad(img_data[2], image_size, mode='constant'), affine=aff)
    nib.save(test_img, results_dir + 'test_img.nii')

    avg_img = nib.load(average_image_file)
    avg_img = avg_img.get_data()

    worst_img = nib.Nifti1Image(max_img, affine=aff)
    print("Sum of worst image is %f" % np.sum(max_img))
    nib.nifti1.save(worst_img, results_dir + 'worst_img.nii')
    worst_img = nib.Nifti1Image(max_orig, affine=aff)
    print("Sum of worst image original is %f" % np.sum(max_orig))

    nib.nifti1.save(worst_img, results_dir + 'worst_img_orig.nii')
    best_img = nib.Nifti1Image(min_img, affine=aff)
    print("Sum of best image is %f" % np.sum(min_img))
    nib.nifti1.save(best_img, results_dir + 'best_img.nii')
    best_img = nib.Nifti1Image(min_orig, affine=aff)
    nib.nifti1.save(best_img, results_dir + 'best_img_orig.nii')
    print("Sum of best image original is %f" % np.sum(min_orig))
    print("Sum of average image is %f" % np.sum(avg_img))
    print("Original size is ", min_orig.shape)
    print("NIFTI size is ", test_img.shape)


def test_gmd_classifier(model, test_indices, f):
    images = f['GMD']
    labels = f['one_hot_label']

    for i in test_indices:
        img = np.reshape(images[i, ...], input_size)[np.newaxis, ...]
        label = labels[i]

        output = model.predict(img)[0]
        #print(output)
        pred = np.argmax(output, axis=-1)  # axis=-1, last axis

        prediction = [1, 0]
        if pred == 1:
            prediction = [0, 1]

        if np.array_equal(prediction, label):
            print("%i C " % i, label, prediction)
        else:
            print("%i I " % i, label, prediction)


def test_mean_classifer(model, test_indices, f):
    means = f['regional_GMD_mean']
    vars = f['regional_GMD_var']
    labels = f['one_hot_label']

    for i in test_indices:
        label = labels[i]
        mean = means[i]
        var = vars[i]

        output = model.predict([mean, var])[0]
        #print(output)
        pred = np.argmax(output, axis=-1)  # axis=-1, last axis

        prediction = [1, 0]
        if pred == 1:
            prediction = [0, 1]

        if np.array_equal(prediction, label):
            print("%i C " % i, label, prediction)
        else:
            print("%i I " % i, label, prediction)


def test_no_gmd_classifier(model, test_indices, f):
    means = f['regional_GMD_mean']
    vars = f['regional_GMD_var']
    labels = f['one_hot_label']
    ages = f['age']
    sites = f['site']
    genders = f['gender']
    tivs = f['tiv']

    for i in test_indices:
        label = labels[i]
        mean = means[i]
        var = vars[i]
        age = ages[i]
        site = sites[i]
        gender = genders[i]
        tiv = tivs[i]

        output = model.predict([age, site, gender, tiv, mean, var])[0]
        #print(output)
        pred = np.argmax(output, axis=-1)  # axis=-1, last axis

        prediction = [1, 0]
        if pred == 1:
            prediction = [0, 1]

        if np.array_equal(prediction, label):
            print("%i C " % i, label, prediction)
        else:
            print("%i I " % i, label, prediction)


def test_merged_classifier(model, test_indices, f):
    means = f['regional_GMD_mean']
    vars = f['regional_GMD_var']
    images = f['GMD']
    labels = f['one_hot_label']
    ages = f['age']
    sites = f['site']
    genders = f['gender']
    tivs = f['tiv']

    for i in test_indices:
        img = np.reshape(images[i, ...], input_size)[np.newaxis, ...]
        label = labels[i]
        mean = means[i]
        var = vars[i]
        age = ages[i]
        site = sites[i]
        gender = genders[i]
        tiv = tivs[i]

        output = model.predict([img, age, site, gender, tiv, mean, var])[0]
        #print(output)
        pred = np.argmax(output, axis=-1)  # axis=-1, last axis

        prediction = [1, 0]
        if pred == 1:
            prediction = [0, 1]

        if np.array_equal(prediction, label):
            print("%i C " % i, label, prediction)
        else:
            print("%i I " % i, label, prediction)


def summarize_saved_model(filename):
    model = load_model(filename)

    model.summary()


def save_average_img():
    f = h5py.File(workdir + data_file, 'r')
    images = f['GMD']

    avg_img = np.zeros(image_size)
    for img in images:
        avg_img += img

    avg_img /= len(images)

    aff = np.eye(4)
    img = nib.Nifti1Image(avg_img, aff)
    nib.nifti1.save(img, average_image_file)


def plot_metrics(hist, results_dir, fn, metric1, metric2):
    print(hist.history)

    epoch_num = range(len(hist.history[metric1]))
    # train_error = np.subtract(1, np.array(hist.history['acc']))
    # test_error = np.subtract(1, np.array(hist.history['val_acc']))

    plt.clf()
    plt.plot(epoch_num, np.array(hist.history[metric1]), label='Training Accuracy')
    plt.plot(epoch_num, np.array(hist.history[metric2]), label="Validation Error")
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

    # summarize_saved_model(workdir + 'experiment-350/stacked_model.hdf5')

    f = h5py.File(workdir + data_file, 'r')
    images = f['GMD']
    labels = f['label'].value
    one_hot_labels = f['one_hot_label'].value

    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    pac2018_indices = list(range(len(images)))

    # 3D CAE
    print('number of samples in dataset:', images.shape[0])

    train_indices = pac2018_indices
    train_labels = labels

    sss_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    sss_test = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)

    train_indices, validation_indices, test_indices = None, None, None

    for train_index, validation_index in sss_validation.split(np.zeros(len(labels)), labels):
        train_indices = train_index
        validation_indices = validation_index

    labs = labels[validation_indices, ...]

    for validation_index, test_index in sss_test.split(np.zeros(len(labs)), labs):
        validation_indices = validation_index
        test_indices = test_index

    print('train:', train_indices)
    print('val:', validation_indices)
    print('test:', test_indices)

    if train_stacked_model:
        print("Training stacked classifier model")
        model = gmd_classifier()  # no_gmd_classifier() # merged_gmd_classifier() # mean_classifier()
        #  strided_merged_classifier()
        best_model_filename = results_dir + 'best_stacked_model.hdf5'
        model_filename = results_dir + 'stacked_model.hdf5'
        metrics_filename = results_dir + 'test_metrics_stacked'
        batch_func = batch  # batch_all  # batch_no_gmd # batch_mean_var  # batch # balanced_batch
        monitor = 'val_categorical_accuracy'
        test_function = test_gmd_classifier  # test_no_gmd_classifier  # test_mean_classifer  #  test_merged_classifier
        zero_indices = np.where((one_hot_labels == (1, 0)).all(axis=1))[0]
        steps_per_epoch = len(train_indices)  # len(train_indices) # len(zero_indices)
    else:
        print("Training 3D convolutional autoencoder")
        model = cae_model()
        best_model_filename = results_dir + 'best_3d_cae_model.hdf5'
        model_filename = results_dir + '3d_cae_model.hdf5'
        metrics_filename = results_dir + 'test_metrics_3d_cae'
        batch_func = batch_cae
        monitor = 'val_loss'
        steps_per_epoch = len(train_indices)

    # print summary of model
    model.summary()

    num_epochs = 100

    early_stopping = EarlyStopping(monitor=monitor, patience=2)
    model_checkpoint = ModelCheckpoint(best_model_filename,
                                       monitor=monitor,
                                       save_best_only=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.2,
                                  patience=4, min_lr=0.00001)
    tensorboard = TensorBoard(log_dir='./logs/' + str(experiment_number), histogram_freq=0, write_graph=True,
                              write_grads=True,
                              write_images=True)

    hist = model.fit_generator(
        batch_func(train_indices, f),
        steps_per_epoch,
        epochs=num_epochs,
        callbacks=[model_checkpoint, tensorboard],  # , early_stopping
        validation_data=batch_func(validation_indices, f),
        validation_steps=len(validation_indices), class_weight=class_weights
    )

    model.load_weights(best_model_filename)
    model.save(model_filename)

    metrics = model.evaluate_generator(batch_func(test_indices, f), len(test_indices))

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(metrics_filename, 'wb'))

    if train_stacked_model:
        test_function(model, test_indices, f)
    else:
        visualize_cae(results_dir, model, test_indices, f)

    print('This experiment brought to you by the number:', experiment_number)
