from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Conv3D, MaxPooling3D, Flatten, BatchNormalization
from keras.layers import Conv3DTranspose, Reshape, UpSampling3D, Input, Lambda
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.utils import to_categorical
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

mpl.use('Agg')
import matplotlib.pyplot as plt

# Set seed for experiment replicability
#random_seed = 38493
#seed(random_seed)
#set_random_seed(random_seed)

workdir = os.path.expanduser('~/pac2018_root/')
cae_model_file = workdir + 'experiment-62/3d_cae_model.hdf5'  # 'experiment-1/3d_cae_model.hdf5' 62 is deep, 1 is shallow
data_file = 'pac2018.hdf5'
average_image_file = workdir + 'average_nifti.nii'

image_size = (152, 152, 152)     # (121, 145, 121) is original size, resized in make_dataset.py
input_size = (152, 152, 152, 1)  # Input size to 3D CAE

conv_size = (3, 3, 3)
pool_size = (2, 2, 2)

cae_loss_function = 'binary_crossentropy'
cae_optimizer = 'adam'
cae_metrics = ['accuracy']
activation_function = 'relu'

train_stacked_model = True

cae_filter_count = 8
cae_output_shape = (15, 15, 15, cae_filter_count)  # (34, 34, 34, 8)
cae_output_count = cae_output_shape[0]*cae_output_shape[1]*cae_output_shape[2]*cae_output_shape[3]


def cae_encoder(trainable=True):
    # 3D Convolutional Auto-Encoder
    inputs = Input(shape=input_size)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function, trainable=trainable)(inputs)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function, trainable=trainable)(x)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function, trainable=trainable)(x)
    x = MaxPooling3D(pool_size=pool_size, trainable=trainable)(x)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function, trainable=trainable)(x)

    encoder = Flatten(name='encoded', trainable=trainable)(x)

    model = Model(inputs=inputs, outputs=encoder)

    model.summary()

    return model


def cae_decoder():
    # 3D Convolutional Auto-Decoder
    inputs = Input(shape=(cae_output_count, 1))

    x = Reshape(cae_output_shape)(inputs)

    x = Conv3DTranspose(cae_filter_count, conv_size, activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(cae_filter_count, conv_size, activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(cae_filter_count, conv_size, activation=activation_function)(x)

    x = UpSampling3D(pool_size)(x)
    x = Conv3DTranspose(cae_filter_count, conv_size, activation=activation_function)(x)

    decoder = Conv3DTranspose(1, conv_size, activation='sigmoid', name='decoded')(x)

    model = Model(inputs=inputs, outputs=decoder)

    return model


def cae_model():
    encoder = cae_encoder()
    decoder = cae_decoder()

    autoencoder = Model(encoder.input, decoder(encoder.output))

    autoencoder.compile(loss=cae_loss_function, optimizer=cae_optimizer, metrics=cae_metrics)

    return autoencoder


def binary_classifier():
    inputs = Input(shape=input_size)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function)(inputs)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function)(x)
    x = MaxPooling3D(pool_size=pool_size)(x)

    x = Conv3D(cae_filter_count, conv_size, activation=activation_function)(x)

    encoder = Flatten(name='encoded')(x)

    x = Dense(50, activation=activation_function)(encoder)
    x = Dense(10, activation=activation_function)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["categorical_accuracy"])

    return model


def top_level_classifier():
    inputs = Input(shape=(cae_output_count, 1))
    x = Flatten()(inputs)

    x = Dense(50, activation=activation_function)(x)
    x = Dense(10, activation=activation_function)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def top_level_one_hot_classifier():
    inputs = Input(shape=(cae_output_count, 1))
    x = Flatten()(inputs)

    x = Dense(50, activation=activation_function)(x)
    x = Dense(10, activation=activation_function)(x)
    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)

    return model


def load_cae(path):
    # Load Encoder with model weights
    encoder = cae_encoder(False)
    encoder.load_weights(path, by_name=True)
    encoder.compile(loss=cae_loss_function, optimizer=cae_optimizer, metrics=cae_metrics)
    return encoder


def cae_classifier_one_hot_model():
    encoder = load_cae(cae_model_file)
    top_level = top_level_classifier()

    model = Model(encoder.input, top_level(encoder.output))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["categorical_accuracy"])

    return model


def cae_classifier_model():
    encoder = load_cae(cae_model_file)
    top_level = top_level_classifier()

    model = Model(encoder.input, top_level(encoder.output))

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=["categorical_accuracy"])

    return model


def batch(indices, f):
    images = f['GMD']
    labels = f['one_hot_label']

    while True:
        np.random.shuffle(indices)

        for index in indices:
            try:
                label = labels[index]

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
        if mn == 'acc' or mn == 'categorical_accuracy':
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

    avg_img = nib.load(average_image_file)
    avg_img = avg_img.get_data()

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
    print("Sum of average image is %f" % np.sum(avg_img))
    print("Original size is ", max_orig.shape)
    print("NIFTI size is ", test_img.shape)


def test_stacked_classifer(model, test_indices, f):
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

    f = h5py.File(workdir + data_file, 'r')
    images = f['GMD']
    labels = f['label'].value

    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    pac2018_indices = list(range(len(images)))

    # 3D CAE
    print('number of samples in dataset:', images.shape[0])

    train_indices = pac2018_indices
    train_labels = labels

    sss_validation = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
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
        model = cae_classifier_one_hot_model()  # binary_classifier()
        best_model_filename = workdir + 'best_stacked_model.hdf5'
        model_filename = results_dir + 'stacked_model.hdf5'
        metrics_filename = results_dir + 'test_metrics_stacked'
        results_plot_filename = 'results_stacked.png'
        batch_func = batch
        # monitor = 'val_binary_accuracy'
        # metric1 = 'binary_accuracy'
        # metric2 = 'val_binary_accuracy'
        monitor = 'val_categorical_accuracy'
        metric1 = 'categorical_accuracy'
        metric2 = 'val_categorical_accuracy'
    else:
        print("Training 3D convolutional autoencoder")
        model = cae_model()
        best_model_filename = workdir + 'best_3d_cae_model.hdf5'
        model_filename = results_dir + '3d_cae_model.hdf5'
        metrics_filename = results_dir + 'test_metrics_3d_cae'
        results_plot_filename = 'results_3d_cae.png'
        batch_func = batch_cae
        monitor = 'val_acc'
        metric1 = 'acc'
        metric2 = 'val_acc'

    # print summary of model
    model.summary()

    num_epochs = 10

    early_stopping = EarlyStopping(monitor=monitor, patience=2)
    model_checkpoint = ModelCheckpoint(best_model_filename,
                                       monitor=monitor,
                                       save_best_only=True)
    tensorboard = TensorBoard(log_dir=results_dir + '/logs', histogram_freq=0, write_graph=True, write_grads=True, write_images=True)

    hist = model.fit_generator(
        batch_func(train_indices, f),
        len(train_indices),
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

    plot_metrics(hist, results_dir, results_plot_filename, metric1, metric2)

    if train_stacked_model:
        test_stacked_classifer(model, test_indices, f)
    else:
        visualize_cae(results_dir, model, test_indices, f)

    print('This experiment brought to you by the number:', experiment_number)
