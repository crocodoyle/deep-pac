from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input, Conv3D, MaxPooling3D, Flatten, BatchNormalization, concatenate
from keras.constraints import max_norm
from keras.utils import to_categorical

from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras.optimizers import SGD, Adam
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
import numpy as np

import h5py, pickle, os


workdir = os.path.expanduser('~/pac2018_root/')

data_file = 'pac2018.hdf5'
# 500, 502, 503 are first batchnorm/32 batch sized runs with 100 epochs
# 454, 455 are decent batchsize=1 100 epochs
# 458 is best 1200 epoch with batch size 1
batch_size = 32
batch_input_size = (batch_size, 121, 145, 121)
batch_output_size = (batch_size, 121, 145, 121, 1)
image_size = (121, 145, 121)        # (121, 145, 121) is original size, resized in make_dataset.py
input_size = (121, 145, 121, 1)     # Input size to classifier


def gmd_classifier():
    image = Input(shape=input_size)
    meta = Input(shape=(7,))

    padding = 'same'
    strides = 2
    filters = 8
    cs = (4, 4, 4)

    x = Conv3D(filters, cs, padding=padding, strides=strides, activation='relu')(image)

    x = BatchNormalization()(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, kernel_constraint=max_norm(), activation='relu')(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, kernel_constraint=max_norm(), activation='relu')(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, kernel_constraint=max_norm(), activation='relu')(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, kernel_constraint=max_norm(), activation='relu')(x)

    x = Conv3D(filters, cs, padding=padding, strides=strides, kernel_constraint=max_norm(), activation='relu')(x)

    flat = Flatten(name='encoded')(x)

    joined = concatenate([flat, meta])

    x = Dense(64, activation='relu', kernel_constraint=max_norm())(joined)

    x = Dropout(0.5)(x)

    x = Dense(64, activation='relu', kernel_constraint=max_norm())(x)

    x = Dropout(0.5)(x)

    x = Dense(2, activation='softmax')(x)

    model = Model(inputs=[image, meta], outputs=x)

    adam = Adam(lr=0.0002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=adam,
                  metrics=["categorical_accuracy"])

    return model


def batch(indices, f, bs):
    images = f['GMD']
    labels = f['one_hot_label']
    sites = f['site']
    age = f['age']
    gender = f['gender']
    tiv = f['tiv']

    while True:
        np.random.shuffle(indices)

        for j in range(0, len(indices)):
            idx = indices[j:j+bs]

            this_bs = len(idx)

            output = np.empty((this_bs, 121, 145, 121))
            lbls = np.empty((this_bs, 2))
            meta = np.empty((this_bs, 7))

            for i in range(0, this_bs):
                id = idx[i]
                img = images[id, :]
                output[i, :] = np.reshape(img, image_size)
                lbls[i, :] = labels[id]
                meta[i, 0:3] = to_categorical(sites[id] - 1, num_classes=3)
                meta[i, 3] = age[id] / 100
                meta[i, 4:6] = to_categorical(gender[id] - 1, num_classes=2)
                meta[i, 6] = tiv[id] / 2000

            yield ([np.reshape(output, (this_bs, 121, 145, 121, 1)),
                   meta],
                   lbls)

            if j + bs > len(indices):
                break
            else:
                j += bs - 1


def test_gmd_classifier(model, test_indices, f):
    images = f['GMD']
    labels = f['one_hot_label']
    sites = f['site']
    age = f['age']
    gender = f['gender']
    tiv = f['tiv']

    meta = np.empty((1, 7))

    for i in test_indices:
        img = np.reshape(images[i, ...], input_size)[np.newaxis, ...]
        label = labels[i]

        meta[0, 0:3] = to_categorical(sites[i] - 1, num_classes=3)
        meta[0, 3] = age[i] / 100
        meta[0, 4:6] = to_categorical(gender[i] - 1, num_classes=2)
        meta[0, 6] = tiv[i] / 2000

        output = model.predict([img, meta])[0]
        #print(output)
        pred = np.argmax(output, axis=-1)  # axis=-1, last axis

        prediction = [1, 0]
        if pred == 1:
            prediction = [0, 1]

        if np.array_equal(prediction, label):
            print("%i C " % i, label, prediction)
        else:
            print("%i I " % i, label, prediction)


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
    one_hot_labels = f['one_hot_label'].value

    class_weights = class_weight.compute_class_weight('balanced', np.unique(labels), labels)

    pac2018_indices = list(range(len(images)))

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

    model = gmd_classifier()
    best_model_filename = results_dir + 'best_stacked_model.hdf5'
    model_filename = results_dir + 'stacked_model.hdf5'
    metrics_filename = results_dir + 'test_metrics_stacked'
    batch_func = batch
    monitor = 'val_categorical_accuracy'
    test_function = test_gmd_classifier
    zero_indices = np.where((one_hot_labels == (1, 0)).all(axis=1))[0]
    steps_per_epoch = len(train_indices)

    # print summary of model
    model.summary()

    num_epochs = 100

    model_checkpoint = ModelCheckpoint(best_model_filename,
                                       monitor=monitor,
                                       save_best_only=True)
    tensorboard = TensorBoard(log_dir='./logs/' + str(experiment_number), histogram_freq=0, write_graph=True,
                              write_grads=True,
                              write_images=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_categorical_accuracy', factor=0.5, patience=10, min_lr=0.0000001)

    hist = model.fit_generator(
        batch_func(train_indices, f, batch_size),
        steps_per_epoch/batch_size,
        epochs=num_epochs,
        callbacks=[model_checkpoint, tensorboard, reduce_lr],
        validation_data=batch_func(validation_indices, f, batch_size),
        validation_steps=len(validation_indices)/batch_size, class_weight=class_weights
    )

    model.load_weights(best_model_filename)
    model.save(model_filename)

    metrics = model.evaluate_generator(batch_func(test_indices, f, batch_size), len(test_indices)/batch_size)

    print(model.metrics_names)
    print(metrics)

    pickle.dump(metrics, open(metrics_filename, 'wb'))

    test_function(model, test_indices, f)

    print('This experiment brought to you by the number:', experiment_number)
