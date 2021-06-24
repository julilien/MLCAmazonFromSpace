####
#
# Note that this code is based on the tutorial by Jason Brownlee
# (https://machinelearningmastery.com/how-to-develop-a-convolutional-neural-network-to-classify-satellite-photos-of-the-amazon-rainforest/).
# We simply ported this to TF 2, credit belongs to him.
#
####
import sys
import numpy as np
from numpy import load
from matplotlib import pyplot
from sklearn.model_selection import KFold
from tensorflow.keras import backend
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Load Amazon from Space dataset
def load_dataset():
    data = load('planet_data.npz')
    X, y = data['arr_0'], data['arr_1']

    return X, y


# calculate fbeta score for multi-class/label classification
def f_beta(y_true, y_pred, beta=2):
    # clip predictions
    y_pred = backend.clip(y_pred, 0., 1.)
    # calculate elements
    tp = backend.sum(backend.round(backend.clip(y_true * y_pred, 0, 1)), axis=1)
    fp = backend.sum(backend.round(backend.clip(y_pred - y_true, 0, 1)), axis=1)
    fn = backend.sum(backend.round(backend.clip(y_true - y_pred, 0, 1)), axis=1)
    # calculate precision
    p = tp / (tp + fp + backend.epsilon())
    # calculate recall
    r = tp / (tp + fn + backend.epsilon())
    # calculate fbeta, averaged across each class
    bb = beta ** 2
    fbeta_score = backend.mean((1 + bb) * (p * r) / (bb * p + r + backend.epsilon()))
    return fbeta_score


# Define a VGG16-based model with classification head
def get_model(in_shape=(128, 128, 3), out_shape=17):
    # Load VGG16 encoder (pretrained on ImageNet)
    model = VGG16(include_top=False, input_shape=in_shape)

    # Freeze encoder
    for layer in model.layers:
        layer.trainable = False

    # Unfreeze only last conv block
    model.get_layer('block5_conv1').trainable = True
    model.get_layer('block5_conv2').trainable = True
    model.get_layer('block5_conv3').trainable = True
    model.get_layer('block5_pool').trainable = True
    # add new classifier layers
    flat1 = Flatten()(model.layers[-1].output)
    class1 = Dense(128, activation='relu', kernel_initializer='he_uniform')(flat1)
    output = Dense(out_shape, activation='sigmoid')(class1)
    model = Model(inputs=model.inputs, outputs=output)

    # Compile the model using binary cross-entropy
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=[f_beta])

    return model


# plot diagnostic learning curves
def summarize_diagnostics(history, fold_idx):
    # plot loss
    pyplot.subplot(211)
    pyplot.title('Cross Entropy Loss')
    pyplot.plot(history.history['loss'], color='blue', label='train')
    pyplot.plot(history.history['val_loss'], color='orange', label='test')
    # plot accuracy
    pyplot.subplot(212)
    pyplot.title('Fbeta')
    pyplot.plot(history.history['f_beta'], color='blue', label='train')
    pyplot.plot(history.history['val_fbeta'], color='orange', label='test')
    # save plot to file
    filename = sys.argv[0].split('/')[-1]
    pyplot.savefig(filename + '_plot_{}.png'.format(fold_idx))
    pyplot.close()


def run_cv():
    # Load Amazon from Space dataset
    X, y = load_dataset()

    # Load image filenames (such that we can map them to the predictions later on)
    import pickle
    with open('filenames.pkl', 'rb') as f:
        filenames = pickle.load(f)
    filenames = np.asarray(filenames)

    # Perform cross-validation (3-fold)
    skf = KFold(n_splits=3, shuffle=True, random_state=0)
    fold_idx = 0
    for train_index, test_index in skf.split(X, y):
        # Get fold splits
        trainX, testX = X[train_index], X[test_index]
        trainY, testY = y[train_index], y[test_index]

        # Get file names of the test images for better traceability
        test_file_names = filenames[test_index]

        # Create data generators with augmentation
        train_datagen = ImageDataGenerator(featurewise_center=True, horizontal_flip=True, vertical_flip=True,
                                           rotation_range=90)
        test_datagen = ImageDataGenerator(featurewise_center=True)
        # ImageNet feature means for centering
        train_datagen.mean = [123.68, 116.779, 103.939]
        test_datagen.mean = [123.68, 116.779, 103.939]
        train_it = train_datagen.flow(trainX, trainY, batch_size=64, shuffle=True)
        test_it = test_datagen.flow(testX, testY, batch_size=64, shuffle=False)

        model = get_model()
        model.summary()

        history = model.fit_generator(train_it, steps_per_epoch=len(train_it),
                                      validation_data=test_it, validation_steps=len(test_it), epochs=25, verbose=1)
        # evaluate model
        loss, fbeta = model.evaluate_generator(test_it, steps=len(test_it), verbose=1)
        print('> loss=%.3f, fbeta=%.3f' % (loss, fbeta))
        # learning curves
        summarize_diagnostics(history, fold_idx)

        # Store test predictions
        predictions = model.predict(test_it)
        np.save("predictions_{}".format(fold_idx), predictions)
        with open("filenames_{}.pkl".format(fold_idx), 'wb') as f:
            pickle.dump(test_file_names, f)

        fold_idx += 1


# Entry point
run_cv()
