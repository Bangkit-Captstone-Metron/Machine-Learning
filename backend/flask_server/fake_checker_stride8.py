from imageio import imread

import numpy as np

from keras.applications import VGG16
from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.optimizers import Adam

from localfile import session


def sample_test_image(img):
    kernel_size = 64
    stride = 32
    i = 0

    samples = np.ndarray(shape=(((img.shape[0] - kernel_size)//stride+1)*(
        (img.shape[1] - kernel_size)//stride+1), 64, 64, 3), dtype=np.dtype('uint8'))

    for y_start in range(0, img.shape[0] - kernel_size + 1, stride):
        for x_start in range(0, img.shape[1] - kernel_size + 1, stride):
            samples[i, :, :, :] = img[y_start:y_start +
                                      kernel_size, x_start:x_start + kernel_size, :3]
            i += 1

    return samples


def process_image(image):
    vgg_model = VGG16(weights='imagenet', include_top=False,
                      input_shape=(64, 64, 3))

    model_aug = Sequential()
    model_aug.add(vgg_model)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=(2, 2, 512)))

    top_model.add(Dense(64, activation='relu'))

    top_model.add(Dense(1, activation='sigmoid'))

    model_aug.add(top_model)

    for layer in model_aug.layers[0].layers[:17]:
        layer.trainable = False

    model_aug.load_weights(
        './model/fine_tuned_model_adam_weights_new.h5')

    model_aug.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=1e-6), metrics=['accuracy'])

    result = model_aug.predict(image)

    return bool(((result > 0.75).sum()/len(result)) > 0.25)


def process_image_file_stride8(raw_image_fila):
    image = _get_image_file(raw_image_fila)
    return process_image(image)


def process_image_url_stride8(url):
    image = _get_image(url)
    return process_image(image)


def _get_image_file(image_file):
    original_image = imread(image_file)
    return sample_test_image(original_image)


def _get_image(url, session=session):
    original_image = imread(session.get(url).content)

    return sample_test_image(original_image)
