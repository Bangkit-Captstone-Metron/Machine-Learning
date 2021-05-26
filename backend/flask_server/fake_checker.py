from imageio import imread

import numpy as np
# import matplotlib.pyplot as plt

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


def process_image(url):
    _THRESHOLD = 0.5
    
    image = _get_image(url)
    vgg_model = VGG16(weights='imagenet', include_top=False,
                      input_shape=(64, 64, 3))

    model_aug = Sequential()
    model_aug.add(vgg_model)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=(2, 2, 512)))

    top_model.add(Dense(64, activation='relu'))

    top_model.add(Dense(1, activation='sigmoid'))

    top_model.load_weights('./model/top_model_full_data_custom_lr_weights.h5')
    
    model_aug.add(top_model)

    for layer in model_aug.layers[0].layers[:17]:
        layer.trainable = False

    model_aug.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=1e-6), metrics=['accuracy'])

    result = model_aug.predict_classes(image)

    return np.count_nonzero(result)/len(result) > _THRESHOLD


def _get_image(url, session=session):
    original_image = imread(session.get(url).content)
    
    # original_image = imread('tests/fake/010543abfbd0db1e9aa1b24604336e0c.png')
    
    # plt.imshow(original_image)
    # plt.show()
    
    return sample_test_image(original_image)


if __name__ == '__main__':
    # result = process_image("https://i.ibb.co/zSNzCt4/244a7433a307b9a2c839cefe14c0ba1d.png") #fake
    result = process_image("https://i.ibb.co/FWMF04K/0ad15fb810f4cf428742462824236158.png") #pristine
    print(result)