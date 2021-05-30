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

def process_image(image):
    _THRESHOLD = 0.5
    
    vgg_model = VGG16(weights='imagenet', include_top=False,
                      input_shape=(64, 64, 3))

    model_aug = Sequential()
    model_aug.add(vgg_model)

    top_model = Sequential()
    top_model.add(Flatten(input_shape=(2, 2, 512)))

    top_model.add(Dense(64, activation='relu'))

    top_model.add(Dense(1, activation='sigmoid'))

    # top_model.load_weights('./model/top_model_full_data_custom_lr_weights.h5')
    
    model_aug.add(top_model)

    for layer in model_aug.layers[0].layers[:17]:
        layer.trainable = False
        
    model_aug.load_weights('./model/fine_tuned_model_adam_weights_new.h5')

    model_aug.compile(loss='binary_crossentropy',
                      optimizer=Adam(lr=1e-6), metrics=['accuracy'])

    result = model_aug.predict_classes(image)

    return np.count_nonzero(result)/len(result) <= _THRESHOLD


def process_image_file(raw_image_fila):
    image = _get_image_file(raw_image_fila)
    return process_image(image)


def process_image_url(url):
    image = _get_image(url)
    return process_image(image)


def _get_image_file(image_file):
    original_image = imread(image_file)
    return sample_test_image(original_image)

def _get_image(url, session=session):
    original_image = imread(session.get(url).content)
    
    # plt.imshow(original_image)
    # plt.show()
    
    return sample_test_image(original_image)


if __name__ == '__main__':
    list_url = [
        'https://i.ibb.co/4j1yxMR/152681a0017a5fded699c43cd6df97d1.png', #fake from dataset
        'https://i.ibb.co/FWMF04K/0ad15fb810f4cf428742462824236158.png', # pristine from dataset
        'https://i.stack.imgur.com/peB3w.png', #7 digit display 
        'https://i.stack.imgur.com/MIe6s.png', #7 digit display
        'https://i.stack.imgur.com/2rbal.png', #7 digit display
        'https://i.stack.imgur.com/KUfwD.png', #7 digit display
        'https://i.stack.imgur.com/oGsK8.png',  #7 digit display
        'https://medialampung.co.id/wp-content/uploads/2020/01/KWH-Pascabayar.jpg', #fake
        'https://i.ibb.co/Hp4FCcZ/1622199455244.jpg', #pristine
        'https://i.ibb.co/6WXN05s/13989611014162.jpg' #pristine
    ]
    for url in list_url:
        result = process_image(url) #7 digit display
        print(f'URL: {url} ==> Fake: {result}')