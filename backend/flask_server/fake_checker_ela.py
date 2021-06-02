# from imageio import imread
from io import BytesIO
import numpy as np
from PIL import Image
from PIL import Image, ImageChops, ImageEnhance

from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.models import Sequential
from keras.optimizers import Adam

from localfile import session


def convert_to_ela_image(image, quality):
    resaved_filename = 'resaved.jpg'

    im = Image.open(image).convert('RGB')
    im.save(resaved_filename, 'JPEG', quality=quality)
    resaved_im = Image.open(resaved_filename)

    ela_im = ImageChops.difference(im, resaved_im)

    extrema = ela_im.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff

    ela_im = ImageEnhance.Brightness(ela_im).enhance(scale)

    # ela_im.save("ela.jpg", 'JPEG', quality=100)

    return ela_im


def build_model():
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
              activation='relu', input_shape=(128, 128, 3)))
    model.add(Conv2D(filters=32, kernel_size=(5, 5), padding='valid',
              activation='relu', input_shape=(128, 128, 3)))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(2, activation='softmax'))
    return model


def process_image(image):
    model = build_model()

    model.load_weights('./model/ela_model_casia.h5')

    optimizer = Adam(lr=1e-4, decay=1e-4/30)

    model.compile(optimizer=optimizer,
                  loss="binary_crossentropy", metrics=["accuracy"])

    return bool(np.argmax(model.predict(image), axis=1)[0])


def process_image_file_ela(raw_image_fila):
    image = _get_image_file(raw_image_fila)
    return process_image(image)


def process_image_url_ela(url):
    image = _get_image(url)
    return process_image(image)


def _get_image_file(image_file):
    ela_image = np.array(convert_to_ela_image(
        image_file, 90).resize((128, 128))).flatten() / 255.0
    ela_image = ela_image.reshape(-1, 128, 128, 3)
    return ela_image


def _get_image(url, session=session):
    original_image = BytesIO(session.get(url).content)

    # plt.imshow(original_image)
    # plt.show()

    ela_image = np.array(convert_to_ela_image(
        original_image, 90).resize((128, 128))).flatten() / 255.0
    ela_image = ela_image.reshape(-1, 128, 128, 3)

    return ela_image
