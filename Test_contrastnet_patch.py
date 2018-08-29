
"""
    2017-2018 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi

    This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
    License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
    version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
    implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
    more details. You should have received a copy of the GNU General Public License along with this program.
    If not, see <http://www.gnu.org/licenses/>.

    If you are using this software, please cite:
    M. Barni, A. Costanzo, E. Nowroozi, B. Tondi., “CNN-based detection of generic contrast adjustment with
    JPEG post-processing", ICIP 2018 (http://clem.dii.unisi.it/~vipp/files/publications/CM-icip18.pdf)

"""

import os
import configuration as config
import keras
import tensorflow as tf
from keras.preprocessing import image
from keras.models import load_model
from utils import random_jpeg_augmentation
from glob import glob


def generator():

    color_space = 'rgb' if config.PATCH_CHANNELS == 3 else 'grayscale'

    if config.JPEG_AUGMENTATION:
        gen_test = keras.preprocessing.image.ImageDataGenerator(preprocessing_function=random_jpeg_augmentation)

    # No augmentation, simply scale data
    else:
        gen_test = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

    test_g = gen_test.flow_from_directory(directory=config.TEST_FOLDER,
                                          target_size=(config.PATCH_SIZE, config.PATCH_SIZE),
                                          batch_size=config.TEST_BATCH, color_mode=color_space,
                                          classes=[config.CLASS_0_TAG, config.CLASS_1_TAG],
                                          class_mode='categorical')

    print(test_g.class_indices)

    return test_g


def count_patches():
    return 2*len(glob(os.path.join(config.TEST_FOLDER, config.CLASS_0_TAG, '*.*')))


if __name__ == '__main__':

    model_file = 'model_contrastnet.h5' if config.JPEG_AUGMENTATION else 'model_contrastnet.h5'
    model = load_model(os.path.join(config.MODEL_FOLDER, model_file))

    n_test_imgs = count_patches()

    test_generator = generator()
    score = model.evaluate_generator(test_generator, steps=int(n_test_imgs / config.TEST_BATCH))
    print('Test accuracy {:g}. Loss {:g}'.format(score[1], score[0]))
