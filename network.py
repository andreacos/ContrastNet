
"""
    2017-2018 Department of Information Engineering and Mathematics, University of Siena, Italy.

    Authors:  Andrea Costanzo (andreacos82@gmail.com) and Benedetta Tondi (....)

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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D


def contrast_net(in_shape=(64, 64, 3), num_classes=2, nf_base=64, layers_depth=(4, 3)):

    """ Builds the graph for a CNN based on Keras (TensorFlow backend)

    Args:
       in_shape: shape of the input image (Height x Width x Depth).
       num_classes: number of output classes
       nf_base: number of filters in the first layer
       layers_depth: number of convolutions at each layer

    Returns:
       Keras sequential model.
    """

    model = Sequential()

    # First batch of convolutions followed by Max Pooling 
    model.add(Conv2D(nf_base, kernel_size=(3, 3), strides=(1, 1), input_shape=in_shape, activation='relu', name='conv1_1'))

    for i in range(0, layers_depth[0]):
        model.add(Conv2D(nf_base+nf_base*(i+1),
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         activation='relu',
                         name='conv1_{}'.format(i+2)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    last_size = nf_base+nf_base*(i+1)

    # Second batch of convolutions followed by Max Pooling
    for i in range(0, layers_depth[1]):
        model.add(Conv2D(last_size+nf_base*(i+1),
                         kernel_size=(3, 3),
                         strides=(1, 1),
                         activation='relu',
                         name='conv2_{}'.format(i+2)))

    model.add(MaxPooling2D(pool_size=(2, 2)))

    # One last convolution with half the number of filters of the previous step
    nf = int(model.layers[-1].output_shape[-1]/2)
    model.add(Conv2D(nf, kernel_size=(1,1), strides=1, name='conv3_1'))

    # Flatten before fully-connected layer
    model.add(Flatten())
    model.add(Dense(250, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax', name='predictions'))

    return model
