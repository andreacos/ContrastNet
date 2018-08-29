
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

from keras.models import load_model
from skimage.util import view_as_windows
import numpy as np
import cv2


def window_decision(img_data, model, win_size=(64, 64)):
    """Divide input image into blocks then classify each block with a trained CNN model. Block scores are aggregated
       by means of average soft decision **on probability of being contrast enhanced**.

    Args:
       img_data: image matrix.
       model: trained CNN model (Keras)
       win_size: (height, width) of the image blocks. Default: (64, 64)

    Returns:
       Score
    """

    # Make sure image has 3-channels
    if len(img_data.shape) == 2:
        img_data = np.tile(np.reshape(img_data, (img_data.shape[0], img_data.shape[1], 1)), (1, 1, 3))

    img_data = img_data[:, :, 0:3]

    # Divide image into non-overlapping blocks
    image_view = view_as_windows(img_data, [win_size[0], win_size[1], 3], win_size[0])

    h, w, c, _, _, _ = image_view.shape

    # Reshape the image view so that it is a stack of size N_blocks. Each element is a color patch BxBx3
    slices = image_view.reshape(h * w * c, win_size[0], win_size[1], 3)

    # Test the stack
    predicted_values = model.predict(slices / 255, verbose=0)

    # Soft decision on each block (label = 1 means manipulated image)
    n_blocks =  predicted_values.shape[0]
    return np.sum(predicted_values[:, 1]) / n_blocks


if __name__ == '__main__':

    img = cv2.imread('./resources/rdcc2b214t_clahe.TIF')
    img = img[:, :, ::-1]    # BGR to RGB

    model = load_model('./models/model_contrastnet.h5')
    score = window_decision(img, model, (64, 64, 3))
    print('Score: {} (values near 1 (0) indicate presence (absence) of manipulation)'.format(score))
