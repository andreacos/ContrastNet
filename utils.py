
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

import numpy as np
import io
from scipy.misc import imread
import cv2
import bisect


def random_jpeg_augmentation(x_in, quality_factors=None, prob=None):

    """ In-memory JPEG compression with random quality factor

    Keyword arguments:
    x_in: input image matrix (Numpy)
    quality_factors: array of quality factors
    prob: array of probabilities for each quality factor in quality_factors
    Returns:
      enhanced image
    """

    # If no array of quality factors is provided, use default array
    if quality_factors is None:
        from configuration import AUG_JPEG_QFS
        quality_factors = np.array(AUG_JPEG_QFS)

    # Probability to choose a certain quality factor. Default: same probability for all values
    if prob is None:
        prob = np.ones(quality_factors.shape, dtype=np.float32) / len(quality_factors)
    else:
        # If both input arrays have been provided, make sure they have the same size
        assert quality_factors.shape == prob.shape

    def jpeg_compression_to_buffer(im, fmt='jpeg', jpg_quality=100):
        buf = io.BytesIO()
        im.save(buf, format=fmt, quality=jpg_quality)
        buf.seek(0)
        return buf

    # Convert input Numpy image to PIL Image
    from PIL import Image
    x = Image.fromarray(np.uint8(x_in))

    # Pick up a random quality factor
    qf = int(np.random.choice(quality_factors, p=prob))

    if qf != -1:
        # Compress the image in memory
        x_jpg_buff = jpeg_compression_to_buffer(x, 'jpeg', qf).getvalue()

        # Read image from memory as a Numpy matrix
        with io.BytesIO(x_jpg_buff) as stream:
            x_jpg = imread(stream, flatten=False)
    else:
        x_jpg = x

    return np.float32(x_jpg) / 255.


def gamma_correction(img, gamma=1., channels=3):

    """ Performs per-channel gamma correction

    Args:
      img_file: input image
      gamma: gamma correction value
      channels: image channel, determines grayscale or color processing
    Returns:
      enhanced image
    """

    def adjust_gamma(image, g=1.0):
        # build a lookup table mapping the pixel values [0, 255] to their adjusted gamma values
        table = np.array([((i / 255.0) ** (1.0 / g)) * 255
                          for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        return cv2.LUT(image, table)

    if channels == 1:
        img_gamma = adjust_gamma(img, gamma)

    else:
        bimage, gimage, rimage = cv2.split(img)
        b_adj = adjust_gamma(bimage, gamma)
        g_adj = adjust_gamma(gimage, gamma)
        r_adj = adjust_gamma(rimage, gamma)
        img_gamma = cv2.merge((b_adj, g_adj, r_adj))

    return img_gamma


def imadjust(img, channels=3, tol=5, vin=[0, 255], vout=(0, 255)):

    """ Performs Matlab-like histogram stretch

    Args:
      img: file path to the input image
      channels: image channel, determines grayscale or color processing
      tol:
      vin: input pixelrange
      vout: output pixel range
    Returns:
      enhanced image
    """

#    if channels == 1:
#        img = cv2.imread(img_file, cv2.IMREAD_GRAYSCALE)
#    else:
#        img = cv2.imread(img_file)

    h_hsv, s_hsv, v_hsv = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))

    # Stretching on V-channel
    tol = max(0, min(100, tol))

    if tol > 0:
        # Compute in and out limits
        # Histogram
        hist = np.histogram(v_hsv, bins=list(range(256)), range=(0, 255))[0]

        # Cumulative histogram
        cum = hist
        for i in range(1, 255): cum[i] = cum[i - 1] + hist[i]

        # Compute bounds
        total = v_hsv.shape[0] * v_hsv.shape[1]
        low_bound = total * tol / 100
        upp_bound = total * (100 - tol) / 100
        vin[0] = bisect.bisect_left(cum, low_bound)
        vin[1] = bisect.bisect_left(cum, upp_bound)

    # Stretching
    scale = (vout[1] - vout[0]) / (vin[1] - vin[0])
    vs = v_hsv - vin[0]
    vs[v_hsv < vin[0]] = 0
    vd = vs * scale + vout[0]
    vd[vd > vout[1]] = vout[1]

    dst = np.uint8(vd)
    hsv_img_stretch = cv2.merge((h_hsv, s_hsv, dst))
    img_stretch = cv2.cvtColor(hsv_img_stretch, cv2.COLOR_HSV2BGR)

    return img_stretch


def clahe_enhancement(img, channels=3, cliplim=5):

    """ Performs CLAHE enhancement

    Args:
      img_file: file path to the input image
      channels: image channel, determines grayscale or color processing
      cliplim: CLAHE clip limit
    Returns:
      enhanced image
    """

    # Initialise CLAHE operator
    clahe = cv2.createCLAHE(clipLimit=cliplim, tileGridSize=(8, 8))

    # Work on grayscale or on colors. CLAHE is performed on the V channel in HSV color space
    if channels == 1:
        img_clahe = clahe.apply(img)

    else:

        # Convert image to the HSV color space and perform CLAHE on V channel
        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv_img)
        cl = clahe.apply(v)

        # Back to BGR
        v_img = cv2.merge((h, s, cl))
        img_clahe = cv2.cvtColor(v_img, cv2.COLOR_HSV2BGR)

    return img_clahe
