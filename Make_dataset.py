
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

import math
from glob import glob
import time
import cv2
import os
import configuration as config
import numpy as np
from utils import clahe_enhancement, gamma_correction, imadjust


def prepare_dataset(in_dir, out_dir, ext='*', size_image=config.PATCH_SIZE, image_channels=config.PATCH_CHANNELS,
                    max_blocks=1e6, blocks_per_image=500, jpeg_quality=None):

    """ 

    Args:
      in_dir: input image directory
      out_dir: output patches directory
      ext: input file format filter (e.g. "TIF" or "PNG")
      size_image: size of the patches
      image_channels: input image channels determining color mode (either rgb (3) or grayscale (1))
      max_blocks: the number of patches that must be created
      blocks_per_image: the maximum amount of patches that are created from the same image
      jpeg_quality: if None, patches are stored as uncompressed PNG; if in [0, 100] as JPEG with jpeg_quality factor

    """

    assert jpeg_quality is None or jpeg_quality in range(0, 101)

    # Create folders
    if not os.path.exists(os.path.join(out_dir, config.CLASS_0_TAG)):
        os.makedirs(os.path.join(out_dir, config.CLASS_0_TAG))

    if not os.path.exists(os.path.join(out_dir, config.CLASS_1_TAG)):
        os.makedirs(os.path.join(out_dir, config.CLASS_1_TAG))

    image_files = glob(in_dir + '/*.' + ext)
    n_files = len(image_files)

    # Start creating dataset
    it = 0
    cnt_arr = [0, 0, 0]
    count = 0
    begin_time = time.time()
    print('-'*100)
    print(' Creating data set {}: '.format(in_dir))
    print('-' * 100)
    for f in image_files:

        # Pick up a random processing among Clahe, Histogram Stretching and Gamma Correction
        enh_flip = np.random.choice(np.arange(0, 3), p=config.ENHANCEMENT_PROBS)

        if image_channels == 1:
            pristine = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        else:
            pristine = cv2.imread(f)

        if enh_flip == 0:
            enhanced = clahe_enhancement(pristine, channels=config.PATCH_CHANNELS, cliplim=config.CLAHE_CLIP)
            enh_id = 'cl'

        elif enh_flip == 1:
            enhanced = imadjust(pristine, channels=config.PATCH_CHANNELS, tol=config.ADJUST_TOL)
            enh_id = 'hs'

        elif enh_flip == 2:
            # Pick up a random gamma value in GAMMA
            g_flip = np.random.choice(np.arange(0, len(config.GAMMA_PROBS)), p=config.GAMMA_PROBS)
            enhanced = gamma_correction(pristine, gamma=config.GAMMA[g_flip], channels=3)
            enh_id = 'g{}'.format(g_flip)

        print('{} Processed {:6d} of {:6d} ({:3.1f}%) / {:6d} of {:6d} patches ({:3.1f}%) ({}). Elapsed: {:5.1f} seconds)'
              .format(time.strftime("%H:%M:%S"), it, n_files, 100*it/n_files, count, int(max_blocks), 100*count/max_blocks,
                      enh_id, time.time() - begin_time))

        # Now the enhanced image is sub-divided into blocks. Find largest size multiple of the chosen block size
        multiple_height = int(math.floor(pristine.shape[0]/float(size_image)) * size_image)
        multiple_width = int(math.floor(pristine.shape[1] / float(size_image)) * size_image)

        # Count available blocks in current image, pick up random block indices until max_per_image is reached
        available_blocks = int(np.floor(multiple_height / float(size_image))) * int(np.floor(multiple_width / float(size_image)))
        perm = np.random.permutation(available_blocks)[:blocks_per_image]

        # Track how many images have been used for each manipulation
        cnt_arr[enh_flip] += 1

        if image_channels == 1:
            pristine = pristine[:multiple_height, :multiple_width]
        else:
            enhanced = enhanced[:multiple_height, :multiple_width, :image_channels]

        # Divide the pristine image and its enhanced version into blocks
        n_blocks_img = 0
        for k in range(0, multiple_height, size_image):
            for l in range(0, multiple_width, size_image):

                if n_blocks_img in perm:

                    if image_channels == 1:
                        pristine_patch = pristine[k:k + size_image, l:l + size_image]
                        enhanced_patch = enhanced[k:k + size_image, l:l + size_image]
                    else:
                        pristine_patch = pristine[k:k + size_image, l:l + size_image, :image_channels]
                        enhanced_patch = enhanced[k:k + size_image, l:l + size_image, :image_channels]

                    # Save as uncompressed PNG
                    if jpeg_quality is None:
                        cv2.imwrite(os.path.join(out_dir, config.CLASS_0_TAG,
                                                 '{}_patch_{}.png'.format(os.path.basename(f), count)),
                                    pristine_patch, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

                        cv2.imwrite(os.path.join(out_dir, config.CLASS_1_TAG,
                                                 '{}_patch{}_{}.png'.format(os.path.basename(f), count, enh_id)),
                                    enhanced_patch, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

                    # Or save as JPEG compressed
                    else:
                        cv2.imwrite(os.path.join(out_dir, config.CLASS_0_TAG,
                                                 '{}_patch{}.jpg'.format(os.path.basename(f), count)),
                                    pristine_patch, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

                        cv2.imwrite(os.path.join(out_dir, config.CLASS_1_TAG,
                                                 '{}_patch{}_{}.jpg'.format(os.path.basename(f), count, enh_id)),
                                    enhanced_patch, [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality])

                    count += 1

                n_blocks_img += 1

                # Stop when the desired amount of patches has been created
                if count == max_blocks:
                    print("Reached {} patches".format(max_blocks))
                    print('Clahe images: {}'.format(cnt_arr[0]))
                    print('Histogram stretching images: {}'.format(cnt_arr[1]))
                    print('Gamma correction images: {}'.format(cnt_arr[2]))
                    print(' ... Done!')
                    return

        it += 1

    print('{} Processed {:6d} of {:6d} images ({:3.1f}%) / {:6d} of {:6d} patches ({:3.1f}%). Elapsed: {:5.1f} seconds)'
          .format(time.strftime("%H:%M:%S"), it, n_files, 100 * it / n_files, count, int(max_blocks),
                  100 * count / max_blocks, time.time() - begin_time))
    print('Clahe images: {}'.format(cnt_arr[0]))
    print('Histogram stretching images: {}'.format(cnt_arr[1]))
    print('Gamma correction images: {}'.format(cnt_arr[2]))
    print(' ... Done!')

    return


if __name__ == '__main__':

    # Create TRAIN data set
    prepare_dataset(in_dir=os.path.join(config.DATASET_FOLDER, 'train'), out_dir=config.TRAIN_FOLDER,
                    max_blocks=config.MAX_TRAIN_BLOCKS, blocks_per_image=config.MAX_PER_IMAGE,
                    jpeg_quality=config.JPEG_Q, size_image=config.PATCH_SIZE, image_channels=config.PATCH_CHANNELS)

    # Create VALIDATION data set
    prepare_dataset(in_dir=os.path.join(config.DATASET_FOLDER, 'validation'), out_dir=config.VALIDATION_FOLDER,
                    max_blocks=config.MAX_VAL_BLOCKS, blocks_per_image=config.MAX_PER_IMAGE,
                    jpeg_quality=config.JPEG_Q, size_image=config.PATCH_SIZE, image_channels=config.PATCH_CHANNELS)

    # Create TEST data set
    prepare_dataset(in_dir=os.path.join(config.DATASET_FOLDER, 'test'), out_dir=config.TEST_FOLDER,
                    max_blocks=config.MAX_TEST_BLOCKS, blocks_per_image=config.MAX_PER_IMAGE,
                    jpeg_quality=config.JPEG_Q, size_image=config.PATCH_SIZE, image_channels=config.PATCH_CHANNELS)
