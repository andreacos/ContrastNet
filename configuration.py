
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

# ---------------------------------------------------------------------------------
# Dataset parameters
# ---------------------------------------------------------------------------------

DATASET_FOLDER = 'D:/Datasets/RAISE8K/'         # Dataset directory

PATCH_SIZE = 64                                 # Patch size
PATCH_CHANNELS = 3                              # Color mode

CLASS_0_TAG = 'original'                        # Tag for pristine image class
CLASS_1_TAG = 'enhanced'                        # Tag for enhanced image class

JPEG_Q = None									# JPEG compression of the dataset images
CLAHE_CLIP = 5									# Clahe clip limit
GAMMA = [0.7, 1.5]                              # Gamma correction values
ADJUST_TOL = 5                                  # Percentage of saturated pixel values (top % and bottom %)
ENHANCEMENT_PROBS = [1/3., 1/3., 1/3.]          # Prob. to choose enhancement (order: Clahe, HistStretch, Gamma)
GAMMA_PROBS = [1/2., 1/2.]                      # Prob. to choose a gamma value among those available in GAMMA

MAX_TRAIN_BLOCKS = 2e6                          # Number of training patches
MAX_VAL_BLOCKS = 2e5                            # Number of validation patches
MAX_TEST_BLOCKS = 2e5                           # Number of test patches
MAX_PER_IMAGE = 1500                            # Maximum number of patches from the same image

# ---------------------------------------------------------------------------------
# Directories
# ---------------------------------------------------------------------------------

MODEL_FOLDER = './models/'                      # Models are stored here
TRAIN_FOLDER = './db/train'                     # Training patches are stored here
VALIDATION_FOLDER = './db/validation'           # Validation patches are stored here
TEST_FOLDER = './db/test'                       # Test patches are stored here

# ---------------------------------------------------------------------------------
# Network parameters
# ---------------------------------------------------------------------------------

NUM_EPOCHS = 3                                  # Number of training epochs
TRAIN_BATCH = 32                                # Training batch size
VALIDATION_BATCH = 32                           # Validation batch size
TEST_BATCH = 100                                # Test batch size

# Resume training

RESUME_TRAINING = False
RESUME_MODEL = './models/model_contrastnet.h5'

# Augmentations

JPEG_AUGMENTATION = False
AUG_JPEG_QFS = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, -1]
