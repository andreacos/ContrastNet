![Image](./resources/vippdiism.png)

# ContrastNet Implementation (Keras)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public
License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later
version. This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the
implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details. You should have received a copy of the GNU General Public License along with this program.
If not, see <http://www.gnu.org/licenses/>.

If you are using this software, please cite:

[M. Barni, A. Costanzo, E. Nowroozi, B. Tondi., “CNN-based detection of generic contrast adjustment with
JPEG post-processing", ICIP 2018](http://clem.dii.unisi.it/~vipp/files/publications/CM-icip18.pdf)


<br>

## Installing dependencies

To install the packages required by our software, you may use the provided *requirements.txt*;
feel free to edit or ignore the file but keep in mind that we tested our codes only on the package 
versions that you find in it:
```
cd ContrastNet
pip install -r resources/requirements.txt
```
We tested our codes on Python 3.5 and Python 3.6 under Windows 7/10 and Ubuntu 17.x (64 bit).

<br>

## Preparing datasets

We used a copy of the [RAISE8K dataset](http://mmlab.science.unitn.it/RAISE/), which 
was preliminarily (and randomly) split into three sub-folders:
* **Train**: 6500 TIF images
* **Validation**: 500 TIF images
* **Test**: 1196 TIF images

The lists of images composing each sub-set can be found in the *./resources* folder: files are named respectively *r8k_train.txt*, *r8k_validation.txt* and *r8k_test.txt*. To split your 
own copy of the RAISE8K dataset, you can use the following script:

```
python Split_RAISE8K.py -rdir "/path/to/full/raise8k" -o "output/path/to/split/raise8k" -res "./resources" -copy 1
```

where: *-rdir* is the path to the RAISE8K directory; *-o* is the path to the directory where RAISE8K is split
into three subfolders (Train, Validation, Test); *-res* is the directory where .txt lists are stored; *-copy* either
duplicates (1) or moves (0) files

> py -3.5 Split_RAISE8K.py -rdir "D:\Datasets\RAISE8Kfull" -o "E:\SplitRaise8K" -res ./resources -copy 1 <br>
>  <br>
> Creating folders ...   > Done <br>
> Reading lists ...   > Done <br>
> Copying files ... <br>
> Retrieving test images from ./resources\r8k_test.txt: 100%|##################| 1157/1157 05:05<00:00,  3.79it/s <br>
>       Test set done <br>
> Retrieving validation images from ./resources\r8k_validation.txt: 100%|##############| 500/500 02:09<00:00,  3.86it/s <br>
>       Validation set done <br>

#### Dataset configuration

The creation of the datasets of image patches is controlled by the parameters listed in *configuration.py* (also reported below in this document);
default values in the file coincide with those we used in our paper's implementation.
The script *Make_dataset.py* creates a dataset of pristine and enhanced image patches, where 
enhancement is chosen among [CLAHE](https://en.wikipedia.org/wiki/Adaptive_histogram_equalization), 
Histogram Stretching and [Gamma Correction](https://en.wikipedia.org/wiki/Gamma_correction) according to input probabilities in <font face="Courier New">ENHANCEMENT_PROBS</font>. 
Concerning Gamma Correction, gamma value is randomly chosen among values in <font face="Courier New">GAMMA</font> with probability <font face="Courier New">GAMMA_PROBS</font>.

```
# Dataset parameters

DATASET_FOLDER = 'path/to/split/raise8k'    # Dataset directory

PATCH_SIZE = 64                             # Patch size (square)
PATCH_CHANNELS = 3                          # Color mode (RGB = 3, grayscale = 1)

CLASS_0_TAG = 'original'                    # Tag for pristine image class
CLASS_1_TAG = 'enhanced'                    # Tag for enhanced image class

JPEG_Q = None                               # JPEG compression quality factor
CLAHE_CLIP = 5                              # Clahe clip limit (OpenCV2)
GAMMA = [0.7, 1.5]                          # Gamma correction values
ADJUST_TOL = 5                              # Percentage of saturated pixel values (top % and bottom %)
ENHANCEMENT_PROBS = [1/3., 1/3., 1/3.]      # Prob. to choose each enhancement (order: Clahe, HistStretch, Gamma)
GAMMA_PROBS = [1/2, 1/2]                    # Prob. to choose a gamma value among those available in GAMMA

MAX_TRAIN_BLOCKS = 2e6                      # Number of training patches
MAX_VAL_BLOCKS = 2e5                        # Number of validation patches
MAX_TEST_BLOCKS = 2e5                       # Number of test patches
MAX_PER_IMAGE = 1500                        # Maximum number of patches from the same image

...

TRAIN_FOLDER = './db/train'                   # Training patches are stored here
VALIDATION_FOLDER = './db/validation'         # Validation patches are stored here
TEST_FOLDER = './db/test'                     # Test patches are stored here

```

Note:
* <font face="Courier New">DATASET_FOLDER</font> points to the RAISE8K split into Training/Validation/Test (e.g. by means of *Split_RAISE8K.py*)
* <font face="Courier New">JPEG_Q=*None*</font> means that patches are saved as uncompressed PNG. To save them as compressed JPEG,
use a value in [0, 100]. Note that, rather than physically creating JPEG compressed images on hard drives, we chose
to augment uncompressed images during training phase.
* <font face="Courier New">CLAHE_CLIP</font> is the clip limit parameter of CLAHE (OpenCV2 implementation)
* MAX_TRAIN_BLOCKS, MAX_VAL_BLOCKS and MAX_TEST_BLOCKS are the number of patches for each class (pristine and enhanced), hence 
each dataset consists of twice these amounts of patches
* <font face="Courier New">MAX_PER_IMAGE</font> limits the number of patches that are extracted from the same image to
obtain more variety within datasets. Patches are randomly chosen from each image. Make sure you have enough images when
setting MAX_PER_IMAGE's value 
* Patches will be stored in TRAIN_FOLDER, VALIDATION_FOLDER and TEST_FOLDER. Each folder has two sub-folders: *original* 
and *enhanced*


#### Dataset creation

With *configuration.py* modified according to your paths, you can create training, validation and test sets as follows:

```
python Make_dataset.py
```
>Creating data set D:/Datasets/RAISE8K/train: <br>
> <br>
>15:05:11 Processed 0 of 6500 (0.0%) /    0 of 1000000 patches (0.0%) (hs). Elapsed: 0.7 seconds)<br>
>15:05:12 Processed 1 of 6500 (0.0%) /  500 of 1000000 patches (0.1%) (g0). Elapsed: 1.7 seconds)<br>
>15:05:13 Processed 2 of 6500 (0.0%) / 1000 of 1000000 patches (0.1%) (hs). Elapsed: 3.2 seconds)<br>
> ... <br>
>15:43:05 Processed   1999 of   6500 (30.8%) / 999500 of 1000000 patches (100.0%) (hs). Elapsed: 2274.8 seconds) <br>
>Reached 1000000.0 patches <br>
>Clahe images: 686 <br>
>Histogram stretching images: 666 <br>
>Gamma correction images: 648 <br>
> ... Done! <br>

Concerning enhanced images, it is possible to track the undergone enhancement by looking at file name suffix:
* \_cl\_: Clahe
* \_hs\_: Histogram Stretching
* \_g\<number\>\_: Gamma Correction, where \<number\> is the index of the used gamma value from GAMMA


<br>

## Training network

Training is controlled by the following parameters in *configuration.py*,
whose default values in the file correspond to those we used in our implementation:

```
CLASS_0_TAG = 'original'                        # Tag for pristine image class
CLASS_1_TAG = 'enhanced'                        # Tag for enhanced image class

...

MODEL_FOLDER = './models'                # Models are stored here
TRAIN_FOLDER = './db/train'              # Training patches
VALIDATION_FOLDER = ':/db/validation'    # Validation patches
TEST_FOLDER = ':/db/test'                # Test patches

NUM_EPOCHS = 3                               # Number of training epochs
TRAIN_BATCH = 32                             # Training batch size
VALIDATION_BATCH = 32                        # Validation batch size
TEST_BATCH = 100                             # Test batch size
```

To train a model, run the following script:
```
python Train_contrastnet.py
```

Note:
* All patches are scaled by 1/255 
* CLASS_0_TAG = 'original' has class label 0
* CLASS_1_TAG = 'enhanced' has class label 1

### Warning: known issue / bug

After updating to the latest version of Keras at them time we reworked our code,
we encountered [an error with custom preprocessing](https://github.com/keras-team/keras/issues/9624),
which may be already fixed by now. We temporarily by-passed the problem by commenting line 1244 in Keras file
*keras/preprocessing/image.py*: 

```
    #img = img.resize(width_height_tuple, resample)
```

We can do this since we are already providing to the netowrk the expected patch size.

<br>


#### Training JPEG-unaware ContrastNet

To train ContrastNet on uncompressed PNG images, run *Train_contrastnet.py* with the following setup:

```
JPEG_AUGMENTATION = False
AUG_JPEG_QFS = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, -1] # ignored, since JPEG_AUGMENTATION is False

RESUME_TRAINING = False
RESUME_MODEL = './models/model_contrastnet.h5'                   # ignored, since RESUME_TRAINING is False
```

At the end of <font face="Courier New">NUM_EPOCHS</font> a <u>JPEG-unaware</u> trained model named *model_contrastnet.h5* is created
in *./models* folder.

#### Training JPEG-aware ContrastNet (one-pass)

To train ContrastNet on compressed JPEG images, augmented on-the-fly during training, run *Train_contrastnet.py* with 
the setup below. You may modify AUG_JPEG_QFS to include the JPEG quality factors you want. 

```
JPEG_AUGMENTATION = True
AUG_JPEG_QFS = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, -1]

RESUME_TRAINING = False
RESUME_MODEL = './models/model_contrastnet.h5'                   # ignored, since RESUME_TRAINING is False
```

Note that:
* JPEG quality factor -1 means that no JPEG compression is carried out
* Our implementation assigns to each QF the same probability to be chosen (modify this behaviour from line 46 in *utils.py* if you need)

At the end of <font face="Courier New">NUM_EPOCHS</font> a <u>JPEG-aware</u> trained model named *model_contrastnet_aware.h5* is created
in *./models* folder.

<br>

#### Training JPEG-aware ContrastNet (two-pass)

This is the strategy we  adopted in the [ICIP 2018 paper](http://clem.dii.unisi.it/~vipp/files/publications/CM-icip18.pdf).

1) First, we trained for 3 epochs a JPEG-unaware model as described in *Training JPEG-unaware ContrastNet*
2) Then, we updated *configuration.py* as follows:

    ```
    JPEG_AUGMENTATION = True
    AUG_JPEG_QFS = [90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, -1]

    RESUME_TRAINING = True
    RESUME_MODEL = './models/model_contrastnet.h5'
    ```
3) Finally, we resumed training for 3 more epochs by starting from the JPEG-unaware model

At the end of 3+3 epochs a JPEG-aware trained model named *model_contrastnet_aware.h5* is created
in *./models* folder.


<br>

## Testing network

#### Testing on patches

After training, you may run:

```
python Test_contrastnet_patch.py
```

or use the script as template for your own tests. We used *Make_dataset.py* to generate test sets with arbitrary JPEG quality factors by
setting the <font face="Courier New">JPEG_Q</font> parameter in *configuration.py* to
the value we wanted to test.

#### Testing on full-frame images

To perform classification on a full-image see the example in:
```
Test_contrastnet_fullimage.py
```

The image under analysis is divided into patches, which are classified by means of one of the trained models.
The final decision is taken as the average class score over all patches.

<br>


