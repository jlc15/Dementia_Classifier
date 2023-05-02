import pandas as pd
import numpy as np  # helper libraries
from tensorflow.python.data.experimental.ops.readers import CsvDatasetV2
import tensorflow_datasets as tfds
# Make numpy values easier to read.
# np.set_printoptions(precision=3, suppress=True)
import matplotlib.pyplot as plt  # helper libraries
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import layers
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import numpy as np
from nibabel.testing import data_path
import matplotlib.image as mpimg

import pandas as pd

# Quantify subjects for AD, FTD, Controls with fMRI and DTI imaging
AD_fMRI_DTI = pd.read_csv("fMRI_DTI_AD_5_01_2023.csv")
print(len(pd.unique(AD_fMRI_DTI['Subject'])))

CN_fMRI_DTI = pd.read_csv("CN_fMRI_DWI.csv")
print(len(pd.unique(CN_fMRI_DTI['Subject ID'])))

FTD_fMRI_DTI = pd.read_csv("FTD_DWI_fMRI.csv")
print(len(pd.unique(FTD_fMRI_DTI['Subject ID'])))

Control_NIFD_fMRI_DTI = pd.read_csv("Control_NIFD_DWI_fMRI.csv")
print(len(pd.unique(Control_NIFD_fMRI_DTI['Subject ID'])))

# def convert_NII_PNG():
#     python3 nii2png.py - i < inputfile > -o < outputfolder >

# def rgb2gray(rgb):
#     return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# img = mpimg.imread('image.png')
# gray = rgb2gray(img)
# plt.imshow(gray, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
# plt.show()


# Input
# Output
# Parameters takes nii_Filename of type string
def more_Than_One():
    for i in range(5):
        plt.subplot(5, 5,i + 1)
        plt.imshow(test[:,:,59 + i])
        plt.gcf().set_size_inches(10, 10)
        plt.show()

def open_NII(data_path, nii_Filename, multiple):
    if multiple == 1:
        more_Than_One()
    else:
        example_Filename = os.path.join(data_path, nii_Filename)
        img = nib.load(example_Filename).get_fdata()
        test_Load = img[:, :, 59]
        plt.figure()
        plt.imshow(test_Load)
        plt.colorbar()
        plt.grid(False)
        plt.show()

    #plt.imshow(test_Load)
    #plt.show()

open_NII(data_path, "HCP_mgh_1033_MR_T2_GradWarped_and_Defaced_Br_20140919152047293_S227858_I444392.nii", 0)
example_Filename = os.path.join(data_path,
                                    "HCP_mgh_1033_MR_T2_GradWarped_and_Defaced_Br_20140919152047293_S227858_I444392.nii")
img = nib.load(example_Filename).get_fdata()
# img.shape (320, 320, 256)
# length of the training labels must equal length of training set
# len(train_labels)
# Each label is an integer between 0 and 3:
# train_labels
test_Load = img[:, :, 60]
def test(test_IMG):
    m = 0
    for p in test_IMG:
        for i in p:
            for ii in i:
                if ii > m:
                    m = ii
    return m

m = test(img)
# Each image is mapped to a single label
class_names = ['Alzheimer', 'Vascular', 'Frontotemporal', 'Control']
#training = # first dimension should be length of training set; second dimension will be a matrix
#img.shape
#(128, 96, 24, 2)


# Input is CSV filename
# Output is pandas dataframe
def get_CSV(filename):
    return pd.read_csv(filename)  # returns df

# Input is dataframe
# Output is tensorflow dataset
def convert_DF_to_DS(dataframe):
    return tf.data.Dataset.from_tensor_slices(dict(dataframe))


def get_dataset_partitions_tf(ds, length_DS, train_split=0.8, val_split=0.1, test_split=0.1, shuffle=True,
                              shuffle_size=10000):
    assert (train_split + test_split + val_split) == 1

    if shuffle:
        # Specify seed to always have the same split distribution between runs

        ds_Shuffled = ds.shuffle(shuffle_size, seed=12)  # , shuffle_each_iteration=False)

    train_size = int(train_split * length_DS)
    val_size = int(val_split * length_DS)

    train_ds = ds_Shuffled.take(train_size)
    val_ds = ds_Shuffled.skip(train_size).take(val_size)
    test_ds = ds_Shuffled.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds


df_ADNI = get_CSV("~/Documents/DSP/Dementia_Classifier/TEST.csv")

ds_ADNI = convert_DF_to_DS(df_ADNI)
train_ADNI, val_ADNI, test_ADNI = get_dataset_partitions_tf(ds_ADNI, (len(ds_ADNI)))

import numpy as np
import matplotlib.pyplot as plt

# dataframe['thal'] = dataframe.thal.cat.codes
# dataframe.head()

# dataset_ADNI_Shuffled = dataset_ADNI.random.shuffle(
#    value, seed=None, name=None)


# train_ADNI, val_ADNI, test_ADNI = get_dataset_partitions_tf(dataset_ADNI, length_ADNI)

# train_size = int(0.7 * DATASET_SIZE)
# val_size = int(0.15 * DATASET_SIZE)
# test_size = int(0.15 * DATASET_SIZE)
# test_set
# valid_set
# train_set = tfds.load("~/Documents/DSP/Dementia_Classifier/TEST.csv",
#                                           split=["test", "train[0%:20%]", "train[20%:]"],
#                                           as_supervised=True, with_info=True)
#
# print("Train set size: ", len(train_set)) # Train set size:  40000
# print("Test set size: ", len(test_set))   # Test set size:  10000
# print("Valid set size: ", len(valid_set)) # Valid set size:  10000
# print(tf.__version__)


# training_Half = len(dataset_ADNI) / 2

#     record_defaults,
#     compression_type=None,
#     buffer_size=None,
#     header=False,
#     field_delim=',',
#     use_quote_delim=True,
#     na_value='',
#     select_cols=None,
#     exclude_cols=None
# )

# (train_images, train_labels), (test_images, test_labels) = dataset_ADNI.load_data()
# separate labels and features
# dataset_train_features = dataset_train.copy()
# dataset_train_labels = dataset_train_features.pop('Subject ID')

# dataset_train_features_NP = np.array(dataset_train_features)

# for batch, label in dataset_train.take(1):
# for key, value in batch.items():
#  print(f"{key:10s}: {value}")

# names=["Subject_ID", "Project", "Phase", "Sex", "Age",
#            "Description"]
