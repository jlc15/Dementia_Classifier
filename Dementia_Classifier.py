# Structural MR images; T1-weighted images come in the flavors: MPRAGE
import pandas as pd
import numpy as np  # helper libraries
from tensorflow.python.data.experimental.ops.readers import CsvDatasetV2
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt  # helper libraries
# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow.keras import layers
import nibabel as nib
import matplotlib.pyplot as plt
import os
import matplotlib.image as mpimg
from nibabel import load
import ants

# Preprocessing

# Standardize resolution: any non-standard resolution will be standardized by resampling the image
# with cubic B-splines for best quality per https://towardsdatascience.com/deep-learning-with-magnetic-resonance-and-computed-tomography-images-e9f32273dcb5
ants_Img = ants.image_read("Test_1.nii")
resampled_Img = ants.resample_image(ants_Img, [1, 1, 1], use_voxels=False, interp_type=4)
ants.plot(resampled_Img)

ants_Img_Moving = ants.image_read("Test_2.nii")
resampled_Img_Moving = ants.resample_image(ants_Img_Moving, [1, 1, 1], use_voxels=False, interp_type=4)
ants.plot(resampled_Img_Moving)

#  bias-field correction because of inhomogeneous image intensities due to the scanner in MR images
image_n4 = ants.n4_bias_field_correction(resampled_Img)
image_n4_Moving = ants.n4_bias_field_correction(resampled_Img_Moving)

# Intensity normalization with histogram matching
#template_img_path = os.path.join('Base_DIR', 'assets', 'templates', 'mni_ic')
#template_img_sitk = sitk.ReadImage(template)
#def normalize_intensity(img_tensor, normalization="mean"):
#   """
#   Accepts an image tensor and normalizes it
#   :param normalization: choices = "max", "mean" , type=str
#   For mean normalization we use the non zero voxels only.
#   """
#   if normalization == "mean":
#       mask = img_tensor.ne(0.0)
#       desired = img_tensor[mask]
#       mean_val, std_val = desired.mean(), desired.std()
#       img_tensor = (img_tensor - mean_val) / std_val
#   elif normalization == "max":
#     MAX, MIN = img_tensor.max(), img_tensor.min()
#     img_tensor = (img_tensor - MIN) / (MAX - MIN)
#   return img_tensor

# image_n4_Normalized = normalize_intensity(image_n4, "mean")

# Registration to align multiple images to ensure spatial correspondence of anatomy across different images
mytx = ants.registration(fixed=image_n4, moving=image_n4_Moving, type_of_transform='SyN')
print(mytx)
warped_moving = mytx['warpedmovout']
resampled_Img.plot(overlay=warped_moving,
           title='After Registration')

# Input
# Output: None
# Parameters takes nii_Filename of type string; NIfTI corresponds to .nii extension
# Display
def more_Than_One():
    for i in range(5):
        plt.subplot(5, 5, i + 1)
        plt.imshow(test[:, :, 59 + i])
        plt.gcf().set_size_inches(10, 10)
        plt.show()


def open_NII(nii_Filename, multiple):
    if multiple == 1:
        more_Than_One()
    else:
        #example_Filename = os.path.join(data_path, nii_Filename)
        img = load(nii_Filename).get_fdata()
        test_Load = img[:, :, 59]
        plt.figure()
        plt.imshow(test_Load)
        plt.colorbar()
        plt.grid(False)
        plt.show()


open_NII("Test_1.nii", 0)





# MR are inconsistent tissue intensities across different MR scanners
