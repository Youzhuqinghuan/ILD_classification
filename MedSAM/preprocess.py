# -*- coding: utf-8 -*-
# %% import packages
import argparse
import numpy as np
import SimpleITK as sitk
import os
from skimage import transform
from tqdm import tqdm


# Set up argument parser
parser = argparse.ArgumentParser(description='Convert NIfTI images to NPZ files.')
parser.add_argument('--modality', type=str, default='CT', help='Modality type (default: CT)')
parser.add_argument('--nii_path', type=str, default='/jizhicfs/chengpenghu/med_datasets/Dataset503_ILD/imagesTr',
                    help='Path to the NIfTI images')
parser.add_argument('--gt_path', type=str, default='/jizhicfs/chengpenghu/med_datasets/Dataset503_ILD/labelsTr',
                    help='Path to the ground truth')
parser.add_argument('--npy_path', type=str, default='/jizhicfs/chengpenghu/med_datasets/data_preprocessed/npy/BIN_CT',
                    help='Path to save the NPZ files')

args = parser.parse_args()

# Convert nii image to npz files, including original image and corresponding masks
modality = args.modality
img_name_suffix = "_0000.nii.gz"
gt_name_suffix = ".nii.gz"
prefix = modality + "_"

nii_path = args.nii_path  # Path to the nii images
gt_path = args.gt_path  # Path to the ground truth
npy_path = args.npy_path  # Path to save the npz files
os.makedirs(os.path.join(npy_path, "gts"), exist_ok=True)
os.makedirs(os.path.join(npy_path, "imgs"), exist_ok=True)

image_size = 1024
# voxel_num_thre2d = 100
# voxel_num_thre3d = 1000

names = sorted(os.listdir(gt_path))
print(f"Original number of files: {len(names)=}")
names = [
    name
    for name in names
    if os.path.exists(os.path.join(nii_path, name.split(gt_name_suffix)[0] + img_name_suffix))
]
print(f"Number of files after sanity check: {len(names)=}")

# Set window level and width (for CT images)
WINDOW_LEVEL = 40  # Only for CT images
WINDOW_WIDTH = 400  # Only for CT images

# %% Save preprocessed images and masks as npz files
for name in tqdm(names):  # Process all files
    image_name = name.split(gt_name_suffix)[0] + img_name_suffix
    gt_name = name
    
    gt_sitk = sitk.ReadImage(os.path.join(gt_path, gt_name))
    gt_data_ori = np.uint8(sitk.GetArrayFromImage(gt_sitk))
    
    # # Exclude the objects with less than 1000 pixels in 3D
    # gt_data_ori = cc3d.dust(
    #     gt_data_ori, threshold=voxel_num_thre3d, connectivity=26, in_place=True
    # )
    
    # # Remove small objects with less than 100 pixels in 2D slices
    # for slice_i in range(gt_data_ori.shape[0]):
    #     gt_i = gt_data_ori[slice_i, :, :]
    #     gt_data_ori[slice_i, :, :] = cc3d.dust(
    #         gt_i, threshold=voxel_num_thre2d, connectivity=8, in_place=True
    #     )
    
    # Find non-zero slices
    z_index, _, _ = np.where(gt_data_ori > 0)
    z_index = np.unique(z_index)

    if len(z_index) > 0:
        # Crop the ground truth with non-zero slices
        gt_roi = gt_data_ori[z_index, :, :]
        
        # Load image and preprocess
        img_sitk = sitk.ReadImage(os.path.join(nii_path, image_name))
        image_data = sitk.GetArrayFromImage(img_sitk)
        
        # Nii preprocess start
        lower_bound = WINDOW_LEVEL - WINDOW_WIDTH / 2
        upper_bound = WINDOW_LEVEL + WINDOW_WIDTH / 2
        image_data_pre = np.clip(image_data, lower_bound, upper_bound)
        image_data_pre = (
            (image_data_pre - np.min(image_data_pre))
            / (np.max(image_data_pre) - np.min(image_data_pre))
            * 255.0
        )
        
        image_data_pre = np.uint8(image_data_pre)
        img_roi = image_data_pre[z_index, :, :]
        
        # np.savez_compressed(os.path.join(npy_path, prefix + gt_name.split(gt_name_suffix)[0]+'.npz'), imgs=img_roi, gts=gt_roi, spacing=img_sitk.GetSpacing())
        
        # Save the image and ground truth as nii files for sanity check; they can be removed
        # img_roi_sitk = sitk.GetImageFromArray(img_roi)
        # gt_roi_sitk = sitk.GetImageFromArray(gt_roi)
        # sitk.WriteImage(
        #     img_roi_sitk,
        #     os.path.join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_img.nii.gz"),
        # )
        # sitk.WriteImage(
        #     gt_roi_sitk,
        #     os.path.join(npy_path, prefix + gt_name.split(gt_name_suffix)[0] + "_gt.nii.gz"),
        # )
        
        # Save each CT image as npy file
        for i in range(img_roi.shape[0]):
            img_i = img_roi[i, :, :]
            img_3c = np.repeat(img_i[:, :, None], 3, axis=-1)
            resize_img_skimg = transform.resize(
                img_3c,
                (image_size, image_size),
                order=3,
                preserve_range=True,
                mode="constant",
                anti_aliasing=True,
            )
            resize_img_skimg_01 = (resize_img_skimg - resize_img_skimg.min()) / np.clip(
                resize_img_skimg.max() - resize_img_skimg.min(), a_min=1e-8, a_max=None
            )  # Normalize to [0, 1], (H, W, 3)
            gt_i = gt_roi[i, :, :]
            resize_gt_skimg = transform.resize(
                gt_i,
                (image_size, image_size),
                order=0,
                preserve_range=True,
                mode="constant",
                anti_aliasing=False,
            )
            resize_gt_skimg = np.uint8(resize_gt_skimg)
            assert resize_img_skimg_01.shape[:2] == resize_gt_skimg.shape
            np.save(
                os.path.join(
                    npy_path,
                    "imgs",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_img_skimg_01,
            )
            np.save(
                os.path.join(
                    npy_path,
                    "gts",
                    prefix
                    + gt_name.split(gt_name_suffix)[0]
                    + "-"
                    + str(i).zfill(3)
                    + ".npy",
                ),
                resize_gt_skimg,
            )
