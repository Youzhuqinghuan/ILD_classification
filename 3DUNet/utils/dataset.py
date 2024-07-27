from posixpath import join
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import SimpleITK as sitk
import torch
from torch.utils.data import Dataset as dataset
# from .transforms import RandomCrop, RandomFlip_LR, RandomFlip_UD, Center_Crop, Compose, Resize

class CTDataset(dataset):
    def __init__(self, args):
        self.args = args
        self.data_root = self.args.dataset_path
        self.gt_path = os.path.join(self.data_root, "labelsTr")
        self.img_path = os.path.join(self.data_root, "imagesTr")
        self.filename_list = os.listdir(self.img_path)  # 获取所有图像文件名列表
        self.transforms = None
        # self.transforms = Compose([
        #         RandomCrop(self.args.crop_size),
        #         RandomFlip_LR(prob=0.5),
        #         RandomFlip_UD(prob=0.5),
        #         # RandomRotate()
        #     ])

    def __getitem__(self, index):
        img_filename = self.filename_list[index]
        img_path = os.path.join(self.img_path, img_filename)
        
        # 构建标签文件名
        label_filename = img_filename.replace('_0000', '')
        seg_path = os.path.join(self.gt_path, label_filename)

        # 读取图像和标签
        ct = sitk.ReadImage(img_path)
        seg = sitk.ReadImage(seg_path)

        # 转换为numpy数组
        ct_array = sitk.GetArrayFromImage(ct)
        seg_array = sitk.GetArrayFromImage(seg)

        # 归一化处理
        ct_array = ct_array / self.args.norm_factor
        ct_array = ct_array.astype(np.float32)

        # 转换为PyTorch张量
        ct_array = torch.FloatTensor(ct_array).unsqueeze(0)
        seg_array = torch.FloatTensor(seg_array).unsqueeze(0)

        # 应用数据增强
        if self.transforms:
            ct_array, seg_array = self.transforms(ct_array, seg_array)

        return ct_array, seg_array.squeeze(0)

    def __len__(self):
        return len(self.filename_list)


if __name__ == "__main__":
    from config import args
    train_ds = CTDataset(args)
    # 定义数据加载
    train_dl = DataLoader(train_ds, batch_size=2, shuffle=False, num_workers=1)

    for i, (ct, seg) in enumerate(train_dl):
        print(i, ct.size(), seg.size())
