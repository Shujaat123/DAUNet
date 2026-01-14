import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import torch
import torchvision.transforms.functional as TF

import os
import SimpleITK as sitk


def normalize_img(img):
    return torch.from_numpy(img).float().div(255.0) * 2 - 1  # [0,255] → [0,1] → [-1,1]

class PUBIC(Dataset):
    # def __init__(self, dataroot, img_size, split='train', augment=False, data_len=-1):
    def __init__(self, dataroot, img_size, split='train', augment=False):
        self.img_size = img_size
        self.split = split
        self.augment = augment and (split == "train")

        img_root = os.path.join(dataroot, self.split+'dataset/image_mha/')
        gt_root = os.path.join(dataroot, self.split+'dataset/label_mha/')
        self.img_paths = sorted([os.path.join(img_root, f) for f in os.listdir(img_root) if f.endswith('.mha')])
        self.gt_paths = sorted([os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.mha')])
        assert len(self.img_paths) == len(self.gt_paths), "Mismatch in number of images and labels"

    def __len__(self):
        return len(self.img_paths)


    def __getitem__(self, index):
        img_path = self.img_paths[index]
        # gt_path = self.gt_paths[index]
        gt_path = img_path.replace("image_mha", "label_mha")

        # Load 3-slice image volume
        img_vol = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img_vol)  # shape: [3, H, W]
        assert img_array.shape[0] == 3, f"Expected 3 slices, got {img_array.shape[0]}"

        # Load corresponding label (2D segmentation mask)
        # --- label: [H,W] in {0,1,2} ---
        label_vol   = sitk.ReadImage(gt_path)
        label_array = sitk.GetArrayFromImage(label_vol).astype(np.uint8)  # [H,W], values 0/1/2

        # --- Resize and normalize image slices
        img_tensor = []
        for i in range(3):
            img_slice = Image.fromarray(img_array[i]).resize((self.img_size, self.img_size), resample=Image.BILINEAR)
            img_slice_np = np.array(img_slice).astype(np.float32) / 255.0  # [0, 1]
            img_tensor.append(torch.from_numpy(img_slice_np * 2.0 - 1.0))  # [0,1] → [-1,1]
        img = torch.stack(img_tensor, dim=0)  # shape: [3, H, W]

        # Resize label (NEAREST to preserve boundaries), convert to tensor
        label_img = Image.fromarray(label_array).resize((self.img_size, self.img_size), resample=Image.NEAREST)
        # label = torch.from_numpy(np.array(label_img, dtype=np.int64)).unsqueeze(0)  # [1,H,W], long
        label = torch.from_numpy(np.array(label_img, dtype=np.int64))  # [H,W], long

        case_name = os.path.basename(img_path).split('.')[0]

        if self.augment:
            img, label = _geom_augment(img, label)
            img = _intensity_augment(img)

        return {'LD': img, 'FD': label, 'case_name': case_name}


def _geom_augment(img, mask):
    # img: torch.FloatTensor [3,H,W] in [-1,1]; mask: torch.LongTensor [H,W]
    import random
    if random.random() < 0.5:
        img  = TF.hflip(img);  mask = TF.hflip(mask)
    if random.random() < 0.5:
        img  = TF.vflip(img);  mask = TF.vflip(mask)
    angle = random.uniform(-10.0, 10.0)
    scale = random.uniform(0.9, 1.1)
    img  = TF.affine(img, angle=angle, translate=(0,0), scale=scale, shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.BILINEAR)
    m    = TF.affine(mask.unsqueeze(0).float(), angle=angle, translate=(0,0), scale=scale, shear=[0.0, 0.0],
                     interpolation=TF.InterpolationMode.NEAREST).squeeze(0).long()
    return img, m

def _intensity_augment(img):
    import random, torch
    if random.random() < 0.5:
        gamma = random.uniform(0.9, 1.1)
        x = ((img + 1.0) * 0.5).clamp(0,1)
        x = x.pow(gamma)
        img = (x * 2.0 - 1.0).clamp(-1,1)
    if random.random() < 0.5:
        b = random.uniform(-0.05, 0.05)
        img = (img + b).clamp(-1,1)
    if random.random() < 0.5:
        c = random.uniform(0.9, 1.1)
        mean = img.mean(dim=(1,2), keepdim=True)
        img = ((img - mean) * c + mean).clamp(-1,1)
    return img
