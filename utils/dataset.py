from os.path import splitext
from os import listdir
from os.path import exists as os_path_exists
import pathlib
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image
import json


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, width=0, height=0, scale_factor=1.0, mask_suffix='', 
                 load_statistics=False, save_statistics=False):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.width = width
        self.height = height
        self.scale_factor = scale_factor
        self.mask_suffix = mask_suffix

        if width <= 0:
            logging.warning("Using original image width")
        if height <= 0:
            logging.warning("Using original image height")

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

        json_path = pathlib.Path(imgs_dir).parent / "statistics.json"
        if os_path_exists(json_path) and load_statistics:
            save_statistics = False
            with open(json_path) as json_file:
                dataset_statistics = json.load(json_file)
        else: 
            img_files = glob(self.imgs_dir + '*.*')
            dataset_statistics = self.get_dataset_mean_std(img_files)
        self.dataset_mean = dataset_statistics["mean"]
        self.dataset_std = dataset_statistics["std"]
        logging.info(f'Dataset pixel mean: {self.dataset_mean}')
        logging.info(f'Dataset pixel std: {self.dataset_std}')
        if save_statistics:
            with open(json_path, 'w+') as json_file:
                json_statistics = {}
                json_statistics["mean"] = list(self.dataset_mean)
                json_statistics["std"] = list(self.dataset_std)
                json.dump(json_statistics, json_file)

    def __len__(self):
        return len(self.ids)

    @classmethod
    def get_dataset_mean_std(cls, image_paths_list):
        logging.info(f'Calculating dataset mean and standard deviation.')
        
        num_channels = np.array(Image.open(image_paths_list[0])).shape[-1]
        # pixel sum depending on number of channels
        pixel_sum = np.zeros(num_channels, dtype="float64")
        mean_denominator = 0
        pixel_var_sum = np.zeros(num_channels, dtype="float64")
        for image_path in image_paths_list:
            image_np = np.array(Image.open(image_path))
            pixel_sum += image_np.sum(axis=(0,1))
            mean_denominator += image_np.shape[0]*image_np.shape[1]
            pixel_var_sum += np.array(Image.open(image_path)).var(axis=(0,1), dtype="float64")
        
        mean = pixel_sum.astype("float64") / mean_denominator
        std = np.sqrt(pixel_var_sum / len(image_paths_list))
            
        return {"mean": mean, "std": std}

    @classmethod
    def preprocess(cls, pil_img, width, height, scale_factor, dataset_mean=None, dataset_std=None):
        w, h = pil_img.size
        newW = width if width > 0 else w
        newH = height if height > 0 else h

        newW = int(newW * scale_factor)
        newH = int(newH * scale_factor)
        
        pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # Standardize image
        if (dataset_mean is not None) and (dataset_std is not None):
            img_nd = (img_nd - dataset_mean) / dataset_std
            img_nd = np.clip(img_nd, -1.0, 1.0)
            # shift from [-1,1] to [0,1]
            img_nd = (img_nd + 1.0) / 2.0

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        idx = self.ids[i]
        mask_file = glob(self.masks_dir + idx + self.mask_suffix + '.*')
        img_file = glob(self.imgs_dir + idx + '.*')

        assert len(mask_file) == 1, \
            f'Either no mask or multiple masks found for the ID {idx}: {mask_file}'
        assert len(img_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {img_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.width, self.height, self.scale_factor, 
                              self.dataset_mean, self.dataset_std)
        mask = self.preprocess(mask, self.width, self.height, self.scale_factor)

        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, width, height):
        super().__init__(imgs_dir, masks_dir, width, height, mask_suffix='_mask')
