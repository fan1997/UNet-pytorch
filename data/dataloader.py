import os
from collections import deque
import numpy as np
from PIL import Image, ImageSequence
import torch
from torchvision import transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset
from tqdm import tqdm


def _load_multipage_tiff(path):
    """Load tiff images containing many images in the channel dimension"""
    a = []
    for p in ImageSequence.Iterator(Image.open(path)):
        a.append(Image.fromarray(np.array(p)))
    return a

def _get_val_train_indices(length, fold, ratio=0.8):
    assert 0 < ratio <= 1, "Train/total data ratio must be in range (0.0, 1.0]"
    np.random.seed(0)
    indices = np.arange(0, length, 1, dtype=np.int)
    np.random.shuffle(indices)
    if fold is not None:
        indices = deque(indices)
        indices.rotate(fold * round((1.0 - ratio) * length))
        indices = np.array(indices)
        train_indices = indices[:round(ratio * len(indices))]
        val_indices = indices[round(ratio * len(indices)):]
    else:
        train_indices = indices
        val_indices = []
    return train_indices, val_indices


def data_post_process(img, mask):
    img = np.expand_dims(img, axis=0)
    mask = np.expand_dims(mask, axis=0)
    mask = (mask > 0.5).astype(np.int)
    return img.astype(np.float32), mask.astype(np.float32)
def train_data_augmentation(img, mask):
    h_flip = np.random.random()
    if h_flip > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)
    v_flip = np.random.random()
    if v_flip > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
    left = int(np.random.uniform()*0.3*572)
    right = int((1-np.random.uniform()*0.3)*572)
    top = int(np.random.uniform()*0.3*572)
    bottom = int((1-np.random.uniform()*0.3)*572)
    img = img[top:bottom, left:right]
    mask = mask[top:bottom, left:right]
    #adjust brightness
    brightness = np.random.uniform(-0.2, 0.2)
    img = np.float32(img+brightness*np.ones(img.shape))
    mask = np.float32(mask)
    img = np.clip(img, -1.0, 1.0)
    return img, mask


class UnetDataset(Dataset):
    def __init__(self, data_dir, augment=False, cross_val_ind=1, repeat = 1):
        self.data_dir = data_dir
        self.images = _load_multipage_tiff(os.path.join(data_dir, 'train-volume.tif'))
        self.masks = _load_multipage_tiff(os.path.join(data_dir, 'train-labels.tif'))
        train_indices, val_indices = _get_val_train_indices(len(self.images), cross_val_ind)
        self.train_images = [self.images[x] for x in train_indices]
        self.train_masks = [self.masks[x] for x in train_indices]
        self.val_images = [self.images[x] for x in val_indices]
        self.val_masks = [self.masks[x] for x in val_indices]    
        t_resize_572 = transforms.Resize(size=(572, 572))
        t_resize_388 = transforms.Resize(size=(388, 388))
        t_pad = transforms.Pad(padding=92)
        t_center_crop = transforms.CenterCrop(size=388)
        self.trans_image = transforms.Compose([
            t_resize_388,
            t_pad,
        ])
        self.trans_mask = transforms.Compose([
            t_resize_388,
            t_pad,
        ])
    def __len__(self):
        return 24
    def __getitem__(self, idx):
        image = np.array(self.train_images[idx], dtype=np.float32)
        mask = np.array(self.train_masks[idx], dtype=np.float32)
        image = image / 127.5 - 1.0
        mask = mask / 255.0
        image = transforms.ToPILImage()(image)
        mask = transforms.ToPILImage()(mask)
        image = self.trans_image(image)
        mask = self.trans_mask(mask)
        image = np.array(image)
        mask = np.array(mask)
        image, mask = train_data_augmentation(image, mask)
        image = transforms.ToPILImage()(image)
        mask = transforms.ToPILImage()(mask)
        t_resize_572 = transforms.Resize(size=(572, 572))
        image = t_resize_572(image)
        mask = t_resize_572(mask)
        t_center_crop = transforms.CenterCrop(size=388)
        mask = t_center_crop(mask)
        image = np.array(image)
        mask = np.array(mask)
        image, mask= data_post_process(image, mask)
        sample = {'image': image, 'mask': mask}
        return sample

def create_dataset(data_dir, repeat=400, train_batch_size=16, augment=False, cross_val_ind=1, run_distribute=False):
    udataset = UnetDataset(data_dir)
    train_dataloader = DataLoader(
        dataset=udataset, batch_size=train_batch_size, shuffle=True, num_workers=1)
    return train_dataloader, 1

