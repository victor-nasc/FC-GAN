import os
import random

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SRDataset(Dataset):
    def __init__(self, hr_dir, crop_size=128, scale_factor=4, train=True, return_name=False):
        self.hr_dir = hr_dir
        self.crop_size = crop_size
        self.scale_factor = scale_factor
        self.train = train
        self.hr_imgs = sorted(os.listdir(self.hr_dir))
        self.return_name = return_name

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, idx):
        hr_img = Image.open(os.path.join(self.hr_dir, self.hr_imgs[idx]))
        
        if self.train:
            hr_img = transforms.RandomCrop(self.crop_size)(hr_img)

            degree = random.choice([0, 90, 180, 270])
            hr_img = hr_img.rotate(degree)
        
        lr_crop = self._downscale(hr_img)

        transform = transforms.Compose([transforms.ToTensor()])
        
        if self.return_name:
            return transform(lr_crop), transform(hr_img), self.hr_imgs[idx]
        else:
            return transform(lr_crop), transform(hr_img)
    
    def _downscale(self, img):
        new_size = (img.width // self.scale_factor, img.height // self.scale_factor)
        return img.resize(new_size, Image.BICUBIC)