import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tvF
from torch.utils.data import Dataset, DataLoader

import os
import numpy as np
from string import ascii_letters
from PIL import Image, ImageFont, ImageDraw

import random
from random import choice


class NoisyDataset(Dataset):
    
    def __init__(self, root_dir, crop_size=128, train_noise_model=('gaussian', 50), clean_targ=False):
        self.root_dir = root_dir
        self.crop_size = crop_size
        self.clean_targ = clean_targ
        self.noise = train_noise_model[0]
        self.noise_param = train_noise_model[1]
        self.imgs = os.listdir(root_dir)

    
    def _random_crop_to_size(self, imgs):
        
        width, height = imgs[0].size
        assert width >= self.crop_size and height >= self.crop_size, 'Cannot be croppped. Invalid size'

        cropped_imgs = []
        i = np.random.randint(0, height - self.crop_size + 2)
        j = np.random.randint(0, width - self.crop_size + 2)

        for img in imgs:
            if min(width, height) < self.crop_size:
                img = tvF.resize(img, (self.crop_size, self.crop_size))
            cropped_imgs.append(tvF.crop(img, i, j, self.crop_size, self.crop_size))
        
        return cropped_imgs
    
    def _add_gaussian_noise(self, image):
        width, height = image.size
        c = len(image.getbands())
        
        std = np.random.uniform(0, self.noise_param)
        _noise = np.random.normal(0, std, (height, width, c))
        noisy_image = np.array(image) + _noise
        
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        return {'image':Image.fromarray(noisy_image), 'mask': None, 'use_mask': False}

    def _add_bernoulli_noise(self, image):
        size = np.array(image).shape[0]
        prob_ = random.uniform(0, self.noise_param)
        mask = np.random.choice([0, 1], size=(size, size), p=[prob_, 1 - prob_])
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        return {'image':np.multiply(image, mask).astype(np.uint8), 'mask':mask.astype(np.uint8), 'use_mask': True}


    def _add_text_overlay(self, image):

        text_img = image.copy()
        text_draw = ImageDraw.Draw(text_img)

        width, height = image.size
        mask_img = Image.new('RGBA', (width, height))
        mask_draw = ImageDraw.Draw(mask_img)

        serif = 'font/arial.ttf'
        font = ImageFont.truetype(serif, np.random.randint(16, 21))

        length = np.random.randint(10, 25)
        chars = ''.join(choice(ascii_letters) for i in range(length))

        pos = (np.random.randint(0, self.crop_size), np.random.randint(0, self.crop_size))
                
        text_draw.text(pos, chars, font=font, fill=(255, 255, 255, 20))
        mask_draw.text(pos, chars, font=font, fill=(255, 255, 255, 20))
        
        return {'image':text_img, 'mask':None, 'use_mask': False}

    def corrupt_image(self, image):
        
        if self.noise == 'gaussian':
            return self._add_gaussian_noise(image)
        elif self.noise == 'multiplicative_bernoulli':
            return self._add_bernoulli_noise(image)
        elif self.noise == 'text':
            return self._add_text_overlay(image)
        else:
            raise ValueError('No such image corruption supported')

    def __getitem__(self, index):
        """
        Read a image, corrupt it and return it
        """
        img_path = os.path.join(self.root_dir, self.imgs[index])
        image = Image.open(img_path).convert('RGB')

        if self.crop_size > 0:
            image = self._random_crop_to_size([image])[0]

        source_img_dict = self.corrupt_image(image)
        source_img_dict['image'] = tvF.to_tensor(source_img_dict['image'])

        if source_img_dict['use_mask']:
            source_img_dict['mask'] = tvF.to_tensor(source_img_dict['mask'])

        if self.clean_targ:
            target = tvF.to_tensor(image)
        else:
            _target_dict = self.corrupt_image(image)
            target = tvF.to_tensor(_target_dict['image'])

        if source_img_dict['use_mask']:
            return [source_img_dict['image'], source_img_dict['mask'], target]
        else:
            return [source_img_dict['image'], target]

    def __len__(self):
        return len(self.imgs)

