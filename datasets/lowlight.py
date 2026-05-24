import os
import random
import re
from os import listdir

import numpy as np
import PIL
import torch
import torch.utils.data
import torchvision
import torchvision.transforms as transforms


class lowlight:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    @staticmethod
    def _resolve_dir(dir_path):
        if os.path.isabs(dir_path):
            return dir_path

        cwd_path = os.path.abspath(dir_path)
        if os.path.exists(cwd_path):
            return cwd_path

        module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), dir_path))
        if os.path.exists(module_path):
            return module_path

        return cwd_path

    def get_loaders(self, parse_patches=True, validation='lowlight'):
        print("=> evaluating lowlight test set...")
        val_split = getattr(self.config.training, 'val_split', None)

        train_dir = os.path.join(self.config.data.data_dir, 'data', 'lowlight', 'train')
        test_dir = os.path.join(self.config.data.data_dir, 'data', 'lowlight', 'test')

        if val_split is None or val_split == 0:
            train_dataset = lowlightDataset(
                dir=train_dir,
                n=self.config.training.patch_n,
                patch_size=self.config.data.image_size,
                transforms=self.transforms,
                filelist=None,
                parse_patches=parse_patches,
                train=True,
            )
            val_dataset = lowlightDataset(
                dir=test_dir,
                n=self.config.training.patch_n,
                patch_size=self.config.data.image_size,
                transforms=self.transforms,
                filelist='lowlighttesta.txt',
                parse_patches=parse_patches,
                train=False,
            )
        else:
            train_inputs, train_gts, val_inputs, val_gts = self._build_split(train_dir, val_split)
            train_dataset = lowlightDataset(
                dir=train_dir,
                n=self.config.training.patch_n,
                patch_size=self.config.data.image_size,
                transforms=self.transforms,
                filelist=None,
                parse_patches=parse_patches,
                train=True,
                input_names=train_inputs,
                gt_names=train_gts,
            )
            val_dataset = lowlightDataset(
                dir=train_dir,
                n=self.config.training.patch_n,
                patch_size=self.config.data.image_size,
                transforms=self.transforms,
                filelist=None,
                parse_patches=parse_patches,
                train=False,
                input_names=val_inputs,
                gt_names=val_gts,
            )

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.sampling.batch_size,
            shuffle=False,
            num_workers=self.config.data.num_workers,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _build_split(self, train_dir, val_split):
        base_dir = self._resolve_dir(train_dir)
        input_dir = os.path.join(base_dir, 'input')
        images = sorted(f for f in listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)))

        split_file_train = os.path.join(base_dir, 'train_split.txt')
        split_file_val = os.path.join(base_dir, 'val_split.txt')

        if os.path.exists(split_file_train) and os.path.exists(split_file_val):
            with open(split_file_train, 'r') as f:
                train_inputs = [line.strip() for line in f if line.strip()]
            with open(split_file_val, 'r') as f:
                val_inputs = [line.strip() for line in f if line.strip()]
            train_gts = [self._input_to_gt_path(p) for p in train_inputs]
            val_gts = [self._input_to_gt_path(p) for p in val_inputs]
            return train_inputs, train_gts, val_inputs, val_gts

        split_idx = int(len(images) * (1.0 - float(val_split)))
        train_inputs = [os.path.join('input', name) for name in images[:split_idx]]
        train_gts = [os.path.join('gt', name) for name in images[:split_idx]]
        val_inputs = [os.path.join('input', name) for name in images[split_idx:]]
        val_gts = [os.path.join('gt', name) for name in images[split_idx:]]

        try:
            with open(split_file_train, 'w') as f:
                f.write('\n'.join(train_inputs) + ('\n' if train_inputs else ''))
            with open(split_file_val, 'w') as f:
                f.write('\n'.join(val_inputs) + ('\n' if val_inputs else ''))
        except Exception:
            pass

        return train_inputs, train_gts, val_inputs, val_gts


class lowlightDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, train, filelist=None, parse_patches=True, input_names=None, gt_names=None):
        super().__init__()
        self.dir = dir
        self.patch_size = patch_size
        self.transforms = transforms
        self.n = n
        self.parse_patches = parse_patches
        self.batchnum = 0
        self.batchsize = 1
        self.train = train

        if input_names is not None and gt_names is not None:
            self.input_names = list(input_names)
            self.gt_names = list(gt_names)
            return

        if filelist is None:
            self.input_names, self.gt_names = self._load_from_directory(dir)
        else:
            self.input_names, self.gt_names = self._load_from_filelist(dir, filelist)

    @staticmethod
    def _load_from_directory(dir_path):
        base_dir = lowlight._resolve_dir(dir_path)
        input_dir = os.path.join(base_dir, 'input')
        images = sorted(f for f in listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f)))
        print(f'Found {len(images)} images in training set')

        input_names = [os.path.join(input_dir, name) for name in images]
        gt_names = [os.path.join(base_dir, 'gt', name) for name in images]

        paired = list(zip(input_names, gt_names))
        random.shuffle(paired)
        if paired:
            input_names, gt_names = map(list, zip(*paired))
        else:
            input_names, gt_names = [], []
        return input_names, gt_names

    @staticmethod
    def _load_from_filelist(dir_path, filelist):
        base_dir = lowlight._resolve_dir(dir_path)
        train_list = os.path.join(base_dir, filelist)

        input_names = []
        gt_names = []
        with open(train_list) as f:
            for line in f:
                entry = line.strip()
                if not entry:
                    continue
                input_path = lowlightDataset._resolve_input_path(base_dir, entry)
                gt_path = lowlightDataset._resolve_gt_path(input_path)
                input_names.append(input_path)
                gt_names.append(gt_path)

        return input_names, gt_names

    @staticmethod
    def _resolve_input_path(base_dir, entry):
        if os.path.isabs(entry) and os.path.exists(entry):
            return entry

        candidate = os.path.join(base_dir, entry)
        if os.path.exists(candidate):
            return candidate

        if os.path.splitext(entry)[1]:
            return os.path.join(base_dir, 'input', entry)

        return os.path.join(base_dir, 'input', entry + '.png')

    @staticmethod
    def _resolve_gt_path(input_path):
        marker = os.path.sep + 'input' + os.path.sep
        if marker in input_path:
            return input_path.replace(marker, os.path.sep + 'gt' + os.path.sep)
        return os.path.join(os.path.dirname(input_path), 'gt', os.path.basename(input_path))

    def _input_to_gt_path(self, input_path):
        return self._resolve_gt_path(input_path)

    @staticmethod
    def get_params(img, output_size, n, random_size):
        w, h = img.size
        if random_size == 0:
            output_size = (64, 64)
        elif random_size == 1:
            output_size = (128, 128)
        else:
            output_size = (256, 256)

        th, tw = output_size
        if w == tw and h == th:
            zeros = [0 for _ in range(n)]
            sizes = [th for _ in range(n)]
            return zeros, zeros, th, tw, sizes, h, w

        i_list = [random.randint(0, h - th) for _ in range(n)]
        j_list = [random.randint(0, w - tw) for _ in range(n)]
        osize = [th for _ in range(n)]
        return i_list, j_list, th, tw, osize, h, w

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        resize_transform = transforms.Resize(64)
        for i in range(len(x)):
            new_crop = img.crop((y[i], x[i], y[i] + w, x[i] + h))
            if h != 64:
                new_crop = resize_transform(new_crop)
            crops.append(new_crop)
        return tuple(crops)

    @staticmethod
    def get_max(tensor):
        return torch.clamp(tensor.clone(), 0.0, 1.0)

    def get_images(self, index):
        if self.train:
            if self.batchnum == 0:
                self.random_size = random.randint(0, 2)
            self.batchnum += 1
            if self.batchnum == self.batchsize:
                self.batchnum = 0
        else:
            self.random_size = 0

        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]

        input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        try:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        except Exception:
            gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else PIL.Image.open(gt_name).convert('RGB')

        if self.parse_patches:
            i, j, h, w, osize, h_org, w_org = self.get_params(
                input_img,
                (self.patch_size, self.patch_size),
                self.n,
                self.random_size,
            )
            input_img = self.n_random_crops(input_img, i, j, h, w)
            gt_img = self.n_random_crops(gt_img, i, j, h, w)

            outputs = [
                torch.cat(
                    [
                        self.transforms(input_img[idx]),
                        self.get_max(self.transforms(input_img[idx])),
                        self.transforms(gt_img[idx]),
                    ],
                    dim=0,
                )
                for idx in range(self.n)
            ]

            ii = torch.tensor(i)
            jj = torch.tensor(j)
            ii = (ii / h_org) * 2 - 1
            jj = (jj / w_org) * 2 - 1
            osize = torch.tensor(osize)
            return torch.stack(outputs, dim=0), img_id, ii, jj, osize

        wd_new, ht_new = input_img.size
        wd = wd_new
        ht = ht_new

        if ht_new > wd_new and ht_new > 1024:
            wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            ht_new = 1024
        elif ht_new <= wd_new and wd_new > 1024:
            ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            wd_new = 1024

        wd_new = int(8 * np.ceil(wd_new / 8.0))
        ht_new = int(8 * np.ceil(ht_new / 8.0))
        input_img = input_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)
        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.ANTIALIAS)

        return torch.cat([self.transforms(input_img), self.get_max(self.transforms(input_img)), self.transforms(gt_img)], dim=0), img_id, wd, ht

    def __getitem__(self, index):
        return self.get_images(index)

    def __len__(self):
        return len(self.input_names)
