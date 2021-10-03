import math
import os
import random

from myutils.training import patch_image, from_patches
from tensorflow.keras.preprocessing.image import load_img
from PIL import Image
import numpy as np
from tensorflow.keras.utils import Sequence
from myutils.utils import process_images

class Dataset:

    def __init__(self, name):
        self._name = name
        self._patch_size = None
        self._stride = None
        self.default_directories()

    def default_directories(self):
        self._targets_dir = f"./patched_dataset/{self._name}/targets/img/"
        self._inputs_dir = f"./patched_dataset/{self._name}/inputs/img/"
        self._mask_dir = f"./patched_dataset/{self._name}/masks/img/"

    def from_directory(self, input_dir, target_dir, mask_dir):
        self._inputs_dir = input_dir
        self._targets_dir = target_dir
        self._mask_dir = mask_dir

        return self

    def from_files(self, input_files, target_files, mask_files, patch_size=None, stride=None):
        self._patch_size = patch_size
        self._stride = stride

        self.default_directories()

        try:
            os.makedirs(self._inputs_dir)
            os.makedirs(self._targets_dir)
            os.makedirs(self._mask_dir)
        except FileExistsError:
            for directory in (self._inputs_dir, self._targets_dir, self._mask_dir):
                for file in os.listdir(directory):
                    os.remove(os.path.join(directory, file))

        for directory, files in [(self._inputs_dir, input_files),
                                (self._targets_dir, target_files),
                                (self._mask_dir, mask_files)]:
            for file_path in files:
                file_name, extension = file_path.split('/')[-1].split('.')
                img = np.array(load_img(file_path))
                if patch_size is not None:
                    patches = patch_image(img, self._patch_size, self._stride)
                    for i, patch in enumerate(patches):
                        im = Image.fromarray(patch)
                        im.save(os.path.join(directory, f"{file_name}_{i:03d}.{extension}"))
                else:
                    im = Image.fromarray(img)
                    im.save(os.path.join(directory, f"{file_name}.{extension}"))

        return self

    @property
    def images(self):
        images = np.array([
            np.array(load_img(path)) for path in self.x_paths
        ])
        images = images.astype("float32") / 255
        return images
    
    @property
    def targets(self):
        targets = np.expand_dims(np.array([
            np.array(load_img(path, color_mode='grayscale')) for path in self.y_paths
        ]), -1)
        targets = np.clip(targets.astype("float32"), 0, 1)
        return targets

    @property
    def masks(self):
        return np.expand_dims(np.array([
            np.array(load_img(path, color_mode='grayscale')) for path in self.m_paths
        ]), -1)
        masks = np.clip(masks.astype("float32"), 0, 1)
        return masks

    @property
    def targets_img_directory(self):
        return self._targets_dir[:-4]

    @property
    def inputs_img_directory(self):
        return self._inputs_dir[:-4]
    
    @property
    def targets_directory(self):
        return self._targets_dir

    @property
    def inputs_directory(self):
        return self._inputs_dir

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def stride(self):
        return self._stride

    @property
    def x_paths(self):
        return sorted([os.path.join(self._inputs_dir, fname) for fname in os.listdir(self._inputs_dir)])

    @property
    def y_paths(self):
        return sorted([os.path.join(self._targets_dir, fname) for fname in os.listdir(self._targets_dir)])

    @property
    def m_paths(self):
        return sorted([os.path.join(self._mask_dir, fn) for fn in os.listdir(self._mask_dir)])

class PatchedDataset:

    def __init__(self, input_imgs_path, target_imgs_path, patch_size, stride, name):

        self._input_imgs_path = input_imgs_path
        self._target_imgs_path = target_imgs_path
        self._patch_size = patch_size
        self._stride = stride
        self._targets_dir = f"./patched_dataset/{name}/targets/img/"
        self._inputs_dir = f"./patched_dataset/{name}/inputs/img/"
        self._create_directories()

    def _create_directories(self):

        try:
            os.makedirs(self._inputs_dir)
            os.makedirs(self._targets_dir)
        except FileExistsError:
            pass

        for directory, path in [(self._inputs_dir, self._input_imgs_path),
                                (self._targets_dir, self._target_imgs_path)]:
            for file in os.listdir(path):
                file_name, extension = file.split('.')
                img = np.array(load_img(os.path.join(path, file)))
                patches = patch_image(img, self._patch_size, self._stride)
                for i, patch in enumerate(patches):
                    im = Image.fromarray(patch)
                    im.save(os.path.join(directory, f"{file_name}_{i:03d}.{extension}"))

    @property
    def targets_img_directory(self):
        return self._targets_dir[:-4]

    @property
    def inputs_img_directory(self):
        return self._inputs_dir[:-4]
    
    @property
    def targets_directory(self):
        return self._targets_dir

    @property
    def inputs_directory(self):
        return self._inputs_dir

    @property
    def patch_size(self):
        return self._patch_size

    @property
    def stride(self):
        return self._stride

    @property
    def original_filenames(self):
        return sorted(list(set((fname[:-8] for fname in os.listdir(self._inputs_dir)))))

    @property
    def xy_paths(self):
        return sorted([os.path.join(self._inputs_dir, fname) for fname in os.listdir(self._inputs_dir)]), sorted([os.path.join(self._targets_dir, fname) for fname in os.listdir(self._targets_dir)])

    @property
    def x_paths(self):
        return self.xy_paths[0]

    @property
    def y_paths(self):
        return self.xy_paths[1]



class PatchedSequence(Sequence):

    def __init__(self, x_set, y_set, m_set, batch_size, shuffle=False, shuffle_epoch_end=False, random_seed=None):
        self.x, self.y = sorted(x_set), sorted(y_set)
        self.batch_size = batch_size
        self._shuffle = shuffle
        self._shuffle_epoch_end = shuffle_epoch_end
        self._random_seed = random_seed
        
        if self._shuffle:
            self._shuffle_datasets()


    def _shuffle_datasets(self):
        if self._random_seed is not None:
            random.Random(self._random_seed).shuffle(self.x)
            random.Random(self._random_seed).shuffle(self.y)
        else:
            random.shuffle(self.x)
            random.shuffle(self.y)

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                                               self.batch_size]
        X, Y = [], []
        for x_path, y_path in zip(batch_x, batch_y):
            x, y = load_img(x_path), load_img(y_path,  color_mode='grayscale')
            x, y = process_images(x, y)
            X.append(x)
            Y.append(y)
        X, Y = np.asarray(X), np.asarray(Y)
        return X, Y


    def on_epoch_end(self):
        if self._shuffle_epoch_end:
            self._shuffle_datasets()
