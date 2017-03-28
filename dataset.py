import os
import glob
import math
import numpy
from skimage.io import imread
from skimage.transform import resize

from chainer.dataset import dataset_mixin
from utils import im_preprocess_vgg

from fuel.datasets.hdf5 import H5PYDataset

class H5pyDataset(dataset_mixin.DatasetMixin):
    def __init__(self, path, which_set='train', load_size=None, crop_size=None, dtype=numpy.float32):
        self._dtype = dtype
        self._load_size = load_size
        self._crop_size = crop_size
        self._data_set = H5PYDataset(path, which_sets=(which_set,))

    def __len__(self):
        return self._data_set.num_examples

    def get_example(self, i):
        handle = self._data_set.open()
        data = self._data_set.get_data(handle, slice(i, i+1))
        self._data_set.close(handle)

        im = numpy.squeeze(data[0])

        w, h, _ = im.shape
        min_size = min(w, h)
        ratio = self._load_size/min_size
        rw, rh = int(math.ceil(w*ratio)), int(math.ceil(h*ratio))
        im = resize(im, (rw, rh), order=1, mode='constant')

        sx, sy = numpy.random.random_integers(0, rw-self._crop_size), numpy.random.random_integers(0, rh-self._crop_size)
        im = im[sx:sx+self._crop_size, sy:sy+self._crop_size,:]*2 - 1

        im = numpy.asarray(numpy.transpose(im, (2, 0, 1)), dtype=self._dtype)

        return im

class SuperImageDataset(dataset_mixin.DatasetMixin):
    def __init__(self, paths, root='.', load_size=None, crop_size=None, flip=False, dtype=numpy.float32):
        with open(paths) as paths_file:
            self._paths = [path.strip() for path in paths_file]
        self._root = root
        self._dtype = dtype
        self._load_size = load_size
        self._crop_size = crop_size
        self._flip = flip

    def __len__(self):
        return len(self._paths)

    def get_name(self, i):
        return os.path.basename(self._paths[i])

    def get_example(self, i):
        img = im_preprocess_vgg(imread(os.path.join(self._root, self._paths[i])), load_size=self._load_size, dtype=self._dtype)
        if self._crop_size:
            _, w, h = img.shape
            sx, sy = numpy.random.randint(0, w - self._crop_size), numpy.random.randint(0, h - self._crop_size)
            img = img[:, sx:sx+self._crop_size, sy:sy+self._crop_size]
        if self._flip and numpy.random.rand() > 0.5:
            img = img[:, :, ::-1]
        
        return img

class wganDataset(dataset_mixin.DatasetMixin):
    def __init__(self, folders, root, load_size, crop_size, dtype=numpy.float32):
        self._root = root
        self._imgs = [path for folder in folders for path in glob.glob(os.path.join(self._root, folder, '*'))]
        self._len = len(self._imgs)

        self._dtype = dtype
        self._load_size = load_size
        self._crop_size = crop_size

    def __len__(self):
        return self._len

    def get_example(self, i):
        im = imread(self._imgs[i])

        w, h, _ = im.shape
        min_size = min(w, h)
        ratio = self._load_size/min_size
        rw, rh = int(math.ceil(w*ratio)), int(math.ceil(h*ratio))
        im = resize(im, (rw, rh), order=1, mode='constant')

        sx, sy = numpy.random.randint(0, rw-self._crop_size), numpy.random.randint(0, rh-self._crop_size)
        im = im[sx:sx+self._crop_size, sy:sy+self._crop_size,:]*2 - 1

        im = numpy.asarray(numpy.transpose(im, (2, 0, 1)), dtype=self._dtype)

        return im

class ImageFolderDataset(dataset_mixin.DatasetMixin):
    def __init__(self, root, load_size, crop_size, dtype=numpy.float32):
        self._root = root
        self._imgs = glob.glob(os.path.join(self._root, '**/*.jpg'), recursive=True)
        self._len = len(self._imgs)

        self._dtype = dtype
        self._load_size = load_size
        self._crop_size = crop_size

    def __len__(self):
        return self._len

    def get_example(self, i):
        im = imread(self._imgs[i])

        w, h, _ = im.shape
        min_size = min(w, h)
        ratio = self._load_size/min_size
        rw, rh = int(math.ceil(w*ratio)), int(math.ceil(h*ratio))
        im = resize(im, (rw, rh), order=1, mode='constant')

        sx, sy = numpy.random.random_integers(0, rw-self._crop_size), numpy.random.random_integers(0, rh-self._crop_size)
        im = im[sx:sx+self._crop_size, sy:sy+self._crop_size,:]*2 - 1

        im = numpy.asarray(numpy.transpose(im, (2, 0, 1)), dtype=self._dtype)

        return im

class TransientAttributesDataset(dataset_mixin.DatasetMixin):
    def __init__(self, folders, root, min_ratio, max_ratio, load_size=None, sub_mean=True, dtype=numpy.float32):
        self._root = root
        self._folders = folders
        self._imgs_per_folder = {folder:glob.glob(os.path.join(self._root, folder, '*')) for folder in self._folders}
        self._len = numpy.sum([len(v) for v in self._imgs_per_folder.values()])

        self._dtype = dtype
        self._min_ratio = min_ratio
        self._max_ratio = max_ratio
        self._load_size = load_size
        self._sub_mean = sub_mean

    def __len__(self):
        return self._len

    def get_example(self, i):
        folder = numpy.random.choice(self._folders)
        obj_path, bg_path = numpy.random.choice(self._imgs_per_folder[folder], 2, replace=False)

        obj_vgg  = im_preprocess_vgg(imread(obj_path), self._load_size, sub_mean=self._sub_mean, dtype=numpy.float32)
        bg_vgg   = im_preprocess_vgg(imread(bg_path),  self._load_size, sub_mean=self._sub_mean, dtype=numpy.float32)

        # random rect
        width, height = (self._load_size, self._load_size) if self._load_size else obj_vgg.shape[1:]
        w = int(numpy.random.uniform(self._min_ratio, self._max_ratio)*width)
        h = int(numpy.random.uniform(self._min_ratio, self._max_ratio)*height)
        sx, sy = numpy.random.randint(0, width-w), numpy.random.randint(0, height-h)

        expand_mask = numpy.zeros((1, width, height), dtype=numpy.float32)
        expand_mask[:, sx:sx+w, sy:sy+h] = 1
        mask = numpy.zeros_like(expand_mask)
        mask[:, sx+1:sx+w-1, sy+1:sy+h-1] = 1

        return obj_vgg, bg_vgg, expand_mask, mask

class BlendingDataset(dataset_mixin.DatasetMixin):
    def __init__(self, total_examples, folders, root, ratio, load_size, crop_size, dtype=numpy.float32):
        imgs_per_folder = {folder:glob.glob(os.path.join(root, folder, '*')) for folder in folders}
        self._len = total_examples

        self._dtype = dtype
        self._load_size = load_size
        self._crop_size = crop_size
        self._size = int(self._crop_size*ratio)
        self._sx = self._crop_size//2 - self._size//2

        self._imgs = []
        for _ in range(self._len):
            folder = numpy.random.choice(folders)
            obj_path, bg_path = numpy.random.choice(imgs_per_folder[folder], 2, replace=False)
            self._imgs.append((obj_path, bg_path))

    def __len__(self):
        return self._len

    def _crop(self, im, rw, rh, sx, sy):
        im = resize(im, (rw, rh), order=1, preserve_range=False, mode='constant')
        im = im[sx:sx+self._crop_size, sy:sy+self._crop_size,:]*2 - 1
        im = numpy.transpose(im, (2, 0, 1)).astype(self._dtype)

        return im

    def get_example(self, i):
        obj_path, bg_path = self._imgs[i]
        obj = imread(obj_path)
        bg  = imread(bg_path)

        w, h, _ = obj.shape
        min_size = min(w, h)
        ratio = self._load_size/min_size
        rw, rh = int(math.ceil(w*ratio)), int(math.ceil(h*ratio)) 
        sx, sy = numpy.random.random_integers(0, rw-self._crop_size), numpy.random.random_integers(0, rh-self._crop_size)              

        obj_croped = self._crop(obj, rw, rh, sx, sy)
        bg_croped  = self._crop(bg,  rw, rh, sx, sy)

        copy_paste = bg_croped.copy()
        copy_paste[:, self._sx:self._sx+self._size, self._sx:self._sx+self._size] = obj_croped[:, self._sx:self._sx+self._size, self._sx:self._sx+self._size]

        return copy_paste, bg_croped