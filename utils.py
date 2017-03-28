import math

import numpy as np

from skimage.transform import resize

mean = np.array([103.939, 116.779, 123.68])
def im_preprocess_vgg(im, load_size=None, sub_mean=True, dtype=None, order=1, preserve_range=True):
	if not dtype:
		dtype = im.dtype
	if load_size:
		im = resize(im, (load_size, load_size), order=order, preserve_range=preserve_range, mode='constant')
	if im.ndim == 2:
		im = im[:, :, np.newaxis]
	im = im[:, :, ::-1]
	if sub_mean:
		im = im - mean
	im = np.asarray(np.transpose(im, [2, 0, 1]), dtype=dtype)
	return im

def im_deprocess_vgg(im, orig_size=None, add_mean=True, dtype=None, order=1, preserve_range=True):
	if not dtype:
		dtype = im.dtype
	im = np.transpose(im, [1, 2, 0])
	if add_mean:
		im = im + mean
	im = im[:, :, ::-1]
	im = np.squeeze(im)
	if orig_size:
		im = resize(im, orig_size, order=order, preserve_range=preserve_range, mode='constant')
	im = np.clip(im, 0, 255)
	im = np.asarray(im, dtype=dtype)
	return im

def make_kernel(n_out, n_in, kernel):
    w, h = kernel.shape
    W = np.zeros((n_out, n_in, w, h))
    for i in range(n_out):
        if n_out > 1:
            W[i, i] = kernel
        else:
            for j in range(n_in):
                W[i, j] = kernel
    W = np.asarray(W, dtype=np.float32)
    return W

def make_grid(tensor, nrow=8, padding=2):
    """
    Given a 4D mini-batch Tensor of shape (B x C x H x W), makes a grid of images
    """
    # make the mini-batch of images into a grid
    nmaps = tensor.shape[0]
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(nmaps / xmaps))
    height, width = int(tensor.shape[2] + padding), int(tensor.shape[3] + padding)
    grid = np.ones((3, height*ymaps, width*xmaps))
    k = 0
    sy = 1 + padding // 2
    for y in range(ymaps): 
        sx = 1 + padding // 2
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid[:, sy:sy+height-padding, sx:sx+width-padding] = tensor[k]
            sx += width
            k = k + 1
        sy += height
    return grid