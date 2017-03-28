import os
import math

import numpy as np

from skimage.io import imsave
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

def save_result(x, args):
    x = np.reshape(x, args.shape)
    edited = x*args.mask + args.bg_vgg*args.inverse_mask
    result = np.concatenate((args.copy_paste_vgg, x, edited), axis=2)
    result = im_deprocess_vgg(result, orig_size=(args.orig_size[0], args.orig_size[1]*3), dtype=np.uint8)

    im_name = '{}_{}_{}'.format(args.im_name, args.iter, args.prefix_name)
    imsave(os.path.join(args.result_folder, '{}.png'.format(im_name)), result)

def make_kernel(n_out, n_in, kernel):
    w, h = kernel.shape
    W = np.zeros((n_out, n_in, w, h))
    for i in range(n_in):
        W[i, i] = kernel
    W = np.asarray(W, dtype=np.float32)
    return W