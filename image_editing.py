import os
import argparse

import numpy as np

from model import RealismCNN

import chainer
import chainer.functions as F
from chainer import cuda, Variable, serializers

from skimage.io import imread, imsave

from scipy.optimize import minimize

from utils import im_preprocess_vgg

def composite_img(x, obj, bg, mask):
    bias = x[:3, np.newaxis, np.newaxis]
    gain = x[3:, np.newaxis, np.newaxis]
    obj_colored = obj * bias + 255 * gain

    return np.maximum(0, np.minimum(255, obj_colored * mask + bg*(1-mask))), obj_colored

def color_adjust(x, obj, bg, mask, model, args):
    # Input for VGG
    composite, obj_colored = composite_img(x, obj, bg, mask)
    composite = np.asarray(composite - np.array([[[103.939]], [[116.779]], [[123.68]]]), dtype=np.float32)
    # Forward
    model_x = Variable(chainer.dataset.concat_examples([composite], args.gpu))
    y = model(model_x, dropout=False)
    # Realism loss
    b, _, w, h = y.shape
    xp = cuda.get_array_module(model_x.data)
    loss = F.softmax_cross_entropy(y, Variable(xp.ones((b, w, h), dtype=xp.int32)))
    # Backward
    loss.backward()
    # Transfer loss & diff from GPU to CPU
    loss = cuda.to_cpu(loss.data)
    diff = np.squeeze(cuda.to_cpu(model_x.grad))
    # Diff --- Realism loss
    dx = np.zeros_like(x)
    diff = diff * mask
    dx[:3] = np.sum(diff * obj, axis=(1, 2))
    dx[3:] = np.sum(diff * 255, axis=(1, 2))

    # Reg term
    # Part 1
    # Weight
    n = np.sum(mask)
    w1 = args.weight / (255**2 * n)
    # Loss
    color_diff = (obj_colored - obj) * mask
    loss += 0.5 * w1 * np.sum(color_diff ** 2)
    # Diff
    dx[:3] += w1 * np.sum(obj * color_diff, axis=(1, 2))
    dx[3:] += w1 * np.sum(255 * color_diff, axis=(1, 2))

    # Part 2
    # Weight
    w2 = w1 * min(20, np.mean(obj ** 2) / np.mean([np.mean((obj[0] - obj[1]) ** 2), np.mean((obj[1] - obj[2]) ** 2), np.mean((obj[0] - obj[2]) ** 2)]))
    # Loss
    loss += 0.5 * w2 * (np.sum((color_diff[0] - color_diff[1])**2) + np.sum((color_diff[1] - color_diff[2])**2) + np.sum((color_diff[0] - color_diff[2])**2))
    # Diff
    dx_cross_0 = 2*color_diff[0] - color_diff[1] - color_diff[2]
    dx_cross_1 = 2*color_diff[1] - color_diff[0] - color_diff[2]
    dx_cross_2 = 2*color_diff[2] - color_diff[0] - color_diff[1]
    dx[0] += w2 * np.sum(dx_cross_0*obj[0])
    dx[1] += w2 * np.sum(dx_cross_1*obj[1])
    dx[2] += w2 * np.sum(dx_cross_2*obj[2])
    dx[3] += w2 * 255 * np.sum(dx_cross_0)
    dx[4] += w2 * 255 * np.sum(dx_cross_1)
    dx[5] += w2 * 255 * np.sum(dx_cross_2)

    return loss, dx

def main():
    parser = argparse.ArgumentParser(description='Image editing using RealismCNN')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_path', default='model/realismCNN_all_iter3.npz', help='Path for pretrained model')
    parser.add_argument('--data_root', default='/data1/wuhuikai/benchmark/Realistic/color_adjustment', help='Root folder for color adjustment dataset')
    parser.add_argument('--img_folder', default='pngimages', help='Folder for stroing images')
    parser.add_argument('--list_name', default='list.txt', help='Name for file storing image list')
    parser.add_argument('--load_size', type=int, default=224, help='Scale image to load_size')
    parser.add_argument('--result_folder', default='result', help='Name for folder storing results')
    parser.add_argument('--result_name', default='loss.txt', help='Name for file saving loss change')
    args = parser.parse_args()

    args.weight = 50                                               # regulaziation weight
    args.seeds = np.arange(0.6, 1.6, 0.2)                          # multiple initiailization
    args.bounds = [(0.4, 2.0), (0.4, 2.0), (0.4, 2.0), (-0.5, 0.5), (-0.5, 0.5), (-0.5, 0.5)]
                                                                   # bounds for search range
    
    # Init CNN model
    model = RealismCNN()
    print('Load pretrained model from {} ...\n'.format(args.model_path))
    serializers.load_npz(args.model_path, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()                           # Copy the model to the GPU
    
    # Init image list
    im_root = os.path.join(args.data_root, args.img_folder)
    print('Load images from {} according to list {} ...'.format(im_root, args.list_name))
    with open(os.path.join(args.data_root, args.list_name)) as f:
        im_list = f.read().strip().split('\n')
    total = len(im_list)
    print('{} images loaded done!\n'.format(total))

    # Init result list
    if not os.path.isdir(args.result_folder):
        os.makedirs(args.result_folder)
    print('Result will save to {} ...\n'.format(args.result_folder))

    loss_change = []
    for idx, im_name in enumerate(im_list):
        print('Processing {}/{} ...'.format(idx + 1, total))
        obj  = im_preprocess_vgg(imread(os.path.join(im_root, '{}_obj.png'.format(im_name))),      args.load_size, sub_mean=False, dtype=np.float32)
        bg   = im_preprocess_vgg(imread(os.path.join(im_root, '{}_bg.png'.format(im_name))),       args.load_size, sub_mean=False, dtype=np.float32)
        mask = im_preprocess_vgg(imread(os.path.join(im_root, '{}_softmask.png'.format(im_name))), args.load_size, sub_mean=False, dtype=np.float32, preserve_range=False)

        xs = []
        fvals = []
        for n in range(args.seeds.size):
            x0 = np.zeros(6)
            x0[:3] = args.seeds[n]
            print('\tOptimize start with seed {} ...'.format(args.seeds[n]))
            res = minimize(color_adjust, x0, args=(obj, bg, mask, model, args), method='L-BFGS-B', jac=True, bounds=args.bounds, tol=1e-3)
            xs.append(res.x)
            fvals.append(res.fun)
            print('\tOptimize done, loss = {} \n'.format(fvals[-1]))

        # Cut and paste loss
        x0    = np.array([1, 1, 1, 0, 0, 0], dtype=np.float32)
        f0, _ = color_adjust(x0, obj, bg, mask, model, args)

        # Best x
        best_idx = np.argmin(fvals)
        print('\tBest seed = {}, loss = {}, cut_and_paste loss = {}, x = {}\n\n'.format(args.seeds[best_idx], fvals[best_idx], f0, xs[best_idx]))
        loss_change.append((im_name, f0, fvals[best_idx]))

        edited, _       = composite_img(xs[best_idx], obj, bg, mask)
        cut_and_pase, _ = composite_img(x0,           obj, bg, mask)
        result = np.concatenate((cut_and_pase, edited), axis=2)
        result = np.asarray(np.transpose(result[::-1,:,:], (1, 2, 0)), dtype=np.uint8)
        imsave(os.path.join(args.result_folder, '{}.png'.format(im_name)), result)

    with open(os.path.join(args.result_folder, args.result_name), 'w') as f:
        for name, f0, fb in loss_change:
            f.write('{} {} {}\n'.format(name, f0, fb))

if __name__ == '__main__':
    main()