import os
import glob
import argparse

import numpy as np

from model import total_variation

import chainer
import chainer.functions as F
from chainer import cuda, Variable

from skimage.io import imread

from scipy.optimize import minimize

from utils import im_preprocess_vgg, im_deprocess_vgg, make_kernel, save_result

str2list = lambda x: x.split(';')

def color_adjust(x, args):
    if args.iter % args.save_intervel == 0:
        save_result(x, args)
    args.iter += 1
    # Input for VGG
    x_vgg = np.asarray(np.reshape(x, args.shape), dtype=np.float32)
    x_vgg_var = Variable(chainer.dataset.concat_examples([x_vgg], args.gpu))

    # Poisson loss
    poisson_loss = F.sum(F.square(args.content_laplace - F.convolution_2d(x_vgg_var, W=args.W_laplace, pad=1))*args.inverse_border_mask_var)
    
    # tv loss
    tv_loss = total_variation(x_vgg_var)

    # Content loss
    content_loss = F.sum(F.square((args.bg_var - x_vgg_var)*args.inverse_mask_var))

    loss = args.poisson_weight * poisson_loss + args.tv_weight * tv_loss + args.content_weight * content_loss
     
    # Backward
    loss.backward()
    # Transfer loss & diff from GPU to CPU
    loss = cuda.to_cpu(loss.data)
    dx = np.squeeze(cuda.to_cpu(x_vgg_var.grad))
    
    return loss, np.asarray(dx.flatten(), dtype=np.float64)

def main():
    parser = argparse.ArgumentParser(description='Modified Poisson image editing')
    parser.add_argument('--poisson_weight', type=float, default=1, help='Weight for poisson loss')
    parser.add_argument('--content_weight', type=float, default=5e-4, help='Weight for content loss')
    parser.add_argument('--tv_weight', type=float, default=1e-3, help='Weight for tv loss')
    parser.add_argument('--n_iteration', type=int, default=3500, help='# of iterations')
    parser.add_argument('--save_intervel', type=int, default=100, help='save result every # of iterations')
    parser.add_argument('--rand_init', type=lambda x:x == 'True', default=True, help='Random init input if True')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--data_root', default='/data1/wuhuikai/benchmark/TransientAttributes/imageCropped', help='Root folder for cropped transient attributes dataset')
    parser.add_argument('--load_size', type=int, default=224, help='Scale image to load_size')
    parser.add_argument('--total_instance', type=int, default=175, help='# of instance to run test')
    parser.add_argument('--seed', type=int, default=5, help='Random seed')
    parser.add_argument('--min_ratio', type=float, default=0.2, help='Min ratio for size of random rect')
    parser.add_argument('--max_ratio', type=float, default=0.6, help='Max ratio for size of random rect')
    parser.add_argument('--result_folder', default='transient_attributes_result/modified_result', help='Name for folder storing results')
    parser.add_argument('--result_name', default='loss.txt', help='Name for file saving loss change')
    args = parser.parse_args()

    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    args.prefix_name = '_'.join(sorted(['{}({})'.format(key, value) for key, value in vars(args).items()
        if key in set(['poisson_weight', 'realism_weight', 'content_weight', 'tv_weight', 'n_iteration', 'rand_init'])]))

    # Init image list
    print('Load images from {} ...'.format(args.data_root))
    folders = [folder for folder in os.listdir(args.data_root) if os.path.isdir(os.path.join(args.data_root, folder))]
    imgs_each_folder = {folder:glob.glob(os.path.join(args.data_root, folder, '*')) for folder in folders}
    print('\t {} images in {} folders in total ...\n'.format(np.sum([len(v) for k, v in imgs_each_folder.items()]), len(folders)))

    # Init result folder
    if not os.path.isdir(args.result_folder):
        os.makedirs(args.result_folder)
    print('Result will save to {} ...\n'.format(args.result_folder))

    # Init Constant Variable
    args.W_laplace = Variable(make_kernel(3, 3, np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)), volatile='auto')
    args.W_laplace.to_gpu()

    loss_change = []
    np.random.seed(args.seed)
    for i in range(args.total_instance):
        folder = np.random.choice(folders)
        print('Processing {}/{}, select folder {} ...'.format(i+1, args.total_instance, folder))
        obj_path, bg_path = np.random.choice(imgs_each_folder[folder], 2, replace=False)
        print('\tObj: {}, Bg: {} ...'.format(os.path.basename(obj_path), os.path.basename(bg_path)))

        obj_img = imread(obj_path)
        args.orig_size = obj_img.shape[:2]
        obj_vgg  = im_preprocess_vgg(obj_img,         args.load_size, dtype=np.float32)
        bg_vgg   = im_preprocess_vgg(imread(bg_path), args.load_size, dtype=np.float32)
        args.shape = obj_vgg.shape

        # random rect
        w, h = np.asarray(np.random.uniform(args.min_ratio, args.max_ratio, 2)*args.load_size, dtype=np.uint32)
        sx, sy = np.random.randint(0, args.load_size-w), np.random.randint(0, args.load_size-h)

        expand_mask = np.zeros((1, args.load_size, args.load_size), dtype=np.float32)
        expand_mask[:, sx:sx+w, sy:sy+h] = 1
        mask = np.zeros_like(expand_mask)
        mask[:, sx+1:sx+w-1, sy+1:sy+h-1] = 1
        inverse_mask = 1 - mask
        ## vars
        obj_var = Variable(chainer.dataset.concat_examples([obj_vgg], args.gpu), volatile='on')
        bg_var = Variable(chainer.dataset.concat_examples([bg_vgg], args.gpu), volatile='auto')
        mask_var = F.broadcast_to(Variable(chainer.dataset.concat_examples([mask], args.gpu), volatile='auto'), obj_var.shape)
        inverse_mask_var = F.broadcast_to(Variable(chainer.dataset.concat_examples([inverse_mask], args.gpu), volatile='auto'), obj_var.shape)
        inverse_border_mask_var = F.broadcast_to(Variable(chainer.dataset.concat_examples([1-expand_mask+mask], args.gpu), volatile='auto'), obj_var.shape)
        ## Laplace
        content_laplace = F.convolution_2d(obj_var, W=args.W_laplace, pad=1)*mask_var + F.convolution_2d(bg_var, W=args.W_laplace, pad=1)*inverse_mask_var
        content_laplace.volatile = 'off'

        copy_paste_vgg = obj_vgg*mask + bg_vgg*inverse_mask
        ## args
        args.content_laplace = content_laplace
        args.mask = mask
        args.inverse_mask = inverse_mask
        args.inverse_mask_var = inverse_mask_var
        args.inverse_border_mask_var = inverse_border_mask_var
        args.bg_vgg = bg_vgg
        args.bg_var = bg_var
        args.copy_paste_vgg = copy_paste_vgg
        args.im_name = 'folder_{}_obj_{}_bg_{}'.format(folder, os.path.splitext(os.path.basename(obj_path))[0], os.path.splitext(os.path.basename(bg_path))[0])

        args.iter = 0
        x_init = np.asarray(np.random.randn(*args.shape) * 0.001, dtype=np.float32) if args.rand_init else np.copy(copy_paste_vgg)
        print('\tOptimize start ...')
        res = minimize(color_adjust, x_init, args=(args), method='L-BFGS-B', jac=True, options={'maxiter': args.n_iteration, 'disp':True})
        # Cut and paste loss
        args.iter = -1
        f0, _ = color_adjust(copy_paste_vgg, args)
        print('\tOptimize done, loss = {} from {}\n'.format(res.fun, f0))
        loss_change.append((args.im_name, f0, res.fun))

        args.iter = ''
        save_result(res.x, args)

    with open(os.path.join(args.result_folder, args.result_name), 'w') as f:
        for name, f0, fb in loss_change:
            f.write('{} {} {}\n'.format(name, f0, fb))

if __name__ == '__main__':
    main()