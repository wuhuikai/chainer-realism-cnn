import os
import pickle
import argparse

import numpy as np

from model import RealismCNN, total_variation, extract

import chainer
import chainer.functions as F
from chainer import cuda, Variable, serializers

from skimage.io import imread
from skimage.morphology import erosion

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
    poisson_loss = F.mean_squared_error((args.content_laplace + args.border_sum)*args.mask_var, F.convolution_2d(x_vgg_var*args.mask_var, W=args.W_laplace, pad=1)*args.mask_var)
    poisson_loss *= np.prod(x_vgg_var.shape)

    # tv loss
    tv_loss = total_variation(x_vgg_var)

    # Concent loss
    content_loss = 0
    x_features = extract({'data': x_vgg_var}, args.vgg, args.content_layers)
    x_features = {key:value[0] for key, value in x_features.items()}
    for layer in args.content_layers:
        content_loss += F.mean_squared_error(args.content_features[layer], x_features[layer])

    # Realism loss
    y = args.realism_cnn(x_vgg_var, dropout=False)
    b, _, w, h = y.shape
    xp = cuda.get_array_module(x_vgg_var.data)
    realism_loss = F.sum(y[:, 0, :, :])

    loss = args.poisson_weight * poisson_loss + args.realism_weight * realism_loss + args.tv_weight * tv_loss + args.content_weight * content_loss
    
    # Backward
    loss.backward()
    # Transfer loss & diff from GPU to CPU
    loss = cuda.to_cpu(loss.data)
    dx = np.squeeze(cuda.to_cpu(x_vgg_var.grad))
    
    return loss, np.asarray(dx.flatten(), dtype=np.float64)

def main():
    parser = argparse.ArgumentParser(description='Poisson image editing using RealismCNN')
    parser.add_argument('--poisson_weight', type=float, default=1, help='Weight for poisson loss')
    parser.add_argument('--realism_weight', type=float, default=1e4, help='Weight for realism loss')
    parser.add_argument('--content_weight', type=float, default=1, help='Weight for content loss')
    parser.add_argument('--tv_weight', type=float, default=1e-1, help='Weight for tv loss')
    parser.add_argument('--n_iteration', type=int, default=1000, help='# of iterations')
    parser.add_argument('--save_intervel', type=int, default=100, help='save result every # of iterations')
    parser.add_argument('--rand_init', type=lambda x:x == 'True', default=True, help='Random init input if True')
    parser.add_argument('--content_layers', type=str2list, default='conv4_1', help='Layers for content_loss, sperated by ;')

    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--realism_model_path', default='model/realismCNN_all_iter3.npz', help='Path for pretrained Realism model')
    parser.add_argument('--content_model_path', default='model/VGG_ILSVRC_19_layers.pkl', help='Path for pretrained VGG model')
    parser.add_argument('--data_root', default='/data1/wuhuikai/benchmark/Realistic/color_adjustment', help='Root folder for color adjustment dataset')
    parser.add_argument('--img_folder', default='pngimages', help='Folder for stroing images')
    parser.add_argument('--list_name', default='list.txt', help='Name for file storing image list')
    parser.add_argument('--load_size', type=int, default=224, help='Scale image to load_size')
    parser.add_argument('--result_folder', default='image_editing_result', help='Name for folder storing results')
    parser.add_argument('--result_name', default='loss.txt', help='Name for file saving loss change')
    args = parser.parse_args()
    
    args.content_layers = set(args.content_layers)
    
    print('Input arguments:')
    for key, value in vars(args).items():
        print('\t{}: {}'.format(key, value))
    print('')

    args.prefix_name = '_'.join(sorted(['{}({})'.format(key, value) for key, value in vars(args).items()
        if key not in set(['realism_model_path', 'content_model_path', 'data_root', 'img_folder', 'list_name', 'result_folder', 'result_name'])]))

    # Init CNN model
    realism_cnn = RealismCNN()
    print('Load pretrained Realism model from {} ...'.format(args.realism_model_path))
    serializers.load_npz(args.realism_model_path, realism_cnn)
    print('Load pretrained VGG model from {} ...\n'.format(args.content_model_path))
    with open(args.content_model_path, 'rb') as f:
        vgg = pickle.load(f)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        realism_cnn.to_gpu()                     # Copy the model to the GPU
        vgg.to_gpu()
    
    # Init image list
    im_root = os.path.join(args.data_root, args.img_folder)
    print('Load images from {} according to list {} ...'.format(im_root, args.list_name))
    with open(os.path.join(args.data_root, args.list_name)) as f:
        im_list = f.read().strip().split('\n')
    total = len(im_list)
    print('{} images loaded done!\n'.format(total))

    # Init result folder
    if not os.path.isdir(args.result_folder):
        os.makedirs(args.result_folder)
    print('Result will save to {} ...\n'.format(args.result_folder))

    # Init Constant Variable
    W_laplace = Variable(make_kernel(3, 3, np.asarray([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)), volatile='auto')
    W_laplace.to_gpu()
    args.W_laplace = W_laplace
    W_sum = Variable(make_kernel(3, 3, np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)), volatile='auto')
    W_sum.to_gpu()

    loss_change = []
    for idx, im_name in enumerate(im_list):
        print('Processing {}/{}, name = {} ...'.format(idx + 1, total, im_name))
        obj_vgg  = im_preprocess_vgg(imread(os.path.join(im_root, '{}_obj.png'.format(im_name))),  args.load_size, dtype=np.float32)
        bg_vgg   = im_preprocess_vgg(imread(os.path.join(im_root, '{}_bg.png'.format(im_name))),   args.load_size, dtype=np.float32)
        expand_mask = im_preprocess_vgg(imread(os.path.join(im_root, '{}_softmask.png'.format(im_name))), args.load_size, sub_mean=False, dtype=np.uint8, preserve_range=False)
        
        args.orig_size = (args.load_size, args.load_size)
        args.shape = bg_vgg.shape
        ## mask
        mask = erosion(np.squeeze(expand_mask), np.ones((3, 3), dtype=np.uint8))
        mask = np.asarray(mask[np.newaxis, :, :], dtype=np.float32)
        expand_mask = np.asarray(expand_mask, dtype=np.float32)
        inverse_mask = 1-mask
        ## vars
        obj_var = Variable(chainer.dataset.concat_examples([obj_vgg], args.gpu), volatile='on')
        mask_var = F.broadcast_to(Variable(chainer.dataset.concat_examples([mask], args.gpu)), obj_var.shape)
        ## Laplace
        content_laplace = F.convolution_2d(obj_var, W=W_laplace, pad=1)
        content_laplace.volatile = 'off'
        # prefilled
        border = bg_vgg*expand_mask*inverse_mask
        border_var = Variable(chainer.dataset.concat_examples([border], args.gpu), volatile='on')
        border_sum = F.convolution_2d(border_var, W=W_sum, pad=1)
        border_sum.volatile = 'off'

        print('\tExtracting content image features ...')
        copy_paste_vgg = obj_vgg*mask + bg_vgg*inverse_mask
        copy_paste_var = Variable(chainer.dataset.concat_examples([copy_paste_vgg], args.gpu), volatile='on')
        content_features = extract({'data': copy_paste_var}, vgg, args.content_layers)
        content_features = {key:value[0] for key, value in content_features.items()}
        for _, value in content_features.items():
            value.volatile = 'off'

        ## args
        args.vgg = vgg
        args.realism_cnn = realism_cnn
        args.border_sum = border_sum
        args.content_laplace = content_laplace
        args.content_features = content_features
        args.mask = mask
        args.mask_var = mask_var
        args.inverse_mask = inverse_mask
        args.bg_vgg = bg_vgg
        args.copy_paste_vgg = copy_paste_vgg
        args.im_name = im_name

        args.iter = 0
        x_init = np.asarray(np.random.randn(*args.shape) * 0.001, dtype=np.float32) if args.rand_init else np.copy(copy_paste_vgg)
        print('\tOptimize start ...')
        res = minimize(color_adjust, x_init, args=(args), method='L-BFGS-B', jac=True, options={'maxiter': args.n_iteration, 'disp':False})
        # Cut and paste loss
        args.iter = -1
        f0, _ = color_adjust(copy_paste_vgg, args)
        print('\tOptimize done, loss = {} from {}\n'.format(res.fun, f0))
        loss_change.append((im_name, f0, res.fun))

        args.iter = ''
        save_result(res.x, args)

    with open(os.path.join(args.result_folder, args.result_name), 'w') as f:
        for name, f0, fb in loss_change:
            f.write('{} {} {}\n'.format(name, f0, fb))

if __name__ == '__main__':
    main()