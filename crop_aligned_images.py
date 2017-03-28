import os
import glob
import argparse

import numpy as np

from skimage.io import imread, imsave
from skimage.draw import polygon_perimeter
from skimage.morphology import binary_closing

def max_box(mask, bbox=False):
    sx, sy = 0, 0
    ex, ey = mask.shape
    while True:
        if np.any(mask[sx,:]):
            break
        sx += 1
    while True:
        if np.any(mask[:,sy]):
            break
        sy += 1
    while True:
        if np.any(mask[ex-1,:]):
            break
        ex -= 1
    while True:
        if np.any(mask[:,ey-1]):
            break
        ey -= 1

    if bbox:
        return sx, sy, ex, ey

    while True:
        if np.all(mask[sx:ex,sy:ey]):
            break
        sx += 1; sy +=1; ex -= 1; ey -= 1

    while True:
        if sx > 0 and np.all(mask[sx:ex,sy:ey]):
            sx -= 1
        else:
            break
    while True:
        if sy > 0 and np.all(mask[sx:ex,sy:ey]):
            sy -= 1
        else:
            break
    while True:
        if ex < mask.shape[0] and np.all(mask[sx:ex,sy:ey]):
            ex += 1
        else:
            break
    while True:
        if ey < mask.shape[1] and np.all(mask[sx:ex,sy:ey]):
            ey += 1
        else:
            break
    return sx, sy, ex, ey

def main():
    parser = argparse.ArgumentParser(description='Crop aligned images')
    parser.add_argument('--data_root', default='/data1/wuhuikai/benchmark/TransientAttributes/imageAlignedLD', help='Root folder for aligned images')
    parser.add_argument('--save_root', default='/data1/wuhuikai/benchmark/TransientAttributes/imageCropped', help='Name for folder storing results')
    args = parser.parse_args()

    if not os.path.isdir(args.save_root):
            os.makedirs(args.save_root)

    bbox_dict = {}
    folders = os.listdir(args.data_root)
    for idx, folder in enumerate(folders):
        print('Processing {}/{}, folder name: {}'.format(idx+1, len(folders), folder))

        img_paths = glob.glob(os.path.join(args.data_root, folder, '*'))
        mask = None
        for idx, img_path in enumerate(img_paths):
            img_mask = np.asarray(np.sum(imread(img_path), axis=2), dtype=np.bool)
            mask = img_mask if idx==0 else img_mask*mask
            
        mask = binary_closing(mask, selem=np.ones((10, 10)))
        sx, sy, ex, ey = max_box(mask, bbox=True if '+' in folder else False)
        bbox_dict[folder] = (sx, sy, ex, ey)
        
        rr, cc = polygon_perimeter([sx, ex, ex, sx], [sy, sy, ey, ey], shape=mask.shape, clip=True)
        mask = np.asarray(mask, dtype=np.uint8)
        mask[mask == 1] = 128
        mask[rr, cc] = 255
        imsave(os.path.join(args.save_root, '{}.png'.format(folder)), mask)

        save_folder = os.path.join(args.save_root, folder)
        if not os.path.isdir(save_folder):
            os.makedirs(save_folder)
        
        avg_img = None
        for idx, img_path in enumerate(img_paths):
            img = imread(img_path)
            im_name = os.path.basename(img_path)
            imsave(os.path.join(save_folder, im_name), img[sx:ex, sy:ey])

            img = np.asarray(img, dtype=np.float32)
            avg_img = img if idx==0 else img+avg_img
        avg_img = np.asarray(avg_img/len(img_paths), dtype=np.uint8)
        avg_img[rr, cc, :] = 255
        imsave(os.path.join(args.save_root, '{}_avg.png'.format(folder)), avg_img)

    with open(os.path.join(args.save_root, 'bbox.txt'), 'w') as f:
        f.write('\n'.join(['{}:{}'.format(key, ','.join([str(x) for x in value])) for key, value in bbox_dict.items()]))

if __name__ == '__main__':
    main()