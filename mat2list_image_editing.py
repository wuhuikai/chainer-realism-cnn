import os
import argparse

from scipy.io import loadmat

def main():
    parser = argparse.ArgumentParser(description='Turn image list in mat type into plaint text file.')
    parser.add_argument('--data_root', default='/data1/wuhuikai/benchmark/Realistic/color_adjustment', help='Root folder for image editing dataset')
    parser.add_argument('--mat_path', default='imageList.mat', help='Path for mat file')
    parser.add_argument('--list_name', default='list.txt', help='Name for file storing image list')
    args = parser.parse_args()

    print('Load mat {} from {} ...'.format(args.mat_path, args.data_root))
    mat = loadmat(os.path.join(args.data_root, args.mat_path))
    
    print('Save image list into {} ...'.format(args.list_name))
    im_list = mat['imgList']
    with open(os.path.join(args.data_root, args.list_name), 'w') as f:
        for l in im_list:
            f.write('{}\n'.format(l[0][0]))

if __name__ == '__main__':
    main()