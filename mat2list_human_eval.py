import os
import argparse

from scipy.io import loadmat

def main():
    parser = argparse.ArgumentParser(description='Turn image list in mat type into plaint text file.')
    parser.add_argument('--data_root', default='/data1/wuhuikai/benchmark/Realistic/human_evaluation', help='Root folder for test dataset')
    parser.add_argument('--sub_dataset', default='lalonde_and_efros_dataset', help='Folder name for sub_dataset, including images subfolder, as well as image list')
    parser.add_argument('--mat_path', default='human_labels.mat', help='Path for mat file')
    parser.add_argument('--list_name', default='list.txt', help='Name for file storing image list')
    parser.add_argument('--label_name', default='label.txt', help='Name for file stroring groundtruth')
    args = parser.parse_args()

    data_root = os.path.join(args.data_root, args.sub_dataset)
    print('Load mat {} from {} ...'.format(args.mat_path, data_root))
    mat = loadmat(os.path.join(data_root, args.mat_path))
    
    print('Save image list into {} ...'.format(args.list_name))
    im_list = mat['imgList']
    with open(os.path.join(data_root, args.list_name), 'w') as f:
        for l in im_list:
            f.write('{}\n'.format(l[0][0]))

    print('Save labels into {} ...'.format(args.label_name))
    labels = mat['labels']
    with open(os.path.join(data_root, args.label_name), 'w') as f:
        for label in labels:
            f.write('{}\n'.format(label[0]))

if __name__ == '__main__':
    main()