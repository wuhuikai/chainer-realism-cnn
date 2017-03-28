from __future__ import print_function
import os
import argparse

import chainer
from chainer import Variable, serializers

from model import RealismCNN

from utils import im_preprocess_vgg

import numpy as np

from sklearn.metrics import roc_auc_score

def main():
    parser = argparse.ArgumentParser(description='Predict a list of images wheather realistic or not')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID (negative value indicates CPU)')
    parser.add_argument('--model_path', default='model/realismCNN_all_iter3.npz', help='Path for pretrained model')
    parser.add_argument('--data_root', default='/data1/wuhuikai/benchmark/Realistic/human_evaluation', help='Root folder for test dataset')
    parser.add_argument('--sub_dataset', default='lalonde_and_efros_dataset', help='Folder name for sub_dataset, including images subfolder, as well as image list')
    parser.add_argument('--img_folder', default='images', help='Folder stroing images')
    parser.add_argument('--list_name', default='list.txt', help='Name for file storing image list')
    parser.add_argument('--label_name', default='label.txt', help='Name for file stroring groundtruth')
    parser.add_argument('--batch_size', type=int, default=10, help='Batchsize of 1 iteration')
    parser.add_argument('--load_size', type=int, default=256, help='Scale image to load_size')
    parser.add_argument('--result_name', default='result.txt', help='Name for file storing result')
    args = parser.parse_args()

    data_root = os.path.join(args.data_root, args.sub_dataset)
    print('Predict realism for images in {} ...'.format(data_root))

    model = RealismCNN()
    print('Load pretrained model from {} ...'.format(args.model_path))
    serializers.load_npz(args.model_path, model)
    if args.gpu >= 0:
        chainer.cuda.get_device(args.gpu).use()  # Make a specified GPU current
        model.to_gpu()                           # Copy the model to the GPU


    print('Load images from {} ...'.format(args.list_name))
    dataset = chainer.datasets.ImageDataset(paths=os.path.join(data_root, args.list_name), root=os.path.join(data_root, args.img_folder))
    print('{} images in total loaded'.format(len(dataset)))
    data_iterator = chainer.iterators.SerialIterator(dataset, args.batch_size, repeat=False, shuffle=False)

    scores = np.zeros((0, 2))
    for idx, batch in enumerate(data_iterator):
        print('Processing batch {}->{}/{} ...'.format(idx*args.batch_size+1, min(len(dataset), (idx+1)*args.batch_size), len(dataset)))
        batch = [im_preprocess_vgg(np.transpose(im, [1, 2, 0]), args.load_size) for im in batch]
        batch = Variable(chainer.dataset.concat_examples(batch, args.gpu), volatile='on')
        result = chainer.cuda.to_cpu(model(batch, dropout=False).data)
        scores = np.vstack((scores, np.mean(result, axis=(2, 3))))

    print('Processing DONE !')
    print('Saving result to {} ...'.format(args.result_name))
    with open(os.path.join(data_root, args.result_name), 'w') as f:
        for score in scores:
            f.write('{}\t{}\t{}\n'.format(score[0], score[1], np.argmax(score)))

    if not args.label_name:
        return
    print('Load gt from {} ...'.format(args.label_name))
    with open(os.path.join(data_root, args.label_name)) as f:
        gts = np.asarray(f.readlines(), dtype=np.uint8)
    gts[gts > 0.5] = 1
    auc = roc_auc_score(gts, scores[:, 1])
    print('AUC score: {}'.format(auc))

if __name__ == '__main__':
    main()