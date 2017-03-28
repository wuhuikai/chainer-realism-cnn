import collections

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import cuda, function, link, initializers

def extract(inputs, model, layers, train=False):
    features = {}
    layers = set(layers)
    variables = dict(inputs)

    model.train = train
    for func_name, bottom, top in model.layers:
        if len(layers) == 0:
            break
        if func_name not in model.forwards or any(blob not in variables for blob in bottom):
            continue
        func = model.forwards[func_name]
        input_vars = tuple(variables[blob] for blob in bottom)
        output_vars = func(*input_vars)
        if not isinstance(output_vars, collections.Iterable):
            output_vars = output_vars,
        for var, name in zip(output_vars, top):
            variables[name] = var
        if func_name in layers:
            features[func_name] = output_vars
            layers.remove(func_name)
    return features

def total_variation(x):
    _, _, h, w = x.data.shape

    return 0.5*F.sum((x[:, :, :h-1, :w-1] - x[:, :, 1:, :w-1])**2 + (x[:, :, :h-1, :w-1] - x[:, :, :h-1, 1:])**2)

class RealismCNN(chainer.Chain):
    def __init__(self, w_init=None):
        super(RealismCNN, self).__init__(
            conv1_1=L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=w_init),
            conv1_2=L.Convolution2D(None, 64, ksize=3, stride=1, pad=1, initialW=w_init),

            conv2_1=L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, initialW=w_init),
            conv2_2=L.Convolution2D(None, 128, ksize=3, stride=1, pad=1, initialW=w_init),

            conv3_1=L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, initialW=w_init),
            conv3_2=L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, initialW=w_init),
            conv3_3=L.Convolution2D(None, 256, ksize=3, stride=1, pad=1, initialW=w_init),

            conv4_1=L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, initialW=w_init),
            conv4_2=L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, initialW=w_init),
            conv4_3=L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, initialW=w_init),

            conv5_1=L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, initialW=w_init),
            conv5_2=L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, initialW=w_init),
            conv5_3=L.Convolution2D(None, 512, ksize=3, stride=1, pad=1, initialW=w_init),

            fc6=L.Convolution2D(None, 4096, ksize=7, stride=1, pad=0, initialW=w_init),
            fc7=L.Convolution2D(None, 4096, ksize=1, stride=1, pad=0, initialW=w_init),
            fc8=L.Convolution2D(None, 2, ksize=1, stride=1, pad=0, initialW=w_init)
        )

    def __call__(self, x, dropout=True):
        h = F.relu(self.conv1_1(x))
        h = F.relu(self.conv1_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv2_1(h))
        h = F.relu(self.conv2_2(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv3_1(h))
        h = F.relu(self.conv3_2(h))
        h = F.relu(self.conv3_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv4_1(h))
        h = F.relu(self.conv4_2(h))
        h = F.relu(self.conv4_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.relu(self.conv5_1(h))
        h = F.relu(self.conv5_2(h))
        h = F.relu(self.conv5_3(h))
        h = F.max_pooling_2d(h, ksize=2, stride=2)

        h = F.dropout(F.relu(self.fc6(h)), train=dropout)
        h = F.dropout(F.relu(self.fc7(h)), train=dropout)
        h = self.fc8(h)

        return h