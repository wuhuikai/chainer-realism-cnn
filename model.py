import collections

import numpy as np

import chainer
import chainer.links as L
import chainer.functions as F

from chainer import cuda, function, link, initializers

def gram(x):
    b, c, h, w = x.shape
    feature = F.reshape(x, (b, c, h*w))
    return F.batch_matmul(feature, feature, transb=True) / (c*h*w)

def normlize_grad(func, inputs, normalize=True):
    return NormalizeGrad(func)(*inputs) if normalize else func(*inputs)

class NormalizeGrad(function.Function):
    def __init__(self, func):
        self._eps = 10e-8
        self._func = func

    def forward(self, inputs):
        return self._func.forward(inputs)

    def backward(self, inputs, grad_outputs):
        xp = cuda.get_array_module(*inputs)
        grads = self._func.backward(inputs, grad_outputs)

        return tuple(grad/(xp.linalg.norm(grad.flatten(), ord=1)+self._eps) for grad in grads)

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

class InstanceNormalization(link.Link):
    def __init__(self, nc, dtype=np.float32):
        super(InstanceNormalization, self).__init__()
        self.nc = nc
        self.dtype = dtype
        self.bn = None
        self.prev_batch = None

        self.add_param('gamma', nc, dtype=dtype)
        initializers.init_weight(self.gamma.data, np.random.uniform(size=nc))

        self.add_param('beta', nc, dtype=dtype)
        initial_beta = initializers.Zero()
        initializers.init_weight(self.beta.data, initial_beta)

    def __call__(self, x, test=True):
        n, c, h, w = x.shape
        assert(c == self.nc)
        if n != self.prev_batch:
            self.bn = L.BatchNormalization(n*c, dtype=self.dtype)
            self.bn.to_gpu(self._device_id)
            self.bn.gamma = F.tile(self.gamma, n)
            self.bn.beta = F.tile(self.beta, n)
            self.prev_batch = n

        x = F.reshape(x, (1, n*c, h, w))
        return F.reshape(self.bn(x), (n, c, h, w))

## NO InstanceNormalization & padding
class ResidualBlock(chainer.Chain):
    def __init__(self, nc, stride=1, ksize=3, instance_normalization=True,w_init=None):
        BN = InstanceNormalization if instance_normalization else L.BatchNormalization
        super(ResidualBlock, self).__init__(
            c1=L.Convolution2D(None, nc, ksize=ksize, stride=stride, pad=1, initialW=w_init),
            c2=L.Convolution2D(None, nc, ksize=ksize, stride=stride, pad=1, initialW=w_init),
            b1=BN(nc),
            b2=BN(nc)
        )

    def __call__(self, x, test):
        h = F.relu(self.b1(self.c1(x), test=test))
        h = self.b2(self.c2(h), test=test)
        
        return h + x

## NO InstanceNormalization & Deconv ksize
class ImageTransformer(chainer.Chain):
    def __init__(self, feature_map_nc, output_nc, tanh_constant, instance_normalization=True, w_init=None):
        self.tanh_constant = tanh_constant
        BN = InstanceNormalization if instance_normalization else L.BatchNormalization
        super(ImageTransformer, self).__init__(
            c1=L.Convolution2D(None, feature_map_nc, ksize=9, stride=1, pad=4, initialW=w_init),
            c2=L.Convolution2D(None, 2*feature_map_nc, ksize=3, stride=2, pad=1, initialW=w_init),
            c3=L.Convolution2D(None, 4*feature_map_nc, ksize=3,stride=2, pad=1, initialW=w_init),
            r1=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            r2=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            r3=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            r4=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            r5=ResidualBlock(4*feature_map_nc, instance_normalization=instance_normalization, w_init=w_init),
            d1=L.Deconvolution2D(None, 2*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            d2=L.Deconvolution2D(None, feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            d3=L.Convolution2D(None, output_nc, ksize=9, stride=1, pad=4, initialW=w_init),
            b1=BN(feature_map_nc),
            b2=BN(2*feature_map_nc),
            b3=BN(4*feature_map_nc),
            b4=BN(2*feature_map_nc),
            b5=BN(feature_map_nc)
        )

    def __call__(self, x, test=False):
        h = F.relu(self.b1(self.c1(x), test=test))
        h = F.relu(self.b2(self.c2(h), test=test))
        h = F.relu(self.b3(self.c3(h), test=test))
        h = self.r1(h, test=test)
        h = self.r2(h, test=test)
        h = self.r3(h, test=test)
        h = self.r4(h, test=test)
        h = self.r5(h, test=test)
        h = F.relu(self.b4(self.d1(h), test=test))
        h = F.relu(self.b5(self.d2(h), test=test))
        y = self.d3(h)

        return F.tanh(y) * self.tanh_constant

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

class Generator(chainer.Chain):
    def __init__(self, feature_map_nc, output_nc, w_init=None):
        super(Generator, self).__init__(
            c1=L.Convolution2D(None, feature_map_nc, ksize=9, stride=1, pad=4, initialW=w_init),
            c2=L.Convolution2D(None, 2*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            c3=L.Convolution2D(None, 4*feature_map_nc, ksize=4,stride=2, pad=1, initialW=w_init),
            r1=ResidualBlock(4*feature_map_nc, w_init=w_init),
            r2=ResidualBlock(4*feature_map_nc, w_init=w_init),
            r3=ResidualBlock(4*feature_map_nc, w_init=w_init),
            r4=ResidualBlock(4*feature_map_nc, w_init=w_init),
            r5=ResidualBlock(4*feature_map_nc, w_init=w_init),
            d1=L.Deconvolution2D(None, 2*feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            d2=L.Deconvolution2D(None, feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init),
            d3=L.Deconvolution2D(None, output_nc, ksize=9, stride=1, pad=4, initialW=w_init),
            b1=L.BatchNormalization(feature_map_nc),
            b2=L.BatchNormalization(2*feature_map_nc),
            b3=L.BatchNormalization(4*feature_map_nc),
            b4=L.BatchNormalization(2*feature_map_nc),
            b5=L.BatchNormalization(feature_map_nc)
        )

    def __call__(self, x, test=False):
        h = self.b1(F.elu(self.c1(x)), test=test)
        h = self.b2(F.elu(self.c2(h)), test=test)
        h = self.b3(F.elu(self.c3(h)), test=test)
        h = self.r1(h, test=test)
        h = self.r2(h, test=test)
        h = self.r3(h, test=test)
        h = self.r4(h, test=test)
        h = self.r5(h, test=test)
        h = self.b4(F.elu(self.d1(h)), test=test)
        h = self.b5(F.elu(self.d2(h)), test=test)
        y = self.d3(h)
        return F.tanh(y)

class ReLU(chainer.Chain):
    def __init__(self):
        super(ReLU, self).__init__()

    def __call__(self, x):
        return F.relu(x)

class Tanh(chainer.Chain):
    def __init__(self):
        super(Tanh, self).__init__()

    def __call__(self, x):
        return F.tanh(x)

class LeakyReLU(chainer.Chain):
    def __init__(self):
        super(LeakyReLU, self).__init__()

    def __call__(self, x):
        return F.leaky_relu(x)

def init_conv(array):
    xp = cuda.get_array_module(array)
    array[...] = xp.random.normal(loc=0.0, scale=0.02, size=array.shape)
def init_bn(array):
    xp = cuda.get_array_module(array)
    array[...] = xp.random.normal(loc=1.0, scale=0.02, size=array.shape)

class DCGAN_G(chainer.ChainList):
    def __init__(self, isize, nz, nc, ngf, conv_init=None, bn_init=None):
        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        layers = []
        # input is Z, going into a convolution
        layers.append(L.Deconvolution2D(None, cngf, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True))
        layers.append(L.BatchNormalization(cngf, initial_gamma=bn_init))
        layers.append(ReLU())
        csize, cndf = 4, cngf
        while csize < isize//2:
            layers.append(L.Deconvolution2D(None, cngf//2, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
            layers.append(L.BatchNormalization(cngf//2, initial_gamma=bn_init))
            layers.append(ReLU())
            cngf = cngf // 2
            csize = csize * 2
        layers.append(L.Deconvolution2D(None, nc, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
        layers.append(Tanh())

        super(DCGAN_G, self).__init__(*layers)

    def __call__(self, x, test=False):
        for i in range(len(self)):
            if isinstance(self[i], L.BatchNormalization):
                x = self[i](x, test=test)
            else:
                x = self[i](x)
        return x

class DCGAN_D(chainer.ChainList):
    def __init__(self, isize, nz, nc, ndf, conv_init=None, bn_init=None):
        layers = []
        layers.append(L.Convolution2D(None, ndf, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
        layers.append(LeakyReLU())
        csize, cndf = isize / 2, ndf  
        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            layers.append(L.Convolution2D(None, out_feat, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
            layers.append(L.BatchNormalization(out_feat, initial_gamma=bn_init))
            layers.append(LeakyReLU())

            cndf = cndf * 2
            csize = csize / 2
        # state size. K x 4 x 4
        layers.append(L.Convolution2D(None, 1, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True))
        
        super(DCGAN_D, self).__init__(*layers)

    def __call__(self, x, test=False):
        for i in range(len(self)):
            if isinstance(self[i], L.BatchNormalization):
                x = self[i](x, test=test)
            else:
                x = self[i](x)

        x = F.sum(x, axis=0) / x.shape[0]
        return F.squeeze(x)

class EncoderDecoder(chainer.Chain):
    def __init__(self, nef, ngf, nc, nBottleneck, image_size=64, conv_init=None, bn_init=None):
        super(EncoderDecoder, self).__init__(
            conv1 = L.Convolution2D(None, nef,   ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            conv3 = L.Convolution2D(None, nef*2, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            conv4 = L.Convolution2D(None, nef*4, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            conv5 = L.Convolution2D(None, nef*8, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            conv6 = L.Convolution2D(None, nBottleneck, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True),
            bn3   = L.BatchNormalization(nef*2, initial_gamma=bn_init),
            bn4   = L.BatchNormalization(nef*4, initial_gamma=bn_init),
            bn5   = L.BatchNormalization(nef*8, initial_gamma=bn_init),
            bn6   = L.BatchNormalization(nBottleneck, initial_gamma=bn_init),
            dconv6 = L.Deconvolution2D(None, ngf*8, ksize=4, stride=1, pad=0, initialW=conv_init, nobias=True),
            dconv5 = L.Deconvolution2D(None, ngf*4, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            dconv4 = L.Deconvolution2D(None, ngf*2, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            dconv3 = L.Deconvolution2D(None, ngf,   ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            dconv1 = L.Deconvolution2D(None, nc,    ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            dbn6 = L.BatchNormalization(ngf*8, initial_gamma=bn_init),
            dbn5 = L.BatchNormalization(ngf*4, initial_gamma=bn_init),
            dbn4 = L.BatchNormalization(ngf*2, initial_gamma=bn_init),
            dbn3 = L.BatchNormalization(ngf,   initial_gamma=bn_init)
        )
        
        self.image_size = image_size
        if self.image_size == 128:
            self.add_link('conv2', L.Convolution2D(None, nef,   ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
            self.add_link('bn2',   L.BatchNormalization(nef,   initial_gamma=bn_init))
            self.add_link('dconv2',L.Deconvolution2D(None, ngf,   ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
            self.add_link('dbn2',  L.BatchNormalization(ngf,   initial_gamma=bn_init))

    
    def encode(self, x, test=False):
        h = F.leaky_relu(self.conv1(x))
        if self.image_size == 128:
            h = F.leaky_relu(self.bn2(self.conv2(h), test=test))
        h = F.leaky_relu(self.bn3(self.conv3(h), test=test))
        h = F.leaky_relu(self.bn4(self.conv4(h), test=test))
        h = F.leaky_relu(self.bn5(self.conv5(h), test=test))
        h = F.leaky_relu(self.bn6(self.conv6(h), test=test))

        return h

    def decode(self, x, test=False):
        h = F.relu(self.dbn6(self.dconv6(x), test=test))
        h = F.relu(self.dbn5(self.dconv5(h), test=test))
        h = F.relu(self.dbn4(self.dconv4(h), test=test))
        h = F.relu(self.dbn3(self.dconv3(h), test=test))
        if self.image_size == 128:
            h = F.relu(self.dbn2(self.dconv2(h), test=test))
        h = F.tanh(self.dconv1(h))

        return h

    def __call__(self, x, test=False):
        h = self.encode(x, test=test)
        h = self.decode(h, test=test)

        return h

class EncoderDecoderDiscriminator(chainer.Chain):
    def __init__(self, ndf, image_size=64, conv_init=None, bn_init=None):
        super(EncoderDecoderDiscriminator, self).__init__(
            conv1 = L.Convolution2D(None, ndf,   ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            conv3 = L.Convolution2D(None, ndf*2, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            conv4 = L.Convolution2D(None, ndf*4, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            conv5 = L.Convolution2D(None, ndf*8, ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True),
            conv6 = L.Convolution2D(None, 1,     ksize=4, stride=4, pad=0, initialW=conv_init, nobias=True),
            bn3 = L.BatchNormalization(ndf*2, initial_gamma=bn_init),
            bn4 = L.BatchNormalization(ndf*4, initial_gamma=bn_init),
            bn5 = L.BatchNormalization(ndf*8, initial_gamma=bn_init),
        )
        
        self.image_size = image_size
        if self.image_size == 128:
            self.add_link('conv2', L.Convolution2D(None, ndf,   ksize=4, stride=2, pad=1, initialW=conv_init, nobias=True))
            self.add_link('bn2',   L.BatchNormalization(ndf,   initial_gamma=bn_init))

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.conv1(x))
        if self.image_size == 128:
            h = F.leaky_relu(self.bn2(self.conv2(h), test=test))
        h = F.leaky_relu(self.bn3(self.conv3(h), test=test))
        h = F.leaky_relu(self.bn4(self.conv4(h), test=test))
        h = F.leaky_relu(self.bn5(self.conv5(h), test=test))

        h = self.conv6(h)
        h = F.sum(h, axis=0) / h.shape[0]
        
        return F.squeeze(h)
    
class Discriminator(chainer.Chain):
    """
        PatchGAN
        Input: nc x 256 x 256
    """
    def __init__(self, n_layers, feature_map_nc, w_init=None):
        self.n_layers = n_layers
        layers = {
            'c0': L.Convolution2D(None, feature_map_nc, ksize=4, stride=2, pad=1, initialW=w_init)
        }
        for idx in range(1, n_layers):
            nc_mult = min(2**idx, 8)
            layers['c{}'.format(idx)] = L.Convolution2D(None, feature_map_nc*nc_mult, ksize=4, stride=2, pad=1, initialW=w_init)
            layers['b{}'.format(idx)] = L.BatchNormalization(feature_map_nc*nc_mult)
        nc_mult = min(2**n_layers, 8)
        layers['c{}'.format(n_layers)] = L.Convolution2D(None, feature_map_nc*nc_mult, ksize=4, stride=1, pad=1, initialW=w_init)
        layers['b{}'.format(n_layers)] = L.BatchNormalization(feature_map_nc*nc_mult)
        layers['c'] = L.Convolution2D(None, 1, ksize=4, stride=1, pad=1, initialW=w_init)

        super(Discriminator, self).__init__(**layers)

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.c0(x))

        for idx in range(1, self.n_layers):
            h = F.leaky_relu(self['b{}'.format(idx)](self['c{}'.format(idx)](h), test=test))

        h = F.leaky_relu(self['b{}'.format(self.n_layers)](self['c{}'.format(self.n_layers)](h), test=test))
        h = F.sigmoid(self.c(h))

        return h