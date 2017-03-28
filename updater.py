import os

import chainer
import chainer.functions as F
from chainer import Variable

import numpy as np
from skimage.io import imread, imsave
from skimage.transform import resize

from model import extract, gram, total_variation, extract
from utils import im_preprocess_vgg, im_deprocess_vgg, make_kernel, make_grid

def generate_grid_images(G, batch, gpu):
    inputv = Variable(chainer.dataset.concat_examples(batch, gpu), volatile='on')
    fake = G(inputv, test=True)
    fake = chainer.cuda.to_cpu(fake.data)
    img = make_grid(fake)
    img = np.asarray(np.transpose(np.clip((img+1)*127.5, 0, 255), (1, 2, 0)), dtype=np.uint8)

    return img

def encoder_sampler(G, dst, train_batch, val_batch, gpu):
    @chainer.training.make_extension()
    def make_image(trainer):
        imsave(os.path.join(dst, 'fake_samples_train_{}.png'.format(trainer.updater.iteration)), generate_grid_images(G, train_batch, gpu))
        imsave(os.path.join(dst, 'fake_samples_val_{}.png'.format(trainer.updater.iteration)), generate_grid_images(G, val_batch, gpu))

    return make_image

def sampler(G, dst, noisev):
    @chainer.training.make_extension()
    def make_image(trainer):
        fake = G(noisev, test=True)
        fake = chainer.cuda.to_cpu(fake.data)
        img = make_grid(fake)
        img = np.asarray(np.transpose(np.clip((img+1)*127.5, 0, 255), (1, 2, 0)), dtype=np.uint8)

        imsave(os.path.join(dst, 'fake_samples_{}.png'.format(trainer.updater.iteration)), img)

    return make_image

def poisson_editing(transformer, valset, dst, device, sub_mean=True):
    @chainer.training.make_extension()
    def make_image(trainer):
        preview_dir = os.path.join(dst, 'preview')
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        obj, bg, expand_mask, mask = valset.get_example(0)
        inputs = chainer.dataset.concat_examples([np.concatenate((bg*(1-mask), obj*expand_mask))], device)
        if not sub_mean:
            inputs = inputs/127.5-1.0
        input_var = Variable(inputs, volatile='on')

        x_var = transformer(input_var, test=True)
        x = np.squeeze(chainer.cuda.to_cpu(x_var.data))
        x = np.transpose(x, (1, 2, 0))
        x = resize(x, obj.shape[1:], order=1, preserve_range=True, mode='constant')
        x = np.transpose(x, (2, 0, 1))

        if not sub_mean:
            x = (x+1.0)*127.5
        out = np.concatenate((bg*(1-mask)+obj*mask, x, bg*(1-mask)+x*mask), axis=2)
        out_img = im_deprocess_vgg(out, dtype=np.uint8, add_mean=sub_mean)

        imsave(os.path.join(preview_dir, '{}.png'.format(trainer.updater.iteration)), out_img)

    return make_image

def display_image(G, valset, dst, device):
    @chainer.training.make_extension()
    def make_image(trainer):
        preview_dir = os.path.join(dst, 'preview')
        if not os.path.exists(preview_dir):
            os.makedirs(preview_dir)

        idx = np.random.randint(0, len(valset))
        img = valset.get_example(idx)
        input_var = Variable(chainer.dataset.concat_examples([img], device), volatile='on')

        out_var = G(input_var, test=True)
        out = np.squeeze(chainer.cuda.to_cpu(out_var.data))
        out_img = im_deprocess_vgg(out, dtype=np.uint8)

        name = valset.get_name(idx)
        imsave(os.path.join(preview_dir, name), out_img)

    return make_image

class StyleUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.G, self.D = kwargs.pop('models')
        self.args = kwargs.pop('args')
        self.args.content_layers = set(self.args.content_layers)
        self.args.style_layers = set(self.args.style_layers)
        self.layers = self.args.content_layers | self.args.style_layers
        
        print('Extract style feature from {} ...\n'.format(self.args.style_image_path))
        style_image = im_preprocess_vgg(imread(self.args.style_image_path), load_size=self.args.style_load_size, dtype=np.float32)
        style_image_var = Variable(chainer.dataset.concat_examples([style_image], self.args.gpu), volatile='on')
        style_features = extract({'data': style_image_var}, self.D, self.args.style_layers)
        self.grams = {}
        for key, value in style_features.items():
            gram_feature = gram(value[0])
            _, w, h = gram_feature.shape
            gram_feature = F.broadcast_to(gram_feature, (self.args.batch_size, w, h))
            gram_feature.volatile = 'off'
            self.grams[key] = gram_feature
    
        super(StyleUpdater, self).__init__(*args, **kwargs)

    def loss(self, ouput_features, content_features, output_var):
        content_loss = 0
        for layer in self.args.content_layers:
            content_loss += F.mean_squared_error(content_features[layer], ouput_features[layer][0])

        style_loss = 0
        for layer in self.args.style_layers:
            style_loss += F.mean_squared_error(self.grams[layer], gram(ouput_features[layer][0]))

        tv_loss = total_variation(output_var)

        loss = self.args.content_weight*content_loss + self.args.style_weight*style_loss + self.args.tv_weight*tv_loss
        chainer.report({'content_loss': content_loss, 'style_loss': style_loss, 'tv_loss': tv_loss, 'loss': loss}, self.G)

        return loss
    
    def update_core(self):
        batch = self.get_iterator('main').next()
        input_var = Variable(self.converter(batch, self.device), volatile='on')

        content_features = extract({'data': input_var}, self.D, self.args.content_layers)
        content_features = {key:value[0] for key, value in content_features.items()}
        for _, value in content_features.items():
            value.volatile = 'off'
        
        input_var.volatile = 'off'
        output_var = self.G(input_var)
        ouput_features = extract({'data': output_var}, self.D, self.layers)
    
        optimizer = self.get_optimizer('main')
        optimizer.update(self.loss, ouput_features, content_features, output_var)

class PoissonUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.transformer, self.realism_cnn, self.vgg = kwargs.pop('models')
        self.args = kwargs.pop('args')

        self.W_laplace = Variable(make_kernel(3, 3, np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)), volatile='auto')
        self.W_laplace.to_gpu()
        
        super(PoissonUpdater, self).__init__(*args, **kwargs)

    def loss(self, content_laplace, x_var, mask_var, content_features, x_features, inverse_border_mask_var, bg_var, inverse_mask_var):
        # Poisson loss
        poisson_loss = F.sum(F.square((content_laplace - F.convolution_2d(x_var, W=self.W_laplace, pad=1))*inverse_border_mask_var))
        
        # tv loss
        tv_loss = total_variation(x_var)

        # Realism loss
        y = self.realism_cnn(x_var, dropout=False)
        realism_loss = F.sum(y[:, 0, :, :])

        # Content loss
        # content_loss = F.sum(F.square((bg_var - x_var)*inverse_mask_var))
        for layer in self.args.content_layers:
            content_loss += F.mean_squared_error(content_features[layer], x_features[layer])

        loss = self.args.poisson_weight * poisson_loss + self.args.tv_weight * tv_loss + self.args.realism_weight * realism_loss + self.args.content_weight * content_loss
        chainer.report({'content_loss':content_loss, 'poisson_loss': poisson_loss, 'tv_loss': tv_loss, 'realism_loss':realism_loss,'loss': loss}, self.transformer)

        return loss
    
    def update_core(self):
        batch = self.get_iterator('main').next()
        obj_var = Variable(self.converter([obj for obj, _, _, _ in batch], self.device), volatile='on')
        bg_var = Variable(self.converter([bg for _, bg, _, _ in batch], self.device), volatile='auto')
        expand_mask_var = F.broadcast_to(Variable(self.converter([expand_mask for _, _, expand_mask, _ in batch], self.device), volatile='auto'), obj_var.shape)
        mask_var = F.broadcast_to(Variable(self.converter([mask for _, _, _, mask in batch], self.device), volatile='auto'), obj_var.shape)
        inverse_mask_var = 1-mask_var
        inverse_border_mask_var = mask_var + 1 - expand_mask_var

        # Content Laplace
        # content_laplace = F.convolution_2d(obj_var, W=self.W_laplace, pad=1)
        content_laplace = F.convolution_2d(obj_var, W=self.W_laplace, pad=1)*mask_var + F.convolution_2d(bg_var, W=self.W_laplace, pad=1)*inverse_mask_var
        content_laplace.volatile = 'off'

        # Content feature
        content_features = extract({'data': obj_var*mask_var+bg_var*inverse_mask_var}, self.vgg, self.args.content_layers)
        content_features = {key:value[0] for key, value in content_features.items()}
        for value in content_features.values():
            value.volatile = 'off'

        # Input
        obj_var.volatile = 'off'
        input_var = F.concat((bg_var*inverse_mask_var, obj_var*expand_mask_var))
        output_var = self.transformer(input_var)
        # x_var = output_var*mask_var + bg_var*inverse_mask_var
        x_var = output_var

        x_features = extract({'data': x_var}, self.vgg, self.args.content_layers)
        x_features = {key:value[0] for key, value in x_features.items()}

        # BP
        optimizer = self.get_optimizer('main')
        optimizer.update(self.loss, content_laplace, x_var, mask_var, content_features, x_features, inverse_border_mask_var, bg_var, inverse_mask_var)

class MaskUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.transformer, self.discriminator = kwargs.pop('models')
        self.args = kwargs.pop('args')
        self.eps = 10**-12

        self.W_laplace = Variable(make_kernel(3, 3, np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)), volatile='auto')
        self.W_laplace.to_gpu()
        
        super(MaskUpdater, self).__init__(*args, **kwargs)

    def loss(self, content_laplace, x_var, mask_var, fake_D):
        batch_size, _, h, w = x_var.shape
        # Poisson loss
        poisson_loss = F.sum(F.square((content_laplace - F.convolution_2d(x_var, W=self.W_laplace, pad=1))*mask_var)) / (batch_size*h*w)

        # tv loss
        tv_loss = total_variation(x_var)

        batch_size, _, h, w = fake_D.shape
        dis_loss = - F.sum(F.log(fake_D + self.eps)) / (batch_size*h*w)

        loss = self.args.poisson_weight * poisson_loss + self.args.tv_weight * tv_loss + self.args.d_weight * dis_loss
        chainer.report({'poisson_loss': poisson_loss, 'tv_loss': tv_loss, 'dis_loss':dis_loss, 'loss': loss}, self.transformer)

        return loss
    
    def d_loss(self, fake_D, real_D):
        batch_size, _, h, w = real_D.shape

        loss = - F.sum(F.log(real_D + self.eps) + F.log(1 - fake_D + self.eps)) / (2*batch_size*h*w)
        chainer.report({'loss': loss}, self.discriminator)

        return loss

    def update_core(self):
        batch = self.get_iterator('main').next()
        obj_var = Variable(self.converter([obj for obj, _, _, _ in batch], self.device)/127.5-1.0, volatile='on')
        bg_var = Variable(self.converter([bg for _, bg, _, _ in batch], self.device)/127.5-1.0, volatile='auto')
        expand_mask_var = F.broadcast_to(Variable(self.converter([expand_mask for _, _, expand_mask, _ in batch], self.device), volatile='auto'), obj_var.shape)
        mask_var = F.broadcast_to(Variable(self.converter([mask for _, _, _, mask in batch], self.device), volatile='auto'), obj_var.shape)
        inverse_mask_var = 1-mask_var

        # Content Laplace
        content_laplace = F.convolution_2d(obj_var, W=self.W_laplace, pad=1)
        content_laplace.volatile = 'off'

        # Input
        obj_var.volatile = 'off'
        input_var = F.concat((bg_var*inverse_mask_var, obj_var*expand_mask_var))
        x_var = self.transformer(input_var)
        x_var = x_var*mask_var + bg_var*inverse_mask_var

        # Discriminator
        fake_D = self.discriminator(x_var)
        real_D = self.discriminator(bg_var)

        # BP
        optimizer = self.get_optimizer('main')
        dis_optimizer = self.get_optimizer('discriminator')
        optimizer.update(self.loss, content_laplace, x_var, mask_var, fake_D)
        dis_optimizer.update(self.d_loss, fake_D, real_D)

class GradientUpdater(chainer.training.StandardUpdater):

    def __init__(self, *args, **kwargs):
        self.transformer, self.discriminator = kwargs.pop('models')
        self.args = kwargs.pop('args')
        self.eps = 10**-12

        self.W_laplace = Variable(make_kernel(3, 3, np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)), volatile='auto')
        self.W_laplace.to_gpu()
        
        super(GradientUpdater, self).__init__(*args, **kwargs)

    def loss(self, content_laplace, x_var, mask_var, fake_D):
        # Poisson loss
        poisson_loss = F.sum(F.square(content_laplace - F.convolution_2d(x_var, W=self.W_laplace, pad=1))*mask_var)
        
        # # L1 loss
        # l1_loss = F.mean_absolute_error(x_var, bg_var)

        # tv loss
        tv_loss = total_variation(x_var)

        batch_size, _, h, w = fake_D.shape
        dis_loss = - F.sum(F.log(fake_D + self.eps)) / (batch_size*h*w)

        loss = self.args.poisson_weight * poisson_loss + self.args.tv_weight * tv_loss + self.args.d_weight * dis_loss
        chainer.report({'poisson_loss': poisson_loss, 'tv_loss': tv_loss, 'dis_loss':dis_loss, 'loss': loss}, self.transformer)

        return loss
    
    def d_loss(self, fake_D, real_D):
        batch_size, _, h, w = real_D.shape

        loss = - F.sum(F.log(real_D + self.eps) + F.log(1 - fake_D + self.eps)) / (2*batch_size*h*w)
        chainer.report({'loss': loss}, self.discriminator)

        return loss

    def update_core(self):
        batch = self.get_iterator('main').next()
        obj_var = Variable(self.converter([obj for obj, _, _, _ in batch], self.device)/127.5-1.0, volatile='on')
        bg_var = Variable(self.converter([bg for _, bg, _, _ in batch], self.device)/127.5-1.0, volatile='auto')
        mask_var = F.broadcast_to(Variable(self.converter([mask for _, _, _, mask in batch], self.device), volatile='auto'), obj_var.shape)
        inverse_mask_var = 1-mask_var

        # Content Laplace
        content_laplace = F.convolution_2d(obj_var, W=self.W_laplace, pad=1)
        content_laplace.volatile = 'off'

        # Input
        obj_var.volatile = 'off'
        input_var = F.concat((bg_var*inverse_mask_var, obj_var*mask_var))
        x_var = self.transformer(input_var)
        x_var = x_var*mask_var + bg_var*inverse_mask_var

        # Discriminator
        fake_D = self.discriminator(x_var)
        real_D = self.discriminator(bg_var)

        # BP
        optimizer = self.get_optimizer('main')
        dis_optimizer = self.get_optimizer('discriminator')
        optimizer.update(self.loss, content_laplace, x_var, mask_var, fake_D)
        dis_optimizer.update(self.d_loss, fake_D, real_D)

class WassersteinUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.G, self.D = kwargs.pop('models')
        self.args = kwargs.pop('args')

        super(WassersteinUpdater, self).__init__(*args, **kwargs)

    def d_loss(self, errD_real, errD_fake):
        errD = errD_real - errD_fake

        chainer.report({'loss_real':errD_real}, self.D)
        chainer.report({'loss_fake':errD_fake}, self.D)
        chainer.report({'loss':errD}, self.D)

        return errD

    def g_loss(self, errG):
        chainer.report({'loss':errG}, self.G)

        return errG

    def update_core(self):
        d_optimizer = self.get_optimizer('D')
        g_optimizer = self.get_optimizer('main')
        ############################
        # (1) Update D network
        ###########################
        # train the discriminator Diters times
        if self.iteration < 25 or self.iteration % 500 == 0:
            Diters = 100
        else:
            Diters = self.args.d_iters

        for _ in range(Diters):
            # clamp parameters to a cube
            for p in self.D.params():
                p.data.clip(self.args.clamp_lower, self.args.clamp_upper, p.data)

            batch = self.get_iterator('main').next()
            inputv = Variable(self.converter(batch, self.device))
            errD_real = self.D(inputv)

            # train with fake
            noisev = Variable(np.asarray(np.random.normal(size=(self.args.batch_size, self.args.nz, 1, 1)), dtype=np.float32))
            noisev.to_gpu(self.device)
            fake = self.G(noisev)
            errD_fake = self.D(fake)
            
            d_optimizer.update(self.d_loss, errD_real, errD_fake)

        ############################
        # (2) Update G network
        ###########################
        noisev = Variable(np.asarray(np.random.normal(size=(self.args.batch_size, self.args.nz, 1, 1)), dtype=np.float32))
        noisev.to_gpu(self.device)
        fake = self.G(noisev)
        errG = self.D(fake)
        g_optimizer.update(self.g_loss, errG)

class EncoderDecoderUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.G, self.D = kwargs.pop('models')
        self.args = kwargs.pop('args')
        # self.eps = 10**-12

        super(EncoderDecoderUpdater, self).__init__(*args, **kwargs)

    def d_loss(self, errD_real, errD_fake):
        errD = errD_real - errD_fake

        chainer.report({'loss_real':errD_real}, self.D)
        chainer.report({'loss_fake':errD_fake}, self.D)
        chainer.report({'loss':errD}, self.D)

        return errD
    # def d_loss(self, errD_real, errD_fake):
    #     loss = - F.sum(F.log(errD_real + self.eps) + F.log(1 - errD_fake + self.eps)) / (2*self.args.batch_size)
    #     chainer.report({'loss': loss}, self.D)

    #     return loss

    def g_loss(self, errG, fake, inputv):
        l2_loss = F.mean_squared_error(fake, inputv)
        loss = (1-self.args.l2_weight)*errG + self.args.l2_weight*l2_loss

        chainer.report({'loss':loss}, self.G)
        chainer.report({'l2_loss':l2_loss}, self.G)
        chainer.report({'gan_loss':errG}, self.G)

        return loss

    # def update_core(self):
    #     d_optimizer = self.get_optimizer('D')
    #     g_optimizer = self.get_optimizer('main')
    #     ############################
    #     # (1) Update D network
    #     ###########################
        
    #     batch = self.get_iterator('main').next()
    #     inputv = Variable(self.converter(batch, self.device))
    #     errD_real = self.D(inputv)

    #     fake = self.G(inputv)
    #     errD_fake = self.D(fake)
            
    #     d_optimizer.update(self.d_loss, errD_real, errD_fake)

    #     ############################
    #     # (2) Update G network
    #     ###########################
    #     g_optimizer.update(self.g_loss, errD_fake, fake, inputv)
    def update_core(self):
        d_optimizer = self.get_optimizer('D')
        g_optimizer = self.get_optimizer('main')
        ############################
        # (1) Update D network
        ###########################
        # train the discriminator Diters times
        if self.iteration < 25 or self.iteration % 500 == 0:
            Diters = 100
        else:
            Diters = self.args.d_iters
        
        for _ in range(Diters):
            # clamp parameters to a cube
            for p in self.D.params():
                p.data.clip(self.args.clamp_lower, self.args.clamp_upper, p.data)

            batch = self.get_iterator('main').next()
            inputv = Variable(self.converter(batch, self.device))
            errD_real = self.D(inputv)

            # train with fake
            fake = self.G(inputv)
            errD_fake = self.D(fake)
            
            d_optimizer.update(self.d_loss, errD_real, errD_fake)

        ############################
        # (2) Update G network
        ###########################
        batch = self.get_iterator('main').next()
        inputv = Variable(self.converter(batch, self.device))
        fake = self.G(inputv)
        errG = self.D(fake)
        g_optimizer.update(self.g_loss, errG, fake, inputv)

class EncoderDecoderBlendingUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.G, self.D = kwargs.pop('models')
        self.args = kwargs.pop('args')

        super(EncoderDecoderBlendingUpdater, self).__init__(*args, **kwargs)

    def d_loss(self, errD_real, errD_fake):
        errD = errD_real - errD_fake

        chainer.report({'loss_real':errD_real}, self.D)
        chainer.report({'loss_fake':errD_fake}, self.D)
        chainer.report({'loss':errD}, self.D)

        return errD

    def g_loss(self, errG, fake, gtv):
        l2_loss = F.mean_squared_error(fake, gtv)
        loss = (1-self.args.l2_weight)*errG + self.args.l2_weight*l2_loss

        chainer.report({'loss':loss}, self.G)
        chainer.report({'l2_loss':l2_loss}, self.G)
        chainer.report({'gan_loss':errG}, self.G)

        return loss

    def update_core(self):
        d_optimizer = self.get_optimizer('D')
        g_optimizer = self.get_optimizer('main')
        ############################
        # (1) Update D network
        ###########################
        # train the discriminator Diters times
        if self.iteration < 25 or self.iteration % 500 == 0:
            Diters = 100
        else:
            Diters = self.args.d_iters

        for _ in range(Diters):
            # clamp parameters to a cube
            for p in self.D.params():
                p.data.clip(self.args.clamp_lower, self.args.clamp_upper, p.data)

            batch = self.get_iterator('main').next()
            inputv = Variable(self.converter([inputs for inputs, _ in batch], self.device))
            gtv    = Variable(self.converter([gt     for _, gt     in batch], self.device))
            errD_real = self.D(gtv)

            # train with fake
            fake = self.G(inputv)
            errD_fake = self.D(fake)
            
            d_optimizer.update(self.d_loss, errD_real, errD_fake)

        ############################
        # (2) Update G network
        ###########################
        batch = self.get_iterator('main').next()
        inputv = Variable(self.converter([inputs for inputs, _ in batch], self.device))
        gtv    = Variable(self.converter([gt     for _, gt     in batch], self.device))
        fake = self.G(inputv)
        errG = self.D(fake)
        g_optimizer.update(self.g_loss, errG, fake, gtv)

class EncoderDecoderBlendingNoGtUpdater(chainer.training.StandardUpdater):
    def __init__(self, *args, **kwargs):
        self.G, self.D = kwargs.pop('models')
        self.args = kwargs.pop('args')
        
        size = int(self.args.image_size*self.args.ratio)
        expand_sx = self.args.image_size//2 - size//2
        expand_ex = expand_sx + size
        self.sx = expand_sx + 1
        self.ex = expand_ex - 1

        self.W_laplace = Variable(make_kernel(3, 3, np.asarray([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)), volatile='auto')
        self.W_laplace.to_gpu()

        mask = np.ones((self.args.batch_size, self.args.nc, self.args.image_size, self.args.image_size), dtype=np.float32)
        mask[:, :, self.sx:self.ex, self.sx:self.ex] = 0
        self.mask_var = Variable(mask, volatile='auto')
        self.mask_var.to_gpu()
        self.inverse_mask_var = 1-self.mask_var

        self.l2_content_num = np.sum(mask)

        super(EncoderDecoderBlendingNoGtUpdater, self).__init__(*args, **kwargs)

    def d_loss(self, errD_real, errD_fake):
        errD = errD_real - errD_fake

        chainer.report({'loss_real':errD_real}, self.D)
        chainer.report({'loss_fake':errD_fake}, self.D)
        chainer.report({'loss':errD}, self.D)

        return errD

    def g_loss(self, errG, fake, inputv, gtv):
        obj_laplace =  F.convolution_2d(inputv, W=self.W_laplace, pad=1)
        fake_laplace = F.convolution_2d(fake*self.inverse_mask_var+gtv*self.mask_var,   W=self.W_laplace, pad=1)
        l2_loss = F.mean_squared_error(obj_laplace[:, :, self.sx:self.ex, self.sx:self.ex], fake_laplace[:, :, self.sx:self.ex, self.sx:self.ex])
        l2_loss += F.sum(F.square((fake - gtv)*self.mask_var)) / self.l2_content_num

        loss = (1-self.args.l2_weight)*errG + self.args.l2_weight*l2_loss

        chainer.report({'loss':loss}, self.G)
        chainer.report({'l2_loss':l2_loss}, self.G)
        chainer.report({'gan_loss':errG}, self.G)

        return loss

    def update_core(self):
        d_optimizer = self.get_optimizer('D')
        g_optimizer = self.get_optimizer('main')
        ############################
        # (1) Update D network
        ###########################
        # train the discriminator Diters times
        if self.iteration < 25 or self.iteration % 500 == 0:
            Diters = 100
        else:
            Diters = self.args.d_iters

        for _ in range(Diters):
            # clamp parameters to a cube
            for p in self.D.params():
                p.data.clip(self.args.clamp_lower, self.args.clamp_upper, p.data)

            batch = self.get_iterator('main').next()
            inputv = Variable(self.converter([inputs for inputs, _ in batch], self.device))
            gtv    = Variable(self.converter([gt     for _, gt     in batch], self.device))
            errD_real = self.D(gtv)

            # train with fake
            fake = self.G(inputv)
            errD_fake = self.D(fake)
            
            d_optimizer.update(self.d_loss, errD_real, errD_fake)

        ############################
        # (2) Update G network
        ###########################
        batch = self.get_iterator('main').next()
        inputv = Variable(self.converter([inputs for inputs, _ in batch], self.device))
        gtv    = Variable(self.converter([gt     for _, gt     in batch], self.device))
        fake = self.G(inputv)
        errG = self.D(fake)
        g_optimizer.update(self.g_loss, errG, fake, inputv, gtv)