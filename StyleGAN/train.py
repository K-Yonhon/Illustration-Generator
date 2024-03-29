import chainer
import chainer.functions as F
from chainer import cuda,Chain,optimizers,serializers
import numpy as np
import os
from pathlib import Path
import argparse
import matplotlib
matplotlib.use('Agg')
import pylab
import numpy as np
import cv2 as cv
from model import MappingNetwork, Generator, Discriminator, Constant

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimzier(model, alpha=0.0001, beta=0.5):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)

    return optimizer

def prepare_dataset(filename, size = 128):
    if filename.endswith(".png"):
        image_orig = cv.imread(filename)
        image_orig = cv.resize(image_orig,(size,size),interpolation=cv.INTER_CUBIC)
        hflip = np.random.choice([True,False])
        if hflip:
            image_orig = image_orig[:,::-1,:]
        image = image_orig[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5
        
        return image

parser = argparse.ArgumentParser(description="StyleGAN")
parser.add_argument("--e", default=1000, type=int, help="the number of epochs")
parser.add_argument("--b", default=64, type=int, help="batch size")
parser.add_argument("--i", default=2000, type=int, help="the number of iterations")
parser.add_argument("--lgp", default=10.0, type=float, help="the weight of gradient penalty")
parser.add_argument("--stage", default=6, type=int, help="the number of stages")
parser.add_argument("--si", default=1000000, type=int, help="stage interval")
parser.add_argument('--n', default=17000, type=int, help = "the number of train images")

outdir = Path('./outdir')
outdir.mkdir(parents=False, exist_ok=True)

image_path = './face_getchu/'
image_list = os.listdir(image_path)

args = parser.parse_args()
epochs = args.e
batchsize = args.b
iterations = args.i
lambda_gp = args.lgp
stages = args.stage
stage_interval = args.si
Ntrain = args.n

mapping = MappingNetwork()
mapping.to_gpu()
map_opt = set_optimzier(mapping, alpha=0.000001)

generator = Generator()
generator.to_gpu()
gen_opt = set_optimzier(generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimzier(discriminator)

const_class = Constant()
const_class.to_gpu()
const = const_class()
test_latent = chainer.as_variable(xp.random.normal(size=(batchsize, 512)).astype(xp.float32))

ch_list = [512, 512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128]

counter = 0
iteration = 0

for epoch in range(epochs):
    sum_dis_loss = 0
    sum_gen_loss = 0
    stage = int(counter / stage_interval) + 1
    if stage > 5:
        stage = 5
    if stage == 5:
        batchsize = 32
        const = const[:batchsize]
    for batch in range(0, Ntrain, batchsize):
        image_box = []
        alpha = min(1, 0.00002*iteration)
        for _ in range(batchsize):
            rnd = np.random.randint(Ntrain)
            image = prepare_dataset(image_path + image_list[rnd])
            image_box.append(image)

        x = chainer.as_variable(xp.array(image_box).astype(xp.float32))
        x_down = F.average_pooling_2d(x, 2**(5-stage), 2**(5-stage), 0)

        m_list = []
        noise_list = []
        z1 = chainer.as_variable(xp.random.normal(size=(batchsize, 512)).astype(xp.float32))
        z2 = chainer.as_variable(xp.random.normal(size=(batchsize, 512)).astype(xp.float32))
        switch_point = np.random.randint(1, 2*stage)
        for i in range(2*stage + 1 + 1):
            if i >= switch_point:
                z = z2
            else:
                z = z1
            m = mapping(z)
            if i //2 == 0:
                noise = chainer.as_variable(xp.random.normal(size=(batchsize, ch_list[i], 4, 4)).astype(xp.float32))
            else:
                noise = chainer.as_variable(xp.random.normal(size=(batchsize, ch_list[i], 2**(i//2+2), 2**(i//2+2))).astype(xp.float32))

            m_list.append(m)
            noise_list.append(noise)

        y = generator(const, m_list, stage, noise_list, alpha)
        y_dis = discriminator(y, stage, alpha)
        x_dis = discriminator(x_down, stage, alpha)

        dis_loss = F.mean(F.softplus(-x_dis)) + F.mean(F.softplus(y_dis))

        eps = xp.random.uniform(0,1,size = batchsize).astype(xp.float32)[:,None,None,None]
        x_mid = eps * y + (1.0 - eps) * x_down

        y_mid = F.sum(discriminator(x_mid, stage, alpha))
        grad,  = chainer.grad([y_mid], [x_mid], enable_double_backprop=True)
        grad = F.sqrt(F.sum(grad*grad, axis=(1,2,3)))
        loss_gp = lambda_gp * F.mean_squared_error(grad, xp.ones_like(grad.data))

        y.unchain_backward()

        dis_loss += loss_gp

        discriminator.cleargrads()
        dis_loss.backward()
        dis_opt.update()
        dis_loss.unchain_backward()

        m_list = []
        noise_list = []
        z1 = chainer.as_variable(xp.random.normal(size=(batchsize, 512)).astype(xp.float32))
        z2 = chainer.as_variable(xp.random.normal(size=(batchsize, 512)).astype(xp.float32))
        switch_point = np.random.randint(1, 2*stage)
        for i in range(2*stage + 1 + 1):
            if i >= switch_point:
                z = z2
            else:
                z = z1
            m = mapping(z)
            if i //2 == 0:
                noise = chainer.as_variable(xp.random.normal(size=(batchsize, ch_list[i], 4, 4)).astype(xp.float32))
            else:
                noise = chainer.as_variable(xp.random.normal(size=(batchsize, ch_list[i], 2**(i//2+2), 2**(i//2+2))).astype(xp.float32))

            m_list.append(m)
            noise_list.append(noise)

        y = generator(const, m_list, stage, noise_list, alpha)
        y_dis = discriminator(y, stage, alpha)

        gen_loss = F.mean(F.softplus(-y_dis))

        generator.cleargrads()
        mapping.cleargrads()
        gen_loss.backward()
        gen_opt.update()
        map_opt.update()
        gen_loss.unchain_backward()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        counter += batchsize
        iteration += 1

        if batch == 0:
            if epoch % 10 == 0:
                serializers.save_npz('const.model', const_class)
                serializers.save_npz('generator_{}.model'.format(epoch), generator)
                serializers.save_npz('mapping_{}.model'.format(epoch), mapping)
                serializers.save_npz('discriminator_{}.model'.format(epoch),discriminator)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config('train',False):
                m_list = []
                noise_list = []
                z1 = chainer.as_variable(xp.random.normal(size=(batchsize, 512)).astype(xp.float32))
                z2 = chainer.as_variable(xp.random.normal(size=(batchsize, 512)).astype(xp.float32))
                switch_point = np.random.randint(1, 2*stage)
                for i in range(2*stage + 1 + 1):
                    if i >= switch_point:
                        z = z2
                    else:
                        z = z1
                    m = mapping(z)
                    if i //2 == 0:
                        noise = chainer.as_variable(xp.random.normal(size=(batchsize, ch_list[i], 4, 4)).astype(xp.float32))
                    else:
                        noise = chainer.as_variable(xp.random.normal(size=(batchsize, ch_list[i], 2**(i//2+2), 2**(i//2+2))).astype(xp.float32))                        
                    m_list.append(m)
                    noise_list.append(noise)

                x = generator(const,m_list,stage,noise_list, alpha)
            x = x.data.get()
            for i_ in range(batchsize):
                tmp = (np.clip((x[i_,:,:,:]*127.5 + 127.5), 0.0, 255.0)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(8,8,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d_stage%d.png'%(outdir, epoch,stage))

    print('epoch : {}'.format(epoch))
    print('discriminator loss : {}'.format(sum_dis_loss / iterations))
    print('generator loss : {}'.format(sum_gen_loss / iterations))