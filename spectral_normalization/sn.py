import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, cuda, serializers, Variable, initializers, Chain
import os
import cv2 as cv
import pylab
import argparse
import math
from sn_net import Generator, Discriminator

def set_optimizer(model, alpha, beta):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)
    return optimizer

xp = cuda.cupy
cuda.get_device(0).use()

parser = argparse.ArgumentParser(description="Spectral Normalization for GAN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 100, type = int, help = "batch size")
parser.add_argument("--interval", default = 10, type = int, help = "the interval of snapshot")
args = parser.parse_args()

epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
wid = int(math.sqrt(batchsize))

image_out_dir = './output'

if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

x_train = np.load('../DCGAN/face_getchu.npy').astype(np.float32)
Ntrain, channels, width, height = x_train.shape

gen_model = Generator()
dis_model = Discriminator()

gen_model.to_gpu()
dis_model.to_gpu()

gen_opt = set_optimizer(gen_model, 1e-4, 0.5)
dis_opt = set_optimizer(dis_model, 1e-4, 0.5)

zvis = xp.random.uniform(-1,1,(batchsize,100),dtype=xp.float32)

for epoch in range(epochs):
    sum_dis_loss = np.float32(0)
    sum_gen_loss = np.float32(0)
    for batch in range(0,Ntrain,batchsize):
        x_dis = np.zeros((batchsize, channels, width, height), dtype=np.float32)
        for j in range(batchsize):
            rnd = np.random.randint(Ntrain)
            x_dis[j,:,:,:] = x_train[rnd]

        z = xp.random.uniform(-1,1,(batchsize,100),dtype=xp.float32)
        x_fake = gen_model(z)
        y_fake = dis_model(x_fake)

        gen_loss = F.sum(F.softplus(-y_fake)) / batchsize

        gen_model.cleargrads()
        gen_loss.backward()
        gen_opt.update()

        dis_loss = F.sum(F.softplus(y_fake)) / batchsize

        x_real = Variable(cuda.to_gpu(x_dis))
        y_real = dis_model(x_real)
        dis_loss += F.sum(F.softplus(-y_real)) / batchsize
        
        dis_model.cleargrads()
        dis_loss.backward()
        dis_opt.update()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if epoch%interval==0 and batch ==0:
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            z = zvis
            z = Variable(z)
            with chainer.using_config('train',False):
                x = gen_model(z)
            x = x.data.get()
            for i_ in range(batchsize):
                tmp = (np.clip(x[i_,:,:,:]*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(wid,wid,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig( image_out_dir + '/result_%d.png' %epoch)

    print('epoch : {} dis_loss : {} gen_loss : {}'.format(epoch,sum_dis_loss/Ntrain,sum_gen_loss/Ntrain))