import numpy as np
import os
import pylab
import chainer
from chainer import Variable, cuda, initializers, serializers, Chain, optimizers
import chainer.links as L
import chainer.functions as F
import argparse
import math
from model import discriminator, generator

def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))

def set_optimizer(model, alpha, beta):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimizer.setup(model)
    return optimizer

def loss_func_dcgan_dis_real(h):
    return F.sum(F.softplus(-h)) / batchsize

def loss_func_dcgan_dis_fake(h):
    return F.sum(F.softplus(h)) / batchsize

parser = argparse.ArgumentParser(description="test")
parser.add_argument("--type", default = None, help = "select gan type, Normal or RaGAN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--interval", default = 10, type = int, help = "the interval of snapshot")
parser.add_argument("--batchsize", default = 100, type = int, help = "batch size")
parser.add_argument("--lam1", default = 10.0, type = float, help = "the weight of regularizer")

args = parser.parse_args()
gan_type = args.type
epochs = args.epoch
interval = args.interval
batchsize = args.batchsize
lambda1 = args.lam1
wid = int(math.sqrt(batchsize))

xp = cuda.cupy
cuda.get_device(0).use()

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

x_train = np.load('../DCGAN/face_getchu.npy').astype(np.float32)
Ntrain, channels, width, height = x_train.shape

gen_model = generator()
dis_model = discriminator()

gen_model.to_gpu()
dis_model.to_gpu()

gen_opt = set_optimizer(gen_model, 1e-4, 0.5)
dis_opt = set_optimizer(dis_model, 1e-4, 0.5)

zvis = xp.random.uniform(-1,1,(batchsize,100),dtype=xp.float32)

for epoch in range(epochs):
    sum_dis_loss = np.float32(0)
    sum_gen_loss = np.float32(0)
    for batch in range(0,Ntrain,batchsize):
        x_dis = np.zeros((batchsize,channels,width,height), dtype=np.float32)
        for j in range(batchsize):
            rnd = np.random.randint(Ntrain)
            x_dis[j,:,:,:] = x_train[rnd]

        x_dis = cuda.to_gpu(x_dis)

        z = Variable(xp.random.uniform(-1,1,(batchsize,100), dtype = xp.float32))
        x_fake = gen_model(z)
        y_fake = dis_model(x_fake)

        std_data = xp.std(x_dis, axis=0, keepdims = True)
        rnd_x = xp.random.uniform(0,1,x_dis.shape).astype(xp.float32)
        x_perturbed = Variable(cuda.to_gpu(x_dis + 0.5*rnd_x*std_data))

        x_real = Variable(x_dis)
        y_dis = dis_model(x_real)
        y_perturbed = dis_model(x_perturbed)
        grad, = chainer.grad([y_perturbed],[x_perturbed], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        loss_grad = lambda1 * F.mean_squared_error(grad, xp.ones_like(grad.data))

        if gan_type == "Normal":
            loss_dis = loss_func_dcgan_dis_real(y_dis) + loss_func_dcgan_dis_fake(y_fake) + loss_grad
            loss_gen = loss_func_dcgan_dis_real(y_fake)

        if gan_type == "RaGAN":
            fake_mean = F.broadcast_to(F.mean(y_fake), (batchsize,1))
            real_mean = F.broadcast_to(F.mean(y_dis), (batchsize,1))

            loss_dis = (loss_func_dcgan_dis_real(y_dis - fake_mean) + loss_func_dcgan_dis_fake(y_fake - real_mean))/2 + loss_grad
            loss_gen = (loss_func_dcgan_dis_real(y_fake - real_mean) + loss_func_dcgan_dis_fake(y_dis - fake_mean))/2

        dis_model.cleargrads()
        loss_dis.backward()
        dis_opt.update()

        loss_gen = loss_func_dcgan_dis_real(y_fake)
        gen_model.cleargrads()
        loss_gen.backward()
        gen_opt.update()

        sum_dis_loss += loss_dis.data.get()
        sum_gen_loss += loss_gen.data.get()

        if epoch%interval==0 and batch ==0:
            serializers.save_npz('discriminator_getchu.model',dis_model)
            serializers.save_npz('generator.model_getchu',gen_model)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            z = zvis
            z = Variable(z)
            with chainer.using_config('train',False):
                x = gen_model(z)
            x = x.data.get()
            for i_ in range(batchsize):
                tmp = (np.clip((x[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(10,10,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(image_out_dir, epoch))

    print('epoch : {} dis_loss : {} gen_loss : {}'.format(epoch,sum_dis_loss/Ntrain,sum_gen_loss/Ntrain))