import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, Variable, serializers, optimizers
import numpy as np
import os
import pylab
import argparse
from model import Generator, Discriminator
import math

xp = cuda.cupy
cuda.get_device(0).use()

def BCE(x,t):
    return F.average(x - x * t + F.softplus(-x))

def get_fake_tag(dims, threshold):
    prob2 = np.random.rand(34)
    tags = np.zeros((dims)).astype("f")
    tags[:] = -1.0
    tags[np.argmax(prob2[0:13])]=1.0
    tags[27 + np.argmax(prob2[27:])] = 1.0
    prob2[prob2<threshold] = -1.0
    prob2[prob2>=threshold] = 1.0
    
    for i in range(13, 27):
            tags[i] = prob2[i]
            
    return tags

def get_fake_tag_batch(batchsize, dims, threshold):
    tags = xp.zeros((batchsize, dims)).astype("f")
    for i in range(batchsize):
        tags[i] = xp.asarray(get_fake_tag(dims, threshold))
        
    return tags

def set_optimizer(model, alpha=0.00002, beta=0.5):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimizer.setup(model)

    return optimizer

def loss_func_dcgan_dis_real(h):
    return F.sum(F.softplus(-h)) / np.prod(h.data.shape)

def loss_func_dcgan_dis_fake(h):
    return F.sum(F.softplus(h)) / np.prod(h.data.shape)

parser = argparse.ArgumentParser(description="SRResNet")
parser.add_argument("--epoch", default=1000, type=int, help="the number of epochs")
parser.add_argument("--batchsize", default=100, type=int, help="batch size")
parser.add_argument("--interval", default=10, type=int, help="the interval of snapshot")
parser.add_argument("--lam1", default=10.0, type=float, help="the weight of gradient penalty")
parser.add_argument("--lam2", default=34.0 , type=float, help="the weight of adversarial loss")
parser.add_argument("--type", default = None, help ="select Normal or RaGAN")
parser.add_argument("--thre", default = 0.75, type = float, help = "threshold")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
lambda1 = args.lam1
lambda2 = args.lam2
gan_type = args.type
threshold = args.thre
wid = int(math.sqrt(batchsize))

x_train = np.load("../DCGAN/face_getchu.npy").astype(np.float32)
x_label = np.load("../DCGAN/face_tag.npy").astype(np.float32)
print(x_train.shape)
Ntrain, channels, width, height = x_train.shape
_, dims = x_label.shape

image_dir = "./output/"
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator, alpha=0.0002, beta=0.5)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

zvis = xp.random.uniform(-1,1,(batchsize,128),dtype=xp.float32)
ztag = cuda.to_gpu(get_fake_tag_batch(batchsize, dims, threshold))
zenter = F.concat([Variable(zvis), Variable(ztag)])

for epoch in range(epochs):
    sum_dis_loss = np.float32(0)
    sum_gen_loss = np.float32(0)
    for batch in range(0,Ntrain,batchsize):
        x_dis = np.zeros((batchsize,channels,width,height), dtype=np.float32)
        t_dis = np.zeros((batchsize, dims), dtype = np.float32)
        for j in range(batchsize):
            rnd = np.random.randint(Ntrain)
            x_dis[j,:,:,:] = x_train[rnd]
            t_dis[j,:] = x_label[rnd]

        x_dis = cuda.to_gpu(x_dis)
        t_dis = cuda.to_gpu(t_dis)

        z = Variable(xp.random.uniform(-1,1,(batchsize,128), dtype = xp.float32))
        label = cuda.to_gpu(get_fake_tag_batch(batchsize, dims, threshold))
        z = F.concat([z, Variable(label)])
        x_fake = generator(z)
        y_fake, y_fake_cls = discriminator(x_fake)
        label[label < 0] = 0.0
        label = Variable(label)
        loss_gen_class = BCE(y_fake_cls, label)

        loss_gen = lambda2 * loss_func_dcgan_dis_real(y_fake) + loss_gen_class
        generator.cleargrads()
        loss_gen.backward()
        loss_gen.unchain_backward()
        gen_opt.update()

        std_data = xp.std(x_dis, axis=0, keepdims = True)
        rnd_x = xp.random.uniform(0,1,x_dis.shape).astype(xp.float32)
        x_perturbed = Variable(cuda.to_gpu(x_dis + 0.5*rnd_x*std_data))

        x_real = Variable(x_dis)
        y_dis,  y_dis_cls= discriminator(x_real)
        y_perturbed, _  = discriminator(x_perturbed)
        grad, = chainer.grad([y_perturbed],[x_perturbed], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        loss_grad = lambda1 * F.mean_squared_error(grad, xp.ones_like(grad.data))

        t_dis[t_dis < 0] = 0.0
        t_dis = Variable(t_dis)
        loss_dis_class = BCE(y_dis_cls, t_dis)

        if gan_type == "Normal":
            loss_dis = lambda2 * (loss_func_dcgan_dis_real(y_dis) + loss_func_dcgan_dis_fake(y_fake))  + loss_dis_class + loss_grad

        if gan_type == "RaGAN":
            fake_mean = F.broadcast_to(F.mean(y_fake), (batchsize,1))
            real_mean = F.broadcast_to(F.mean(y_dis), (batchsize,1))

            loss_dis = (loss_func_dcgan_dis_real(y_dis - fake_mean) + loss_func_dcgan_dis_fake(y_fake - real_mean))/2 + loss_grad
            loss_dis += loss_dis_class
            loss_gen = (loss_func_dcgan_dis_real(y_fake - real_mean) + loss_func_dcgan_dis_fake(y_dis - fake_mean))/2

        discriminator.cleargrads()
        loss_dis.backward()
        loss_dis.unchain_backward()
        dis_opt.update()

        sum_dis_loss += loss_dis.data.get()
        sum_gen_loss += loss_gen.data.get()

        if epoch%interval==0 and batch ==0:
            serializers.save_npz('discriminator_getchu.model',discriminator)
            serializers.save_npz('generator.model_getchu',generator)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config('train',False):
                x = generator(zenter)
            x = x.data.get()
            for i_ in range(batchsize):
                tmp = (np.clip((x[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(wid,wid,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%svisualize_%d.png'%(image_dir, epoch))

    print('epoch : {} dis_loss : {} gen_loss : {}'.format(epoch,sum_dis_loss/Ntrain,sum_gen_loss/Ntrain))