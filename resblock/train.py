import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, cuda, serializers, Variable, initializers, Chain
import os
import cv2 as cv
import pylab
from model import Discriminator, Generator
import argparse
import math
from prepare import prepare_dataset

xp = cuda.cupy
cuda.get_device(0).use()

def set_optimizer(model, alpha, beta1, beta2):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2 = beta2)
    optimizer.setup(model)
    return optimizer

def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))

parser = argparse.ArgumentParser(description = "DCGAN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 100, type = int, help = "batch size")
parser.add_argument("--interval", default = 1, type = int, help = "the interval of snapshot")
parser.add_argument("--lam1", default = 0.0, type = float, help = "the weight of the gradient penalty")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
lambda1 = args.lam1
wid = int(math.sqrt(batchsize))

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

#x_train = np.load('../DCGAN/face_getchu.npy').astype(np.float32)
#print(x_train.shape)
#Ntrain, channels, width, height = x_train.shape
image_path = "/usr/MachineLearning/Dataset/face_illustration/face_getchu/"
image_list = os.listdir(image_path)
Ntrain = 25000

gen_model = Generator()
dis_model = Discriminator()

gen_model.to_gpu()
dis_model.to_gpu()

gen_opt = set_optimizer(gen_model, 0.0002, 0.5, 0.99)
dis_opt = set_optimizer(dis_model, 0.0002, 0.5, 0.99)

zvis = xp.random.uniform(-1,1,(batchsize,128),dtype=np.float32)

dis_loss_list = []
gen_loss_list = []

for epoch in range(epochs):
    sum_dis_loss = np.float32(0)
    sum_gen_loss = np.float32(0)
    for batch in range(0,Ntrain,batchsize):
        #x_dis = np.zeros((batchsize,channels,width,height), dtype=np.float32)
        image_box = []
        for j in range(batchsize):
            rnd = np.random.randint(Ntrain)
            image_name = image_list[rnd]
            image = prepare_dataset(image_path + image_name)
            image_box.append(image)

        image_array = np.array(image_box).astype(np.float32)
        x_dis = cuda.to_gpu(image_array)

        z = Variable(xp.random.uniform(-1,1,(batchsize,128), dtype = xp.float32))
        x = gen_model(z)
        y = dis_model(x)
        dis_loss = F.mean(F.softplus(y))

        #std_data = xp.std(x_dis, axis=0, keepdims = True)
        #rnd_x = xp.random.uniform(0,1,x_dis.shape).astype(xp.float32)
        #x_perturbed = Variable(cuda.to_gpu(x_dis + 0.5*rnd_x*std_data))

        x_dis = Variable(cuda.to_gpu(x_dis))
        y_dis = dis_model(x_dis)
        dis_loss += F.mean(F.softplus(-y_dis))

        #y_perturbed = dis_model(x_perturbed)
        #grad, = chainer.grad([y_perturbed],[x_perturbed], enable_double_backprop=True)
        #grad = F.sqrt(F.batch_l2_norm_squared(grad))
        #loss_grad = lambda1 * F.mean_squared_error(grad, xp.ones_like(grad.data))
        #dis_loss += loss_grad

        dis_model.cleargrads()
        dis_loss.backward()
        dis_loss.unchain_backward()
        dis_opt.update()

        z = Variable(xp.random.uniform(-1,1,(batchsize,128), dtype = xp.float32))
        x = gen_model(z)
        y = dis_model(x)
        gen_loss = F.mean(F.softplus(-y))

        gen_model.cleargrads()
        gen_loss.backward()
        gen_loss.unchain_backward()
        gen_opt.update()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if epoch%interval==0 and batch ==0:
            serializers.save_npz('discriminator.model',dis_model)
            serializers.save_npz('generator.model',gen_model)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            z = zvis
            z = Variable(z)
            with chainer.using_config('train',False):
                x = gen_model(z)
            x = x.data.get()
            for i_ in range(batchsize):
                tmp = np.clip(x[i_,:,:,:]*127.5 + 127.5, 0, 255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(wid,wid,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(image_out_dir, epoch))

    print('epoch : {} dis_loss : {} gen_loss : {}'.format(epoch,sum_dis_loss/Ntrain,sum_gen_loss/Ntrain))