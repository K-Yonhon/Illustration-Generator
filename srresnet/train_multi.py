import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, Variable, serializers, optimizers
import numpy as np
import os
import pylab
import argparse
from model import Generator, Discriminator_multi, Discriminator_single
import math
from preparing import prepare_dataset
from slack_post import slack

xp = cuda.cupy
cuda.get_device(0).use()

def BCE(x,t):
    return F.average(x - x * t + F.softplus(-x))

def get_fake_tag(dims, threshold):
    prob2 = np.random.rand(9)
    tags = np.zeros((dims)).astype("f")
    tags[:] = 0.0
    tags[np.argmax(prob2)]=1.0
            
    return tags

def get_fake_tag_batch(batchsize, dims, threshold):
    tags = xp.zeros((batchsize, dims)).astype("f")
    for i in range(batchsize):
        tags[i] = xp.asarray(get_fake_tag(dims, threshold))
        
    return tags

def set_optimizer(model, alpha=0.0002, beta=0.5):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta, beta2 = 0.99)
    optimizer.setup(model)

    return optimizer

def loss_func_dcgan_dis_real(h):
    return F.sum(F.softplus(-h)) / np.prod(h.data.shape)

def loss_func_dcgan_dis_fake(h):
    return F.sum(F.softplus(h)) / np.prod(h.data.shape)

def loss_hinge_dis(dis_fake, dis_real):
    loss = F.mean(F.relu(1. - dis_real))
    loss += F.mean(F.relu(1. + dis_fake))
    
    return loss

def loss_hinge_gen(dis_fake):
    loss = -F.mean(dis_fake)
    
    return loss

parser = argparse.ArgumentParser(description="SRResNet")
parser.add_argument("--epoch", default=1000, type=int, help="the number of epochs")
parser.add_argument("--batchsize", default=100, type=int, help="batch size")
parser.add_argument("--testsize", default=25, type=int, help="test size")
parser.add_argument("--interval", default=1, type=int, help="the interval of snapshot")
parser.add_argument("--lam1", default=0.5, type=float, help="the weight of gradient penalty")
parser.add_argument("--lam2", default=1.0 , type=float, help="the weight of adversarial loss")
parser.add_argument("--type", default = None, help ="select Normal or RaGAN")
parser.add_argument("--thre", default = 0.75, type = float, help = "threshold")
parser.add_argument("--size", default = 128, type = int, help = "the width or height of images")
parser.add_argument("--ndis",default=2,type=int,help="the number of discriminatror update in one iteration")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
testsize=args.testsize
interval = args.interval
lambda1 = args.lam1
lambda2 = args.lam2
gan_type = args.type
threshold = args.thre
size = args.size
n_dis = args.ndis
wid = int(math.sqrt(testsize))

x_label = np.load("../DCGAN/face_tag.npy").astype(np.float32)
_, dims = x_label.shape
image_path = "./image/"
Ntrain = 22500
channels = 3
width = size
height = size

image_dir = "./output_9/"
if not os.path.exists(image_dir):
    os.mkdir(image_dir)

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator, alpha = 0.0002)
#serializers.load_npz("generator.model_getchu", generator)

discriminator = Discriminator_multi()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator, alpha = 0.0002)
#serializers.load_npz('discriminator_getchu.model', discriminator)

zvis = xp.random.normal(size=(batchsize,128),dtype=xp.float32)
ztag=[]
for _ in range(batchsize):
    zlist=[0]*9
    zlist[3]=1
    ztag.append(zlist)
ztag = xp.array(ztag).astype(xp.float32)
ztag[ztag < 0.0] = 0.0
zenter = F.concat([Variable(zvis), Variable(ztag)])

for epoch in range(epochs):
    sum_dis_loss = np.float32(0)
    sum_gen_loss = np.float32(0)
    for batch in range(0,Ntrain,batchsize):
        for _ in range(n_dis):
            t_dis = np.zeros((batchsize, dims), dtype = np.float32)
            image_box = []
            for j in range(batchsize):
                rnd = np.random.randint(Ntrain)
                image_name = "b_face_getchu_" + str(rnd) + ".png"
                image = prepare_dataset(image_path + image_name, size)
                image_box.append(image)
                t_dis[j,:] = x_label[rnd]

            x_dis = np.array(image_box).astype(np.float32)

            x_real = cuda.to_gpu(x_dis)
            t_dis = cuda.to_gpu(t_dis)

            z = Variable(xp.random.normal(size=(batchsize,128),dtype=xp.float32))
            label = cuda.to_gpu(get_fake_tag_batch(batchsize, dims, threshold))
            label[label < 0.0] = 0.0
            z = F.concat([z, Variable(label)])
            x_fake = generator(z)
            label = Variable(label)
            y_fake = discriminator(x_fake,label)

            x_fake.unchain_backward()

            std_data = xp.std(x_real, axis=0, keepdims = True)
            rnd_x = xp.random.uniform(0,1,x_dis.shape).astype(xp.float32)
            x_perturbed = Variable(x_real + 0.5*rnd_x*std_data)

            x_real = Variable(x_real)
            t_label = t_dis
            t_label[t_label < 0.0] = 0.0
            t_label = Variable(t_label)
            y_dis = discriminator(x_real,t_label)

            y_perturbed  = discriminator(x_perturbed,t_label)
            grad, = chainer.grad([y_perturbed],[x_perturbed], enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_grad = lambda1 * F.mean_squared_error(grad, xp.ones_like(grad.data))

            loss_dis = loss_hinge_dis(y_fake, y_dis)+lambda1*loss_grad
            
            discriminator.cleargrads()
            loss_dis.backward()
            dis_opt.update()
            loss_dis.unchain_backward()
            
        z = Variable(xp.random.normal(size=(batchsize,128),dtype=xp.float32))
        label = cuda.to_gpu(get_fake_tag_batch(batchsize, dims, threshold))
        label[label < 0.0] = 0.0
        z = F.concat([z, Variable(label)])
        label = Variable(label)

        x_fake = generator(z)
        y_fake = discriminator(x_fake,label)

        loss_gen = loss_hinge_gen(y_fake)
        
        generator.cleargrads()
        loss_gen.backward()
        gen_opt.update()
        loss_gen.unchain_backward()

        sum_dis_loss += loss_dis.data.get()
        sum_gen_loss += loss_gen.data.get()

        if epoch%interval==0 and batch ==0:
            serializers.save_npz('discriminator_getchu.model',discriminator)
            serializers.save_npz('generator_9.model_getchu',generator)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config('train',False):
                x = generator(zenter)
            x.unchain_backward()
            x = x.data.get()
            for i_ in range(testsize):
                tmp = (np.clip((x[i_,:,:,:])*127.5 + 127.5, 0, 255)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(wid,wid,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%svisualize_%d.png'%(image_dir, epoch))

        if epoch%10==0 and batch==0:
            slack('%svisualize_%d.png'%(image_dir, epoch))

    print('epoch : {} dis_loss : {} gen_loss : {}'.format(epoch,sum_dis_loss/Ntrain,sum_gen_loss/Ntrain))