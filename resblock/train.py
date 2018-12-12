import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, cuda, serializers, Variable, initializers, Chain
import os
import cv2 as cv
import matplotlib
matplotlib.use('Agg')
import pylab
from model import Discriminator, Generator
import argparse
import math

xp = cuda.cupy
cuda.get_device(0).use()

def prepare_dataset(filename, size = 128):
    if filename.endswith(".png"):
        image_orig = cv.imread(filename)
        image_orig = cv.resize(image_orig,(size,size),interpolation=cv.INTER_CUBIC)
        hflip = np.random.choice([True,False])
        if hflip:
            image_orig = image_orig[::-1,:,:]
        image = image_orig[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5
        
        return image

def set_optimizer(model, alpha, beta1, beta2):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2 = beta2)
    optimizer.setup(model)
    return optimizer

def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))

def get_fake_tag(dims, threshold):
    prob2 = np.random.rand(9)
    tags = np.zeros((dims)).astype("f")
    tags[:] = 0
    tags[np.argmax(prob2)]=1.0

    return tags

def get_fake_tag_batch(batchsize, dims, threshold):
    tags = xp.zeros((batchsize, dims)).astype("f")
    for i in range(batchsize):
        tags[i] = xp.asarray(get_fake_tag(dims, threshold))
        
    return tags

parser = argparse.ArgumentParser(description = "DCGAN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 100, type = int, help = "batch size")
parser.add_argument("--interval", default = 1, type = int, help = "the interval of snapshot")
parser.add_argument("--lam1", default = 0.5, type = float, help = "the weight of the gradient penalty")
parser.add_argument("--n_dis",default=2,type=int,help="the number of discriminator update")
parser.add_argument("--thre",default=0.75,type=float,help="threshold")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
lambda1 = args.lam1
n_dis = args.n_dis
threshold=args.thre
wid = int(math.sqrt(batchsize))

image_out_dir = './output_vertical'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

model_dir = './model_vertical/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

#x_train = np.load('../DCGAN/face_getchu.npy').astype(np.float32)
#print(x_train.shape)
#Ntrain, channels, width, height = x_train.shape
image_path = "./face_getchu/"
Ntrain = 16000
x_label=np.load("./face_tag.npy").astype(np.float32)
_, dims=x_label.shape

gen_model = Generator()
dis_model = Discriminator()

gen_model.to_gpu()
dis_model.to_gpu()

#serializers.load_npz("./generator_165.model",gen_model)
#serializers.load_npz("./discriminator_165.model",dis_model)

gen_opt = set_optimizer(gen_model, 0.0002, 0.5, 0.99)
dis_opt = set_optimizer(dis_model, 0.0002, 0.5, 0.99)

zvis = xp.random.uniform(-1,1,(25,128)).astype(xp.float32)
ztag = cuda.to_gpu(get_fake_tag_batch(25, dims, threshold))
zenter = F.concat([Variable(zvis), Variable(ztag)])

for epoch in range(epochs):
    sum_dis_loss = np.float32(0)
    sum_gen_loss = np.float32(0)
    for batch in range(0,Ntrain,batchsize):
        #x_dis = np.zeros((batchsize,channels,width,height), dtype=np.float32)
        for _ in range(n_dis):
            image_box = []
            t_dis = np.zeros((batchsize, dims), dtype = np.float32)
            for j in range(batchsize):
                rnd = np.random.randint(Ntrain)
                image_name = "a_face_getchu_"+str(rnd)+".png"
                image = prepare_dataset(image_path + image_name)
                image_box.append(image)
                t_dis[j,:]=x_label[rnd]

            image_array = np.array(image_box).astype(np.float32)
            x_dis = cuda.to_gpu(image_array)
            t_dis = cuda.to_gpu(t_dis)
            t_dis[t_dis<0] = 0

            z = Variable(xp.random.uniform(-1,1,(batchsize,128)).astype(xp.float32))
            label = cuda.to_gpu(get_fake_tag_batch(batchsize, dims, threshold))
            z = F.concat([z, Variable(label)])
            x = gen_model(z)
            y = dis_model(x,Variable(label))
            dis_loss = F.mean(F.softplus(y))
            x.unchain_backward()

            std_data = xp.std(x_dis, axis=0, keepdims = True)
            rnd_x = xp.random.uniform(0,1,x_dis.shape).astype(xp.float32)
            x_perturbed = Variable(cuda.to_gpu(x_dis + 0.5*rnd_x*std_data))

            x_dis = Variable(cuda.to_gpu(x_dis))
            y_dis = dis_model(x_dis,Variable(t_dis))
            dis_loss += F.mean(F.softplus(-y_dis))

            y_perturbed = dis_model(x_perturbed,Variable(t_dis))
            grad, = chainer.grad([y_perturbed],[x_perturbed], enable_double_backprop=True)
            grad = F.sqrt(F.batch_l2_norm_squared(grad))
            loss_grad = lambda1 * F.mean_squared_error(grad, xp.ones_like(grad.data))
            dis_loss += loss_grad

            dis_model.cleargrads()
            dis_loss.backward()
            dis_loss.unchain_backward()
            dis_opt.update()

        z = Variable(xp.random.normal(size=(batchsize,128),dtype=xp.float32))
        label = cuda.to_gpu(get_fake_tag_batch(batchsize, dims, threshold))
        z = F.concat([z, Variable(label)])
        x = gen_model(z)
        y = dis_model(x,Variable(label))
        gen_loss = F.mean(F.softplus(-y))

        gen_model.cleargrads()
        gen_loss.backward()
        gen_loss.unchain_backward()
        gen_opt.update()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if epoch%interval==0 and batch ==0:
            serializers.save_npz(model_dir+'discriminator_{}.model'.format(epoch),dis_model)
            serializers.save_npz(model_dir+'generator_{}.model'.format(epoch),gen_model)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            z = zenter
            with chainer.using_config('train',False):
                x = gen_model(z)
            x = x.data.get()
            for i_ in range(25):
                tmp = np.clip(x[i_,:,:,:]*127.5 + 127.5, 0, 255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(5,5,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d.png'%(image_out_dir, epoch))

    print('epoch : {} dis_loss : {} gen_loss : {}'.format(epoch,sum_dis_loss/Ntrain,sum_gen_loss/Ntrain))
