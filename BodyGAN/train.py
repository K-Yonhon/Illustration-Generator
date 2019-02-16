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
from model import Discriminator, Generator, KeyPointDetector, SimpleDiscriminator
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

parser = argparse.ArgumentParser(description = "DCGAN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 100, type = int, help = "batch size")
parser.add_argument("--interval", default = 1, type = int, help = "the interval of snapshot")
parser.add_argument("--lam1", default = 0.5, type = float, help = "the weight of the gradient penalty")
parser.add_argument("--n_dis",default=2,type=int,help="the number of discriminator update")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
lambda1 = args.lam1
n_dis = args.n_dis
wid = int(math.sqrt(batchsize))

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

model_dir = './model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

image_path = "./face_getchu/"
segmentation_path = "./body_segmentation"
Ntrain = 16000

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

segmentation_generator = Generator()
segmentation_generator.to_gpu()
seg_gen_opt = set_optimizer(segmentation_generator)

key_point_detector = KeyPointDetector()
key_point_detector.to_gpu()
key_opt = set_optimizer(key_point_detector)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

segmentation_discriminator = SimpleDiscriminator()
segmentation_discriminator.to_gpu()
seg_dis_opt = set_optimizer(segmentation_discriminator)

keypoint_discriminator = SimpleDiscriminator()
keypoint_discriminator.to_gpu()
key_dis_opt = set_optimizer(keypoint_discriminator)

zvis = xp.random.normal(size=batchsize, 128).astype(xp.float32)

for epoch in range(epochs):
    sum_dis_loss = np.float32(0)
    sum_gen_loss = np.float32(0)
    for batch in range(0,Ntrain,batchsize):
        for _ in range(n_dis):
            image_box = []
            seg_box = []
            for j in range(batchsize):
                rnd = np.random.randint(Ntrain)
                image_name = "a_face_getchu_"+str(rnd)+".png"
                image = prepare_dataset(image_path + image_name)
                segmentation_name = ''
                seg = prepare_dataset(segmentation_path + segmentation_name)
                image_box.append(image)
                seg_box.append(seg)

            t = chainer.as_variable(xp.array(image_box).astype(xp.float32))
            s = chainer.as_variable(xp.array(seg_box).astype(xp.float32))

            z = chainer.as_variable(xp.random.uniform(-1,1,(batchsize,128)).astype(xp.float32))
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
