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
        image_orig = cv.resize(image_orig,(96,128),interpolation=cv.INTER_CUBIC)
        hflip = np.random.choice([True,False])
        if hflip:
            image_orig = image_orig[:,::-1,:]
        image = image_orig[:,:,::-1]
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5
        
        return image

def set_optimizer(model, alpha=0.0002, beta1=0.5, beta2=0.999):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2 = beta2)
    optimizer.setup(model)
    return optimizer

parser = argparse.ArgumentParser(description = "DCGAN")
parser.add_argument("--epoch", default = 1000, type = int, help = "the number of epochs")
parser.add_argument("--batchsize", default = 64, type = int, help = "batch size")
parser.add_argument("--interval", default = 1, type = int, help = "the interval of snapshot")
parser.add_argument("--lam1", default = 0.5, type = float, help = "the weight of the gradient penalty")
parser.add_argument("--n_dis",default=2,type=int,help="the number of discriminator update")
parser.add_argument('--iter', default=3000, type=int, help="the number of iterations")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
lambda1 = args.lam1
n_dis = args.n_dis
iterations = args.iter
wid = int(math.sqrt(batchsize))

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

model_dir = './model/'
if not os.path.exists(model_dir):
    os.mkdir(model_dir)

image_path = './Dataset/body_illustration/'
segmentation_path = "./Dataset/body_segmentation/"
dir_list = ['sum_0', 'sum_1', 'sum_2', 'sum_3']

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

#segmentation_generator = Generator()
#segmentation_generator.to_gpu()
#seg_gen_opt = set_optimizer(segmentation_generator)

#key_point_detector = KeyPointDetector()
#key_point_detector.to_gpu()
#key_opt = set_optimizer(key_point_detector)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)

#segmentation_discriminator = SimpleDiscriminator()
#segmentation_discriminator.to_gpu()
#seg_dis_opt = set_optimizer(segmentation_discriminator)

#keypoint_discriminator = SimpleDiscriminator()
#keypoint_discriminator.to_gpu()
#key_dis_opt = set_optimizer(keypoint_discriminator)

ztest = chainer.as_variable(xp.random.uniform(-1,1,(batchsize,128)).astype(xp.float32))

for epoch in range(epochs):
    sum_dis_loss = 0
    sum_gen_loss = 0
    for batch in range(0, iterations, batchsize):
        tag = np.random.choice(dir_list)
        seg_dir = segmentation_path + tag + '/'
        body_dir = image_path + tag + '/'
        sum_list = os.listdir(seg_dir)
        for _ in range(n_dis):
            image_box = []
            seg_box = []
            for j in range(batchsize):
                rnd = np.random.choice(sum_list)
                image_name = body_dir + rnd
                image = prepare_dataset(image_name)
                segmentation_name = seg_dir + rnd
                seg = prepare_dataset(segmentation_name)
                image_box.append(image)
                seg_box.append(seg)

            t = chainer.as_variable(xp.array(image_box).astype(xp.float32))
            s = chainer.as_variable(xp.array(seg_box).astype(xp.float32))

            z = chainer.as_variable(xp.random.uniform(-1,1,(batchsize,128)).astype(xp.float32))
            x, seg = generator(z)
            fake = discriminator(x, seg)
            real = discriminator(t,s)
            dis_loss = F.mean(F.softplus(-real)) + F.mean(F.softplus(fake))

            x.unchain_backward()
            seg.unchain_backward()

            #std_data = chainer.as_variable(xp.std(t.data, axis=0, keepdims = True))
            #rnd_x = chainer.as_variable(xp.random.uniform(0,1,t.shape).astype(xp.float32))
            #x_perturbed = t + 0.5 * rnd_x * std_data

            #y_perturbed = discriminator(x_perturbed, seg)
            #grad, = chainer.grad([y_perturbed],[x_perturbed], enable_double_backprop=True)
            #grad = F.sqrt(F.batch_l2_norm_squared(grad))
            #loss_grad = lambda1 * F.mean_squared_error(grad, xp.ones_like(grad.data))
            #dis_loss += loss_grad

            discriminator.cleargrads()
            dis_loss.backward()
            dis_loss.unchain_backward()
            dis_opt.update()

        z = chainer.as_variable(xp.random.uniform(-1, 1, (batchsize, 128)).astype(xp.float32))
        x, seg = generator(z)
        fake = discriminator(x, seg)
        gen_loss = F.mean(F.softplus(-fake))

        generator.cleargrads()
        gen_loss.backward()
        gen_loss.unchain_backward()
        gen_opt.update()

        sum_dis_loss += dis_loss.data.get()
        sum_gen_loss += gen_loss.data.get()

        if epoch%interval==0 and batch ==0:
            serializers.save_npz(model_dir+'discriminator.model',discriminator)
            serializers.save_npz(model_dir+'generator.model',generator)
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            with chainer.using_config('train',False):
                x, seg = generator(ztest)
            x = x.data.get()
            seg = seg.data.get()
            for i_ in range(batchsize):
                tmp = np.clip(x[i_,:,:,:]*127.5 + 127.5, 0, 255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(wid,wid,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_body_%d.png'%(image_out_dir, epoch))

            for i_ in range(batchsize):
                tmp = np.clip(seg[i_,:,:,:]*127.5 + 127.5, 0, 255).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(wid,wid,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_seg_%d.png'%(image_out_dir, epoch))

    print('epoch : {}'.format(epoch))
    print('Discriminator loss : {}'.format(sum_dis_loss / iterations))
    print('Generator loss : {}'.format(sum_gen_loss / iterations))