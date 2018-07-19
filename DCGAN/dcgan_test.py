import numpy as np
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import optimizers, cuda, serializers, Variable, initializers, Chain
import os
import cv2 as cv
import pylab
from model import discriminator, generator

xp = cuda.cupy
cuda.get_device(0).use()

image_vec_dir = './vector'

if not os.path.exists(image_vec_dir):
    os.mkdir(image_vec_dir)

def set_optimizer(model,alpha,beta):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(0.00001))
    return optimizer

def clip_img(x):
    return np.float32(-1 if x < -1 else (1 if x > 1 else x))

gen_model = generator()
dis_model = discriminator()

serializers.load_npz('generator.model',gen_model)
serializers.load_npz('discriminator.model',dis_model)

gen_model.to_gpu()
dis_model.to_gpu()

gen_opt = set_optimizer(gen_model, 1e-4, 0.5)
dis_opt = set_optimizer(dis_model, 1e-4, 0.5)

z1 = xp.random.uniform(-1,1,(1,100),dtype = np.float32)
z2 = xp.random.uniform(-1,1,(1,100),dtype = np.float32)
inter = (z2-z1)/20.0

for i in range(21):
    pylab.rcParams['figure.figsize'] = (1.0,1.0)
    pylab.clf()
    vec = z1 + i*inter
    z = Variable(vec)
    with chainer.using_config('train',False):
        x = gen_model(z)
    x = x.data.get()
    tmp = ((np.vectorize(clip_img)(x[0,:,:,:])+1)/2).transpose(1,2,0)
    pylab.imshow(tmp)
    pylab.axis('off')
    pylab.savefig('%s/morphing_%d.png'%(image_vec_dir, i))