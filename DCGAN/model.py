import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, Variable, initializers, serializers
import numpy as np

xp = cuda.cupy

def add_noise(h, sigma=0.2):
    xp = cuda.get_array_module(h.data)
    if chainer.config.train:
        return h + sigma * xp.random.randn(*h.shape)
    else:
        return h

class discriminator(Chain):
    def __init__(self, base = 64):
        w = initializers.Normal(0.02)
        super(discriminator,self).__init__(
            conv0 = L.Convolution2D(3,base,4,2,1,initialW = w),
            conv1 = L.Convolution2D(base,base*2,4,2,1,initialW = w),
            conv2 = L.Convolution2D(base*2,base*4,4,2,1,initialW = w),
            conv3 = L.Convolution2D(base*4,base*8,4,2,1,initialW = w),
            l0 = L.Linear(4*4*base*8,2),
            bn1 = L.BatchNormalization(base*2),
            bn2 = L.BatchNormalization(base*4),
            bn3 = L.BatchNormalization(base*8),
        )

    def __call__(self,x):
        h = F.leaky_relu(add_noise(self.conv0(x)))
        h = F.leaky_relu(add_noise(self.bn1(self.conv1(h))))
        h = F.leaky_relu(add_noise(self.bn2(self.conv2(h))))
        h = F.leaky_relu(add_noise(self.bn3(self.conv3(h))))
        h = self.l0(h)

        return h

class generator(Chain):
    def __init__(self, base = 64):
        w = initializers.Normal(0.02)
        super(generator,self).__init__(
            l0 = L.Linear(100,4*4*base * 8,initialW = w,),
            dconv0 = L.Deconvolution2D(base*8,base*4,4,2,1,initialW = w),
            dconv1 = L.Deconvolution2D(base*4,base*2,4,2,1,initialW = w),
            dconv2 = L.Deconvolution2D(base*2,base,4,2,1,initialW = w),
            dconv3 = L.Deconvolution2D(base,3,4,2,1,initialW = w),

            bn0 = L.BatchNormalization(4*4*base*8),
            bn1 = L.BatchNormalization(base*4),
            bn2 = L.BatchNormalization(base*2),
            bn3 = L.BatchNormalization(base),
        )
    
    def __call__(self,z):
        h = F.reshape(F.relu(self.bn0(self.l0(z))),(z.data.shape[0],512,4,4))
        h = F.relu(self.bn1(self.dconv0(h)))
        h = F.relu(self.bn2(self.dconv1(h)))
        h = F.relu(self.bn3(self.dconv2(h)))
        h = self.dconv3(h)

        return h