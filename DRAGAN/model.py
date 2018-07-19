import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda, Variable, initializers, serializers
import numpy as np

xp = cuda.cupy

class discriminator(Chain):
    def __init__(self, base = 64):
        w = initializers.Normal(0.02)
        super(discriminator,self).__init__()
        with self.init_scope():
                self.conv0 = L.Convolution2D(3,base,4,2,1,initialW=w)
                self.conv1 = L.Convolution2D(base,base * 2,4,2,1,initialW=w)
                self.conv2 = L.Convolution2D(base * 2,base * 4,4,2,1,initialW=w)
                self.conv3 = L.Convolution2D(base * 4,base * 8,4,2,1,initialW=w)
                self.l0 = L.Linear(4*4*base * 8,1)

    def __call__(self,x):
        h = F.leaky_relu(self.conv0(x))
        h = F.leaky_relu(((self.conv1(h))))
        h = F.leaky_relu(((self.conv2(h))))
        h = F.leaky_relu(((self.conv3(h))))
        h = self.l0(h)

        return h

class generator(Chain):
    def __init__(self, base = 64):
        w = initializers.Normal(0.02)
        super(generator,self).__init__(
            l0 = L.Linear(100,4*4*base * 8,initialW=w),
            dconv0 = L.Deconvolution2D(base * 8,base * 4,4,2,1,initialW=w),
            dconv1 = L.Deconvolution2D(base * 4,base * 2,4,2,1,initialW=w),
            dconv2 = L.Deconvolution2D(base * 2,base,4,2,1,initialW=w),
            dconv3 = L.Deconvolution2D(base,3,4,2,1,initialW=w),
        )
    
    def __call__(self,z):
        h = F.reshape(F.relu((self.l0(z))),(z.data.shape[0],512,4,4))
        h = F.relu((self.dconv0(h)))
        h = F.relu((self.dconv1(h)))
        h = F.relu((self.dconv2(h)))
        h = F.tanh(self.dconv3(h))

        return h