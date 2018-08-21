import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, cuda, Chain, initializers, serializers
import numpy as np
import math
from sn_net import SNConvolution2D, SNLinear

def upsample(x, conv):
    h, w = x.shape[2:]
    x = F.unpooling_2d(x, 2, outsize=(h*2, w*2))

    return conv(x)

class Gen_ResBlock(Chain):
    def __init__(self, in_ch, out_ch,  up = False, activation = F.relu):
        super(Gen_ResBlock, self).__init__()
        self.activation = activation
        self.up = up
        self.learnable_sc = in_ch != out_ch or up
        w = initializers.Normal(0.02)
        w_sc = initializers.Normal(0.02)

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1, initialW=w)

            self.bn0 = L.BatchNormalization(in_ch)
            self.bn1 = L.BatchNormalization(out_ch)

            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_ch, out_ch, 1,1,0, initialW=w_sc)

    def residual(self, x):
        h = self.activation(self.bn0(x))
        h = upsample(h, self.c0) if self.up else self.c0(h)
        h = self.activation(self.bn1(h))
        h = self.c1(h)

        return h
        
    def shortcut(self, x):
        if self.learnable_sc:
            x = upsample(x, self.c_sc) if self.up else self.c_sc
            return x

        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)

class Generator(Chain):
    def __init__(self, base = 64, activation = F.relu):
        super(Generator, self).__init__()

        self.activation = activation
        w = initializers.Normal(0.02)

        with self.init_scope():
            self.l0 = L.Linear(128, 8*8*base*16, initialW=w)
            #self.res0 = Gen_ResBlock(base*16, base*16, activation=activation, up=True)
            self.res1 = Gen_ResBlock(base*16, base*8, activation=activation, up=True)
            #self.res2 = Gen_ResBlock(base*8, base*8, activation=activation, up=True)
            self.res3 = Gen_ResBlock(base*8, base*4, activation=activation, up=True)
            self.res4 = Gen_ResBlock(base*4, base*2, activation=activation, up=True)
            self.res5= Gen_ResBlock(base*2, base, activation=activation, up=True)
            self.bn0 = L.BatchNormalization(base)
            self.c0 = L.Convolution2D(base, 3, 3,1,1,initialW=w)
        
    def __call__(self, x):
        b, _, = x.shape
        h = F.reshape(self.l0(x), (b, 64*16, 8, 8))
        #h = self.res0(h)
        h = self.res1(h)
        #h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.res5(h)
        h = self.activation(self.bn0(h))
        h = F.tanh(self.c0(h))

        return h

def downsample(x):
    return F.average_pooling_2d(x, 2)

class Dis_ResBlock(Chain):
    def __init__(self, in_ch, out_ch, down = False, activation = F.relu):
        super(Dis_ResBlock, self).__init__()

        w = initializers.Normal(0.02)
        w_sc = initializers.Normal(0.02)
        self.down = down
        self.activation = activation

        self.learnable_sc = (in_ch != out_ch) or down
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1,initialW = w)

            if self.learnable_sc:
                self.c_sc = L.Convolution2D(in_ch, out_ch, 1,1,0,initialW=w_sc)

    def residual(self, x):
        h = self.activation(x)
        h = self.activation(self.c0(h))
        h = self.c1(h)
        if self.down:
            h = downsample(h)

        return h

    def shortcut(self, x):
        if self.learnable_sc:
            h = self.c_sc(x)
            if self.down:
                return downsample(h)
            else:
                return h
        else:
            return x

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)

class OptimizedBlock(Chain):
    def __init__(self, in_ch, out_ch, activation=F.relu):
        super(OptimizedBlock, self).__init__()
        w = initializers.GlorotUniform(math.sqrt(2))
        w_sc = initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.c1 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW=w)
            self.c2 = L.Convolution2D(out_ch, out_ch, 3,1,1, initialW=w)
            self.c_sc = L.Convolution2D(in_ch, out_ch, 1,1,0, initialW=w_sc)

    def residual(self, x):
        h = x
        h = self.c1(h)
        h = self.activation(h)
        h = self.c2(h)
        h = downsample(h)
        return h

    def shortcut(self, x):
        return self.c_sc(downsample(x))

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)

class Discriminator(Chain):
    def __init__(self, base=16, activation = F.leaky_relu):
        super(Discriminator, self).__init__()
        w = initializers.Normal(0.02)
        self.activation = activation
        with self.init_scope():
            self.c0 = L.Convolution2D(3, base, 3,1,1,initialW = w)
            self.res0 = Dis_ResBlock(base, base*2, activation=activation, down=True)
            self.res1 = Dis_ResBlock(base*2, base*4, activation=activation, down=True)
            self.res2 = Dis_ResBlock(base*4, base*8, activation=activation, down=True)
            #self.res3 = Dis_ResBlock(base*8, base*8, activation=activation, down=True)
            self.res4 = Dis_ResBlock(base*8, base*16, activation=activation, down=True)
            #self.res5 = Dis_ResBlock(base*16, base*16, activation=activation, down=True)
            self.l0 = L.Linear(base*16*8*8, 1, initialW=w)

    def __call__(self, x):
        h = self.c0(x)
        h = self.res0(h)
        h = self.res1(h)
        h = self.res2(h)
        #h = self.res3(h)
        h = self.res4(h)
        #h = self.res5(h)
        h = self.activation(h)
        h = self.l0(h)

        return h