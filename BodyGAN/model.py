import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Variable, cuda, Chain, optimizers, initializers, serializers
import numpy as np
import math
import argparse
import os

xp=cuda.cupy
cuda.get_device(0).use()

def upsample(x, conv):
    h, w = x.shape[2:]
    x = F.unpooling_2d(x,2,2,0,cover_all=False)

    return conv(x)

class Gen_ResBlock(Chain):
    def __init__(self, in_ch, out_ch,  up = False, activation = F.relu):
        super(Gen_ResBlock, self).__init__()
        self.activation = activation
        self.up = up
        self.learnable_sc = in_ch != out_ch or up
        w = initializers.GlorotUniform(math.sqrt(2))
        w_sc = initializers.GlorotUniform()

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
    def __init__(self, base = 32, activation = F.relu):
        super(Generator, self).__init__()

        self.activation = activation
        w = initializers.GlorotUniform()

        with self.init_scope():
            self.l0_body = L.Linear(128, 4*3*base*8, initialW=w)
            self.res1_body = Gen_ResBlock(base*16, base*4, activation=activation, up=True)
            self.res2_body = Gen_ResBlock(base*8, base*4, activation=activation, up=True)
            self.res3_body = Gen_ResBlock(base*8, base*4, activation=activation, up=True)
            self.res4_body = Gen_ResBlock(base*8, base*2, activation=activation, up=True)
            self.res5_body = Gen_ResBlock(base*4, base, activation=activation, up=True)
            self.bn0_body  = L.BatchNormalization(base*2)

            self.l0_seg = L.Linear(128, 4*3*base*8, initialW=w)
            self.res1_seg = Gen_ResBlock(base*8, base*4, activation=activation, up=True)
            self.res2_seg = Gen_ResBlock(base*4, base*4, activation=activation, up=True)
            self.res3_seg = Gen_ResBlock(base*4, base*4, activation=activation, up=True)
            self.res4_seg = Gen_ResBlock(base*4, base*2, activation=activation, up=True)
            self.res5_seg = Gen_ResBlock(base*2, base, activation=activation, up=True)
            self.bn0_seg  = L.BatchNormalization(base)

            self.c0_body = L.Convolution2D(base * 2, 3, 3,1,1,initialW=w)
            self.c0_seg = L.Convolution2D(base, 3,3,1,1,initialW=w)
        
    def __call__(self, z):
        b, _, = z.shape

        # Segmentation mask generation
        h_1 = F.reshape(self.l0_seg(z), (b, 32*8, 4, 3))
        h_2 = self.res1_seg(h_1)
        h_3 = self.res2_seg(h_2)
        h_4 = self.res3_seg(h_3)
        h_5 = self.res4_seg(h_4)
        h_6 = self.res5_seg(h_5)
        h_seg = self.activation(self.bn0_seg(h_6))
        h_seg = F.tanh(self.c0_seg(h_seg))

        # Illustration generation
        h = F.reshape(self.l0_body(z), (b, 32*8 , 4, 3))
        h = self.res1_body(F.concat([h, h_1]))
        h = self.res2_body(F.concat([h, h_2]))
        h = self.res3_body(F.concat([h, h_3]))
        h = self.res4_body(F.concat([h, h_4]))
        h = self.res5_body(F.concat([h, h_5]))
        h = self.activation(self.bn0_body(F.concat([h, h_6])))
        h = F.tanh(self.c0_body(h))

        return h, h_seg

def downsample(x):
    return F.average_pooling_2d(x, 3,2,1)

class Dis_ResBlock(Chain):
    def __init__(self, in_ch, out_ch, down = False, activation = F.relu):
        super(Dis_ResBlock, self).__init__()

        w = initializers.GlorotUniform(math.sqrt(2))
        w_sc = initializers.GlorotUniform()
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

            self.bn1=L.BatchNormalization(out_ch)
            self.bn2=L.BatchNormalization(out_ch)
            self.bn_sc=L.BatchNormalization(out_ch)

    def residual(self, x):
        h = x
        h = self.bn1(self.c1(h))
        h = self.activation(h)
        h = self.bn2(self.c2(h))
        h = downsample(h)
        return h

    def shortcut(self, x):
        return self.bn_sc(self.c_sc(downsample(x)))

    def __call__(self, x):
        return self.residual(x) + self.shortcut(x)

class Discriminator(Chain):
    def __init__(self, base=32, activation = F.leaky_relu):
        super(Discriminator, self).__init__()
        w = initializers.GlorotUniform()
        self.activation = activation
        with self.init_scope():
            self.c0_body = Dis_ResBlock(3, base, activation=activation, down=True)
            self.res0_body = Dis_ResBlock(base*2, base*2, activation=activation, down=True)
            self.res1_body = Dis_ResBlock(base*4, base*4, activation=activation, down=True)
            self.res2_body = Dis_ResBlock(base*8, base*4, activation=activation, down=True)
            self.res3_body = Dis_ResBlock(base*8, base*8, activation=activation, down=True)
            self.res4_body = Dis_ResBlock(base*16, base*16, activation=activation)
            self.l1_body = L.Linear(None, 1, initialW=w)

            self.c0_seg = Dis_ResBlock(3, base, activation=activation, down=True)
            self.res0_seg = Dis_ResBlock(base, base*2, activation=activation, down=True)
            self.res1_seg = Dis_ResBlock(base*2, base*4, activation=activation, down=True)
            self.res2_seg = Dis_ResBlock(base*4, base*4, activation=activation, down=True)
            self.res3_seg = Dis_ResBlock(base*4, base*8, activation=activation, down=True)
            self.res4_seg = Dis_ResBlock(base*8, base*8, activation=activation)
            self.l1_seg = L.Linear(None, 1, initialW=w)

    def __call__(self, x, seg):
        # Setmentation discriminator
        h_1 = self.c0_seg(seg)
        h_2 = self.res0_seg(h_1)
        h_3 = self.res1_seg(h_2)
        h_4 = self.res2_seg(h_3)
        h_5 = self.res3_seg(h_4)
        h_6 = self.res4_seg(h_5)

        # Illustration discriminator
        h = self.c0_body(x)
        h = self.res0_body(F.concat([h, h_1]))
        h = self.res1_body(F.concat([h, h_2]))
        h = self.res2_body(F.concat([h, h_3]))
        h = self.res3_body(F.concat([h, h_4]))
        h = self.res4_body(F.concat([h, h_5]))
        h = self.activation(F.concat([h, h_6]))
        output = self.l1_body(h)

        return output

class CBR(Chain):
    def __init__(self, in_ch, out_ch, down=False, up=False):
        w = initializers.Normal(0.02)
        self.down = down
        self.up = up
        super(CBR, self).__init__()
        with self.init_scope():
            self.cpara = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)
            self.cdown = L.Convolution2D(in_ch, out_ch, 4,2,1,initialW=w)

            self.bn = L.BatchNormalization(out_ch)
    
    def __call__(self, x):
        if self.down:
            h = F.relu(self.bn(self.cdown(x)))

        elif self.up:
            h = F.unpooling_2d(x,2,2,0,cover_all=False)
            h = F.relu(self.bn(self.cpara(h)))

        else:
            h = F.relu(self.bn(self.cpara(x)))

        return h

class KeyPointDetector(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(KeyPointDetector, self).__init__()
        with self.init_scope():
            self.c0 = CBR(3, base, down=True)
            self.c1 = CBR(base, base*2, down=True)
            self.c2 = CBR(base*2, base*4, down=True)
            self.c3 = CBR(base*4, base*8, down=True)
            self.c4 = CBR(base*16, base*4, up=True)
            self.c5 = CBR(base*8, base*2, up=True)
            self.c6 = CBR(base*4, base, up=True)
            self.c7 = CBR(base*2, base, up=True)
            self.c8 = L.Convolution2D(base, 3,3,1,1,initialW=w)

    def __call__(self, x):
        h1 = self.c0(x)
        h2 = self.c1(h)
        h3 = self.c2(h)
        h4 = self.c3(h)
        h = self.c4(F.concat([h4, h4]))
        h = self.c5(F.concat([h, h3]))
        h = self.c6(F.concat([h, h2]))
        h = self.c7(F.concat([h, h1]))
        h = self.c8(h)

        return h

class SimpleDiscriminator(Chain):
    def __init__(self, base=64, in_ch=3):
        w = initializers.Normal(0.02)
        super(SimpleDiscriminator, self).__init__()
        with self.init_scope():
            self.c0 = CBR(in_ch, base, down=True)
            self.c1 = CBR(base, base*2, down=True)
            self.c2 = CBR(base*2, base*4, down=True)
            self.c3 = CBR(base*4, base*8, down=True)
            self.c4 = CBR(base*8, base*16)
            self.l0 = L.Linear(None, 1)

    def __call__(self, x):
        h = self.c0(x)
        h = self.c1(h)
        h = self.c2(h)
        h = self.c3(h)
        h = self.c4(h)
        h = self.l0(h)
        
        return h