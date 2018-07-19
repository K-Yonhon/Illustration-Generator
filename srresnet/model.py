import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda, Variable
import numpy as np

class CBR(Chain):
    def __init__(self, in_ch, out_ch, bn = True, activation = F.relu):
        super.__init__(CBR, self)
        self.activation = activation
        w = chainer.initializers.Normal(0.02)
        with init_scope():
            self.c0 = L.Deconvolution2D(in_ch, out_ch, 3,1,1, initialW = w)

            self.bn0 = L.BatchNormalization(out_ch)

    def __call__(self,x):
        h = self.c0(x)
        if bn:
            h = self.bn0(h)
        if self.activation is not None:
            h = self.activation(h)

        return h

class Gen_ResBlock(Chain):
    def __init__(self, in_ch, hid_ch):
        super.__init__(Gen_ResBlock,self)
        w = chainer.initializers.Normal(0.02)
        with init_scope():
            self.c0 = L.Deconvolution2D(in_ch, hid_ch,3,1,1, initialW = w)
            self.c1 = L.Deconvolution2D(hid_ch, hid_ch,3,1,1, initialW = w)

            self.bn0 = L.BatchNormalization(hid_ch)
            self.bn1 = L.BatchNormalization(hid_ch)
    
    def __call__(self,x):
        h = F.relu(self.bn0(self.c0(x)))
        h = self.bn1(self.c1(h))

        return h + x

class Dis_ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super.__init__(Dis_ResBlock,self)
        w = chainer.initializers.Normal(0.02)
        with init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW = w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1, initialW = w)
    
    def __call__(self,x):
        h = F.leaky_relu(self.c0(x))
        h = self.c1(h)
        h = h + x
        h = F.leaky_relu(h)

        return h

class Generator(Chain):
    def __init__(self, base = 64):

        super.__init__(Generator, self)
        w = chainer.initializers.Normal(0.02)
        with init_scope():

            self.l0 = L.Linear(100, 64*16*116, initialW = w)
            self.r0 = Gen_ResBlock(base, base)
            self.r1 = Gen_ResBlock(base, base)
            self.r2 = Gen_ResBlock(base, base)
            self.c0 = L.Deconvolution2D(base,base,3,1,1,initialW = w)
            self.cbr0 = CBR(base, base*4)
            self.cbr1 = CBR(base*4, base*4)
            self.cbr2 = CBR(base*4, base*4)
            self.c1 = L.Deconvolution2D(base*4, 3, 9, 1, 1, initialW = w)

            self.bn0 = L.BatchNormalization(64*16*16)
            self.bn1 = L.BatchNormalization(base)

    def __call__(self,x):
        h1 = F.relu(self.bn0(self.l0(x)))
        h = self.r0(h1)
        h = self.r1(h)
        h = self.r2(h)
        h = h1 + F.relu(self.bn1(self.c0(h)))
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = F.tanh(self.c1(h))

        return h

class Discriminator(Chain):
    def __init__(self,base = 32):
        super.__init__(Discriminator, self)
        w = chainer.initializers.Normal(0.02)
        with init_scope():
            self.c0 = L.Convolution2D(3,base,4,2,1,initialW = w)
            self.r0 = Dis_ResBlock(base,base)
            self.r1 = Dis_ResBlock(base,base)
            self.c1 = L.Convolution2D(base,base*2,4,2,1,initialW = w)
            self.r2 = Dis_ResBlock(base*2,base*2)
            self.r3 = Dis_ResBlock(base*2,base*2)
            self.c2 = L.Convolution2D(base*2,base*4,4,2,1,initialW = w)
            self.r4 = Dis_ResBlock(base*4,base*4)
            self.r5 = Dis_ResBlock(base*4,base*4)
            self.c3 = L.Convolution2D(base*4, base*8, 4,2,1,initialW = w)
            self.r6 = Dis_ResBlock(base*8, base*8)
            self.r7 = Dis_ResBlock(base*8, base*8)
            self.c4 = L.Convolution2D(base*8, base*16, 4,2,1,initialW = w)
            self.l0 = L.Linear(base*16,1,initialW = w)
    
    def __call__(self,x):
        h = F.leaky_relu(self.c0(x))
        h = self.r0(h)
        h = self.r1(h)
        h = F.leaky_relu(self.c1(h))
        h = self.r2(h)
        h = self.r3(h)
        h = F.leaky_relu(self.c2(h))
        h = self.r4(h)
        h = self.r5(h)
        h = F.leaky_relu(self.c3(h))
        h = self.r5(h)
        h = self.r6(h)
        h = F.leaky_relu(self.c4(h))
        h = self.h0(h)

        return h
        
        