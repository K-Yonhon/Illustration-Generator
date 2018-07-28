import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda, Variable
from sn_net import SNLinear, SNConvolution2D
import numpy as np

def pixel_shuffler(out_ch, x, r = 2):
    b, c, w, h = x.shape
    x = F.reshape(x, (b, r, r, int(out_ch/(r*2)), w, h))
    x = F.transpose(x, (0,3,4,1,5,2))
    out_map = F.reshape(x, (b, int(out_ch/(r*2)), w*r, h*r))

    return out_map

class CBR(Chain):
    def __init__(self, in_ch, out_ch, bn = True, activation = F.relu):
        super(CBR,self).__init__()
        self.activation = activation
        self.bn = bn
        self.out_ch = out_ch
        w = chainer.initializers.Normal(0.02)

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW = w)

            self.bn0 = L.BatchNormalization(int(out_ch/4))

    def __call__(self,x):
        h = self.c0(x)
        h = pixel_shuffler(self.out_ch, h)

        if self.bn:
            h = self.bn0(h)

        if self.activation is not None:
            h = self.activation(h)

        return h

class Gen_ResBlock(Chain):
    def __init__(self, in_ch, hid_ch):
        super(Gen_ResBlock, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, hid_ch,3,1,1, initialW = w)
            self.c1 = L.Convolution2D(hid_ch, hid_ch,3,1,1, initialW = w)

            self.bn0 = L.BatchNormalization(hid_ch)
            self.bn1 = L.BatchNormalization(hid_ch)
    
    def __call__(self,x):
        h = F.relu(self.bn0(self.c0(x)))
        h = self.bn1(self.c1(h))

        return h + x

class Dis_ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(Dis_ResBlock, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = SNConvolution2D(in_ch, out_ch, 3,1,1, initialW = w)
            self.c1 = SNConvolution2D(out_ch, out_ch, 3,1,1, initialW = w)

            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)
    
    def __call__(self,x):
        h = F.leaky_relu(self.bn0(self.c0(x)))
        h = self.bn1(self.c1(h))
        h = h + x
        h = F.leaky_relu(h)

        return h

class Generator(Chain):
    def __init__(self, base = 64):

        super(Generator, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():

            self.l0 = L.Linear(128+34, base*8*8, initialW = w)
            self.r0 = Gen_ResBlock(base, base)
            self.r1 = Gen_ResBlock(base, base)
            self.r2 = Gen_ResBlock(base, base)
            self.c0 = L.Convolution2D(base,base,3,1,1,initialW = w)
            self.cbr0 = CBR(base, base*4)
            self.cbr1 = CBR(base, base*4)
            self.cbr2 = CBR(base, base*4)
            self.c1 = L.Convolution2D(base, 3, 9, 1, 4, initialW = w)

            self.bn0 = L.BatchNormalization(64*8*8)
            self.bn1 = L.BatchNormalization(base)

    def __call__(self,x):
        b, _ = x.shape
        h1 = F.reshape(F.relu(self.bn0(self.l0(x))), (b, 64, 8,8))
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
        super(Discriminator, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.c0 = SNConvolution2D(3,base,4,2,1,initialW = w)
            self.r0 = Dis_ResBlock(base,base)
            self.r1 = Dis_ResBlock(base,base)
            self.c1 = SNConvolution2D(base,base*2,4,2,1,initialW = w)
            self.r2 = Dis_ResBlock(base*2,base*2)
            self.r3 = Dis_ResBlock(base*2,base*2)
            self.c2 = SNConvolution2D(base*2,base*4,4,2,1,initialW = w)
            self.r4 = Dis_ResBlock(base*4,base*4)
            self.r5 = Dis_ResBlock(base*4,base*4)
            self.c3 = SNConvolution2D(base*4, base*8, 3,2,1,initialW = w)
            self.r6 = Dis_ResBlock(base*8, base*8)
            self.r7 = Dis_ResBlock(base*8, base*8)
            self.c4 = SNConvolution2D(base*8, base*16, 3,2,1,initialW = w)
            #self.r8 = Dis_ResBlock(base*16, base*16)
            #elf.r9 = Dis_ResBlock(base*16, base*16)
            #self.c5 = L.Convolution2D(base*16, base*32, 3,2,1,initialW = w)
            self.l0 = SNLinear(512*2*2,1)
            self.lcls = SNLinear(512*2*2, 34, initialW = w)

            self.bn0 = L.BatchNormalization(base)
            self.bn1 = L.BatchNormalization(base*2)
            self.bn2 = L.BatchNormalization(base*4)
            self.bn3 = L.BatchNormalization(base*8)
            self.bn4 = L.BatchNormalization(base*16)
            self.bn5 = L.BatchNormalization(base*32)
    
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
        h = self.r6(h)
        h = self.r7(h)
        h = F.leaky_relu(self.c4(h))
        #h = self.r8(h)
        #h = self.r9(h)
        #h = F.leaky_relu(self.bn5(self.c5(h)))
        h_cls = self.lcls(h)
        h = self.l0(h)

        return h, h_cls