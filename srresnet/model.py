import chainer
import chainer.links as L
import chainer.functions as F
from chainer import Chain, cuda, Variable, initializers
from sn_net import SNLinear, SNConvolution2D, SNDeconvolution2D
import numpy as np

xp = cuda.cupy

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
        w = chainer.initializers.GlorotUniform()

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW = w)

            self.bn0 = L.BatchNormalization(int(out_ch/4))

    def __call__(self,x):
        #h = F.unpooling_2d(x,2,2 )
        h = self.c0(x)
        h = pixel_shuffler(self.out_ch, h)

        if self.bn:
            h = self.bn0(h)

        if self.activation is not None:
            h = self.activation(h)

        return h

class SE(Chain):
    def __init__(self,in_ch,r=16):
        super(SE,self).__init__()
        w=initializers.GlorotUniform()
        with self.init_scope():
            self.l0=L.Linear(in_ch,int(in_ch/r),initialW=w)
            self.l1=L.Linear(int(in_ch/r),in_ch,initialW=w)

    def __call__(self,x):
        b,c,h,w=x.shape
        y=F.reshape(F.average_pooling_2d(x,(h,w)),(b,c))
        y=F.relu(self.l0(y))
        y=F.sigmoid(self.l1(y))

        return x*F.transpose(F.broadcast_to(y, (h, w, b, c)), (2, 3, 0, 1))

class Gen_SEResBlock(Chain):
    def __init__(self,in_ch,hid_ch):
        super(Gen_SEResBlock,self).__init__()
        w=initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch,hid_ch,3,1,1,initialW=w)
            self.c1 = L.Convolution2D(hid_ch, hid_ch, 3,1,1,initialW=w)

            self.bn0 = L.BatchNormalization(hid_ch)
            self.bn1 = L.BatchNormalization(hid_ch)
            self.se0 = SE(hid_ch,r=16)

    def __call__(self,x):
        h = F.relu(self.bn0(self.c0(x)))
        h = self.bn1(self.c1(h))
        h = self.se0(h)

        return h + x

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
        #h = F.relu(self.c0(x))
        #h = self.c1(h)

        return h + x

class Dis_ResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(Dis_ResBlock, self).__init__()
        w = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW = w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1, initialW = w)

            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)
    
    def __call__(self,x):
        h = F.leaky_relu(self.c0(x))
        h = self.c1(h)
        h = h + x
        h = F.leaky_relu(h)

        return h

class Dis_SEResBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(Dis_SEResBlock, self).__init__()
        w = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1, initialW = w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1, initialW = w)
            self.se0 = SE(out_ch,r=16)

            self.bn0 = L.BatchNormalization(out_ch)
            self.bn1 = L.BatchNormalization(out_ch)
    
    def __call__(self,x):
        h = F.leaky_relu(self.c0(x))
        h = self.c1(h)
        h = self.se0(h)
        h = h + x
        h = F.leaky_relu(h)

        return h

class Generator(Chain):
    def __init__(self, domain=9, base = 64):

        super(Generator, self).__init__()
        w = chainer.initializers.GlorotUniform()
        with self.init_scope():

            self.l0 = L.Linear(128 + domain, base*16*16, initialW = w)
            self.r0 = Gen_ResBlock(base, base)
            self.r1 = Gen_ResBlock(base, base)
            self.r2 = Gen_ResBlock(base, base)
            self.r3 = Gen_ResBlock(base, base)
            self.c0 = L.Convolution2D(base,base,3,1,1,initialW = w)
            self.cbr0 = CBR(base, base*4)
            self.cbr1 = CBR(base, base*4)
            self.cbr2 = CBR(base, base*4)
            self.c1 = L.Convolution2D(base, 3, 9, 1, 4, initialW = w)

            self.bn0 = L.BatchNormalization(base*16*16)
            self.bn1 = L.BatchNormalization(base)

    def __call__(self,x):
        b, _ = x.shape
        h1 = F.reshape(F.relu(self.bn0(self.l0(x))), (b, 64, 16,16))
        #h1 = F.reshape(F.relu(self.l0(x)), (b,64,16,16))
        h = self.r0(h1)
        h = self.r1(h)
        h = self.r2(h)
        h = self.r3(h)
        h = h1 + F.relu(self.bn1(self.c0(h)))
        #h = h1 + F.relu(self.c0(h))
        h = self.cbr0(h)
        h = self.cbr1(h)
        h = self.cbr2(h)
        h = F.tanh(self.c1(h))

        return h

class Discriminator_multi(Chain):
    def __init__(self,base = 32):
        super(Discriminator_multi, self).__init__()
        w = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(3,base,4,2,1,initialW = w)
            self.r0 = Dis_ResBlock(base,base)
            self.r1 = Dis_ResBlock(base,base)
            self.c1 = L.Convolution2D(base,base*2,4,2,1,initialW = w)
            self.r2 = Dis_ResBlock(base*2,base*2)
            self.r3 = Dis_ResBlock(base*2,base*2)
            self.c2 = L.Convolution2D(base*2,base*4,4,2,1,initialW = w)
            self.r4 = Dis_ResBlock(base*4,base*4)
            self.r5 = Dis_ResBlock(base*4,base*4)
            self.c3 = L.Convolution2D(base*4, base*8, 3,2,1,initialW = w)
            self.r6 = Dis_ResBlock(base*8, base*8)
            self.r7 = Dis_ResBlock(base*8, base*8)
            self.c4 = L.Convolution2D(base*8, base*16, 3,2,1,initialW = w)
            self.r8 = Dis_ResBlock(base*16, base*16)
            self.r9 = Dis_ResBlock(base*16, base*16)
            self.c5 = L.Convolution2D(base*16, base*32, 3,2,1,initialW = w)
            self.l0 = L.Linear(256*2*2,1, initialW = w)
            self.lembed = L.Linear(9, base*32, initialW = w)

            self.bn0 = L.BatchNormalization(base)
            self.bn1 = L.BatchNormalization(base*2)
            self.bn2 = L.BatchNormalization(base*4)
            self.bn3 = L.BatchNormalization(base*8)
            self.bn4 = L.BatchNormalization(base*16)
            self.bn5 = L.BatchNormalization(base*32)
    
    def __call__(self,x,category):
        b,c,w,h = x.shape
        x = x + Variable(xp.random.normal(loc=0.0, scale = 0.1, size =(b,c,w,h), dtype = xp.float32))
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
        h = self.r8(h)
        h = self.r9(h)
        h = F.leaky_relu(self.c5(h))
        h = F.sum(h, axis=(2,3))
        hout = self.l0(h)
        hembed = self.lembed(category)
        hout += F.sum(hembed * h, axis = 1, keepdims= True)

        return hout

class Discriminator_single(Chain):
    def __init__(self,base = 32):
        super(Discriminator_single, self).__init__()
        w = chainer.initializers.GlorotUniform()
        with self.init_scope():
            self.c0 = L.Convolution2D(3,base,4,2,1,initialW = w)
            self.r0 = Dis_SEResBlock(base,base)
            self.r1 = Dis_SEResBlock(base,base)
            self.c1 = L.Convolution2D(base,base*2,4,2,1,initialW = w)
            self.r2 = Dis_SEResBlock(base*2,base*2)
            self.r3 = Dis_SEResBlock(base*2,base*2)
            self.c2 = L.Convolution2D(base*2,base*4,4,2,1,initialW = w)
            self.r4 = Dis_SEResBlock(base*4,base*4)
            self.r5 = Dis_SEResBlock(base*4,base*4)
            self.c3 = L.Convolution2D(base*4, base*8, 3,2,1,initialW = w)
            self.r6 = Dis_SEResBlock(base*8, base*8)
            self.r7 = Dis_SEResBlock(base*8, base*8)
            self.c4 = L.Convolution2D(base*8, base*16, 3,2,1,initialW = w)
            self.r8 = Dis_SEResBlock(base*16, base*16)
            self.r9 = Dis_SEResBlock(base*16, base*16)
            self.c5 = L.Convolution2D(base*16, base*32, 3,2,1,initialW = w)
            self.l0 = L.Linear(256*4*4,1, initialW = w)

            self.bn0 = L.BatchNormalization(base)
            self.bn1 = L.BatchNormalization(base*2)
            self.bn2 = L.BatchNormalization(base*4)
            self.bn3 = L.BatchNormalization(base*8)
            self.bn4 = L.BatchNormalization(base*16)
            self.bn5 = L.BatchNormalization(base*32)
    
    def __call__(self,x):
        b,c,w,h = x.shape
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
        h = self.r8(h)
        h = self.r9(h)
        h = F.leaky_relu(self.c5(h))
        h = self.l0(h)

        return h