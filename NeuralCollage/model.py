import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, initializers
from sCBN import SpatialCategoricalConditionalBatchNormalization

xp = cuda.cupy
cuda.get_device(0).use()


class ManifoldProjection(Chain):
    def __init__(self):
        w = initializers.Normal(0.02)
        super(ManifoldProjection, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(128, 1000, initialW=w)
            self.l1 = L.Linear(1000, 10000, initialW=w)
            self.l2 = L.Linear(10000, 1000, initialW=w)
            self.l3 = L.Linear(1000, 128, initialW=w)

    def __call__(self, x):
        h = F.relu(self.l0(x))
        h = F.prelu(self.l1(h))
        h = F.relu(self.l2(h))
        h = F.prelu(self.l3(h))

        return h


class ResBlock_Encoder(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(ResBlock_Encoder, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1,initialW=w)
            self.c_sc = L.Convolution2D(in_ch, out_ch, 1,1,0,initialW=w)

            self.bn0 = L.BatchNormalization(in_ch)
            self.bn1 = L.BatchNormalization(out_ch)

    def straight(self, x):
        h = F.relu(self.bn0(x))
        h = F.average_pooling_2d(h,3,2,1)
        h = self.c0(h)
        h = self.c1(F.relu(self.bn1(h)))

        return h

    def shortcut(self, x):
        h = F.average_pooling_2d(x,3,2,1)
        
        return self.c_sc(h)

    def __call__(self, x):
        return self.straight(x) + self.shortcut(x)


class ResBlock_Discriminator(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(ResBlock_Discriminator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)
            self.c1 = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)
            self.c_sc = L.Convolution2D(in_ch, out_ch, 1,1,0,initialW=w)

    def straight(self, x):
        h = self.c0(F.relu(x))
        h = self.c1(F.relu(h))
        h = F.average_pooling_2d(h,3,2,1)

        return h

    def shortcut(self, x):
        h = self.c_sc(x)

        return F.average_pooling_2d(h,3,2,1)

    def __call__(self, x):
        return self.straight(x) + self.shortcut(x)


class ResBlock_Generator(Chain):
    def __init__(self, in_ch, out_ch):
        w = initializers.Normal(0.02)
        super(ResBlock_Generator, self).__init__()
        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3,1,1,initialW=w)
            self.c1 = L.Convolution2D(out_ch, out_ch, 3,1,1,initialW=w)
            self.c_sc = L.Convolution2D(in_ch, out_ch, 1,1,0,initialW=w)

            self.scbn0 = SpatialCategoricalConditionalBatchNormalization(in_ch, n_cat=0)
            self.scbn1 = SpatialCategoricalConditionalBatchNormalization(out_ch, n_cat=0)

    def straight(self, x):
        h = F.relu(self.scbn0(x))
        h = F.unpooling_2d(h,2,2,0,cover_all=False)
        h = self.c0(h)
        h = self.c1(F.relu(self.scbn1(h)))

        return h

    def shortcut(self, x):
        h = F.unpooling_2d(x,2,2,0,cover_all=False)
        h = self.c_sc(h)

        return h

    def __call__(self, x):
        return self.straight(x) + self.shortcut(x)


class Generator(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(Generator, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(128, base*8*4*4)
            self.res0 = ResBlock_Generator(base*8, base*8)
            self.res1 = ResBlock_Generator(base*8, base*4)
            self.res2 = ResBlock_Generator(base*4, base*4)
            self.res3 = ResBlock_Generator(base*4, base*2)
            self.res4 = ResBlock_Generator(base*2, base)
            self.c0 = L.Convolution2D(base, 3,3,1,1,initialW=w)
            self.bn0 = L.BatchNormalization(base)

    def __call__(self, x):
        h = self.l0(x)
        h = self.res0(h)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = self.c0(F.relu(self.bn0(h)))

        return F.tanh(h)


class Discriminator(Chain):
    def __init__(self, base=64):
        w = initializers.Normal(0.02)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.res0 = ResBlock_Discriminator(3, base)
            self.res1 = ResBlock_Discriminator(base, base*2)
            self.res2 = ResBlock_Discriminator(base*2, base*4)
            self.res3 = ResBlock_Discriminator(base*4, base*4)
            self.res4 = ResBlock_Discriminator(base*4, base*8)
            self.l0 = L.Linear(None, 1, initialW=w)

    def __call__(self, x):
        h = self.res0(x)
        h = self.res1(h)
        h = self.res2(h)
        h = self.res3(h)
        h = self.res4(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, axis=2,3)
        h = self.l0(h)

        return h