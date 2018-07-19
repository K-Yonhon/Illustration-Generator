import chainer
import numpy as np
import chainer.links as L
import chainer.functions as F
from chainer import Variable, initializers, cuda, Chain
import math

xp = cuda.cupy

def feature_vector_normalize(x):
    alpha = 1.0 / F.sqrt(F.mean(x*x, axis = 1, keepdims = True) + 1e-8)
    y = F.broadcast_to(alpha, x.data.shape)*x
    return y

def minibatch_std(x):
    m = F.mean(x, axis = 0, keepdims = True)
    v = F.mean((x - F.broadcast_to(m, x.shape))*(x - F.broadcast_to(m, x.shape)), axis = 0, keepdims = True)
    std = F.mean(F.sqrt(v + 1.0e-8), keepdims = True)
    std = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
    y = F.concat([x,std], axis = 1)
    return y

class EqualizedConv2d(Chain):
    def  __init__(self, in_ch, out_ch, ksize, stride, pad):
        w = chainer.initializers.Normal(1.0)
        self.inv_c = np.sqrt(2.0 / (in_ch * ksize **2))
        super(EqualizedConv2d, self).__init__(
            conv = L.Convolution2D(in_ch, out_ch, ksize, stride, pad, initialW = w),
        )
    
    def __call__(self,x):
        return self.conv(self.inv_c * x)

class EqualizedLinear(Chain):
    def  __init__(self, in_ch, out_ch):
        w = chainer.initializers.Normal(1.0)
        self.inv_c = np.sqrt(2.0 / in_ch)
        super(EqualizedLinear, self).__init__(
            l = L.Linear(in_ch, out_ch, initialW = w),
        )
    
    def __call__(self,x):
         return self.l(self.inv_c * x)

class EqualizedDeconv2d(Chain):
    def  __init__(self, in_ch, out_ch, ksize, stride, pad):
        w = chainer.initializers.Normal(1.0)
        self.inv_c = np.sqrt(2.0 / in_ch)
        super(EqualizedDeconv2d, self).__init__(
            deconv = L.Deconvolution2D(in_ch, out_ch, ksize, stride, pad, initialW = w),
        )
    
    def __call__(self,x):
         return self.deconv(self.inv_c * x)

class GeneratorBlock(Chain):
    def __init__(self, in_ch, out_ch):
        super(GeneratorBlock, self).__init__(
            conv0 = EqualizedConv2d(in_ch, out_ch, 3,1,1),
            conv1 = EqualizedConv2d(out_ch, out_ch, 3,1,1),
        )
    
    def __call__(self,x):
        h = F.unpooling_2d(x,2,2,0,outsize = (x.shape[2]*2, x.shape[3]*2))
        h = F.leaky_relu(feature_vector_normalize(self.conv0(h)))
        h = F.leaky_relu(feature_vector_normalize(self.conv1(h)))
        return h
    
class Generator(Chain):
    def __init__(self, hidden = 256, ch = 256, ch_list = (256,256,256,256,128,64)):
        super(Generator, self).__init__()
        self.max_stage = (len(ch_list) - 1)*2
        self.hidden = hidden
        with self.init_scope():
            self.c0 = EqualizedConv2d(hidden, ch, 4,1,3)
            self.c1 = EqualizedConv2d(ch, ch, 3,1,1)
            bs = [chainer.Link()]
            outs = [EqualizedConv2d(ch_list[0], 3,1,1,0),]

            for i in range(1, len(ch_list)):
                bs.append(GeneratorBlock(ch_list[i-1], ch_list[i]))
                outs.append(EqualizedConv2d(ch_list[i],3,1,1,0))
            self.bs = chainer.ChainList(*bs)
            self.outs = chainer.ChainList(*outs)
    
    def __call__(self, z,stage):
        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = F.reshape(z, (len(z), self.hidden, 1, 1))
        h = F.leaky_relu(feature_vector_normalize(self.c0(h)))
        h = F.leaky_relu(feature_vector_normalize(self.c1(h)))

        for i in range(1, int(stage//2+1)):
            h = self.bs[i](h)

        if int(stage) % 2 == 0:
            out = self.outs[(int(stage//2))]
            x = out(h)

        else:
            out_prev = self.outs[stage//2]
            out_curr = self.outs[stage//2 + 1]
            b_curr = self.bs[stage//2 + 1]

            x_0 = out_prev(F.unpooling_2d(h,2,2,0,outsize=(2*h.shape[2], 2*h.shape[3])))
            x_1 = out_curr(b_curr(h))
            x = (1.0 - alpha) * x_0 + alpha * x_1

        return x

    def make_hidden(self, batchsize):
        z = xp.random.normal(size = (batchsize, self.hidden, 1,1)).astype(np.float32)
        z /= xp.sqrt(xp.sum(z*z, axis = 1, keepdims = True) / self.hidden + 1e-8)
        return z

class DiscriminatorBlock(Chain):
    def __init__(self, in_ch, out_ch, pooling_comp=1.0):
        super(DiscriminatorBlock,self).__init__(
            conv0 = EqualizedConv2d(in_ch, in_ch, 3,1,1,),
            conv1 = EqualizedConv2d(in_ch, out_ch, 3,1,1),
            )
    def __call__(self,x):
        h = F.leaky_relu(self.conv0(x))
        h = F.leaky_relu(self.conv1(h))
        h = 1.0 * F.average_pooling_2d(h,2,2,0)
        return h

class Discriminator(Chain):
    def __init__(self, ch = 256, pooling_comp = 1.0, ch_list = (256,256,256,256,128,64)):
        super(Discriminator, self).__init__()
        self.max_stage = (len(ch_list) - 1)*2
        self.pooling_comp = pooling_comp
        with self.init_scope():
            ins = [EqualizedConv2d(3, ch_list[0], 1,1,0)]
            bs = [chainer.Link()]
            
            for i in range(1, len(ch_list)):
                ins.append(EqualizedConv2d(3,ch_list[i], 1,1,0))
                bs.append(DiscriminatorBlock(ch_list[i], ch_list[i-1], pooling_comp))
            self.ins = chainer.ChainList(*ins)
            self.bs = chainer.ChainList(*bs)

            self.out0 = EqualizedConv2d(ch+1, ch, 3,1,1)
            self.out1 = EqualizedConv2d(ch, ch, 4,1,0)
            self.out2 = EqualizedLinear(ch,1)

    def __call__(self, x, stage):
        stage = min(stage, self.max_stage)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        if int(stage)%2 == 0:
            fromRGB = self.ins[stage//2]
            h = F.leaky_relu(fromRGB(x))
        else:
            fromRGB0 = self.ins[stage//2]
            fromRGB1 = self.ins[stage//2 + 1]
            b1 = self.bs[int(stage//2 + 1)]

            h0 = F.leaky_relu(fromRGB0(self.pooling_comp * F.average_pooling_2d(x,2,2,0)))
            h1 = b1(F.leaky_relu(fromRGB1(x)))
            h = (1 - alpha) * h0 + alpha * h1

        for i in range(int(stage//2),0,-1):
            h = self.bs[i](h)

        h = minibatch_std(h)
        h = F.leaky_relu((self.out0(h)))
        h = F.leaky_relu((self.out1(h)))
        return self.out2(h)