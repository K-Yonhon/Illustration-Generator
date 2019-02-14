import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda,Chain,initializers
import numpy as np

xp = cuda.cupy
cuda.get_device(0).use()

def minibatch_std(x):
    m = F.mean(x, axis = 0, keepdims = True)
    v = F.mean((x - F.broadcast_to(m, x.shape))*(x - F.broadcast_to(m, x.shape)), axis = 0, keepdims = True)
    std = F.mean(F.sqrt(v + 1.0e-8), keepdims = True)
    std = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
    y = F.concat([x,std], axis = 1)
    
    return y

class MappingNetwork(Chain):
    def __init__(self, base=512, layers=8):
        w = initializers.GlorotUniform()
        super(MappingNetwork, self).__init__()
        mapping_net = chainer.ChainList()
        for _ in range(layers):
            mapping = L.Linear(base, base, initialW=w)
            mapping_net.add_link(mapping)

        with self.init_scope():
            self.mapping_net = mapping_net

    def __call__(self, x):
        for m in self.mapping_net.children():
            x = m(x)
            x = F.relu(x)

        return x

class AffineTransform(Chain):
    def __init__(self, ch_list = [512, 512, 512, 256, 256, 256, 256, 128, 128, 128, 128, 64, 64]):
        cbrs = chainer.ChainList()
        super(AffineTransform, self).__init__()
        for i in range(1, len(ch_list)):
            cbrs.add_link(CBR(ch_list[0], ch_list[i]))

        with self.init_scope():
            self.cbrs = cbrs

    def __call__(self, x, ref, stage):
        x = x.reshape(x.shape[0], x.shape[1], 1, 1)
        b, _, h, w = ref.shape
        x = F.broadcast_to(x, (b, 512, h, w))
        #for i, cbr in enumerate(self.cbrs.children()):
        #    if i == 0:
        #        x = F.unpooling_2d(x,2,2,0,cover_all=False)
        #        x = cbr(x)
        #    elif i % 2 == 1:
        #        x = F.unpooling_2d(x,2,2,0,cover_all=False)
        #        x = cbr(x)
        #    elif i % 2 == 0 and i != 0:
        #        x = cbr(x)

        #    if i == stage + 1:
        #        break

        x = self.cbrs[stage](x)

        return x

class ScaleFactors(Chain):
    def __init__(self, ch):
        #scales = chainer.ChainList()
        super(ScaleFactors, self).__init__()
            #scale = chainer.Parameter(initializer=np.zeros((1,1,1,1,), dtype='float32'))
            #scales.add_link(scale)

        with self.init_scope():
            self.scale = chainer.Parameter(initializer=np.zeros((1,ch,1,1), dtype='float32'))
            #self.scale = scales

    def __call__(self, x, stage):
        x = x * F.broadcast_to((self.scale), x.shape)

        return x

def calc_mean_std(feature, eps=1e-5):
    batch, channels, _, _ = feature.shape
    feature_a = feature.reshape(batch, channels, -1).data
    feature_var = xp.var(feature_a,axis = 2) + eps
    feature_var = chainer.as_variable(feature_var)
    feature_std = F.sqrt(feature_var).reshape(batch, channels, 1,1)
    feature_mean = F.mean(feature.reshape(batch, channels, -1), axis = 2)
    feature_mean = feature_mean.reshape(batch, channels, 1,1)
    
    return feature_std, feature_mean

def adain(content_feature, style_feature):
    shape = content_feature.shape
    style_std, style_mean = calc_mean_std(style_feature)
    style_mean = F.broadcast_to(style_mean, shape = shape)
    style_std = F.broadcast_to(style_std, shape = shape)
    
    content_std, content_mean = calc_mean_std(content_feature)
    content_mean = F.broadcast_to(content_mean, shape = shape)
    content_std = F.broadcast_to(content_std, shape = shape)
    normalized_feat = (content_feature - content_mean) / content_std
    
    return normalized_feat * style_std + style_mean

class CBR(Chain):
    def __init__(self, in_channels, out_channels, nobias=False, depthwise=False):
        w = initializers.GlorotUniform()
        self.depthwise = depthwise
        super(CBR, self).__init__()
        with self.init_scope():
            self.c = L.Convolution2D(in_channels, out_channels, 3,1,1,initialW=w, nobias=False)
            self.bn = L.BatchNormalization(out_channels)

            self.c_dw = L.Convolution2D(in_channels, out_channels, 1,1,0,initialW=w)
            self.bn_dw = L.BatchNormalization(out_channels)

    def __call__(self, x):
        if self.depthwise:
            h = F.leaky_relu(self.bn_dw(self.c_dw(x)))
        
        else:
            h = F.leaky_relu(self.bn(self.c(x)))

        return h

class InitialBlock(Chain):
    def __init__(self, in_channels, out_channels):
        w = initializers.GlorotUniform()
        super(InitialBlock, self).__init__()
        with self.init_scope():
            self.cbr0 = CBR(in_channels, out_channels)

            self.b0 = ScaleFactors(in_channels)
            self.b1 = ScaleFactors(out_channels)

            self.a0 = AffineTransform()
            self.a1 = AffineTransform()

    def __call__(self, x, w, noise):
        scale = self.b0(noise[0], stage=0) + x
        #scale = x
        affine = self.a0(w[0], x, stage=0)
        h = adain(scale, affine)
        h = self.cbr0(h)

        scale = self.b1(noise[1], stage=1) + h
        #scale = h
        affine = self.a1(w[1], x, stage=1)
        h = adain(scale, affine)

        return h

class Block(Chain):
    def __init__(self, in_channels, out_channels):
        w = initializers.GlorotUniform()
        super(Block, self).__init__()
        with self.init_scope():
            self.c_sc = L.Convolution2D(in_channels, out_channels,1,1,0,initialW=w)

            self.cbr0 = CBR(in_channels, out_channels)
            self.cbr1 = CBR(out_channels, out_channels)

            self.b0 = ScaleFactors(out_channels)
            self.b1 = ScaleFactors(out_channels)

            self.a0 = AffineTransform()
            self.a1 = AffineTransform()

    def shortcut(self, x):
        h = F.unpooling_2d(x,2,2,0,cover_all=False)
        h = self.c_sc(h)

        return h

    def __call__(self, x, stage, w, noise):
        h = F.unpooling_2d(x, 2,2,0,cover_all=False)
        h = self.cbr0(h)

        scale = self.b0(noise[2*stage], 2*stage) + h
        #scale = h
        affine = self.a0(w[2*stage], x, 2*stage)
        h = adain(scale, affine)
        h = self.cbr1(h)
        
        scale = self.b1(noise[2*stage+1], 2*stage + 1) + h
        #scale = h
        affine = self.a1(w[2*stage+1], x, 2*stage+1)
        h = adain(scale, affine)

        return h + self.shortcut(x)

class Generator(Chain):
    def __init__(self, base=32, ch_list = [512, 256, 256, 128, 128, 64]):
        w = initializers.GlorotUniform()
        super(Generator, self).__init__()
        blocks = chainer.ChainList()
        outs = chainer.ChainList()
        for i in range(1, len(ch_list)):
            blocks.add_link(Block(ch_list[i-1], ch_list[i]))
            outs.add_link(L.Convolution2D(ch_list[i], 3,3,1,1,initialW=w))
        with self.init_scope():
            self.b0 = InitialBlock(base*16, base*16)
            self.blocks = blocks
            self.outs = outs

    def __call__(self, x, w, stage, noise):
        h = self.b0(x, w, noise)
        for i, block in enumerate(self.blocks.children()):
            h = block(h, i + 1, w, noise)
            if i == stage - 1:
                break

        h = self.outs[stage - 1](h)
        h = F.tanh(h)

        return h

class CBR_Dis(Chain):
    def __init__(self, in_channels, out_channels, down=True):
        w = initializers.GlorotUniform()
        self.down = down
        super(CBR_Dis, self).__init__()
        with self.init_scope():
            self.cdown = L.Convolution2D(in_channels, out_channels, 4,2,1,initialW=w)
            self.cpara = L.Convolution2D(in_channels, out_channels,3,1,1,initialW=w)
            self.csec = L.Convolution2D(out_channels, out_channels,3,1,1,initialW=w)
            self.c_sc = L.Convolution2D(in_channels, out_channels, 1,1,0,nobias=True,initialW=w)

            #self.bnpara = L.BatchNormalization(out_channels)
            #self.bndown = L.BatchNormalization(out_channels)
            #self.b_sc = L.BatchNormalization(out_channels)

    def shortcut(self, x, down):
        #h = F.relu(self.b_sc(self.c_sc(x)))
        h = F.relu(self.c_sc(x))
        if down:
            h = F.average_pooling_2d(h,2,2,0)

        return h

    def __call__(self, x):
        if self.down:
            #h = F.relu(self.bndown(self.cdown(x)))
            h = F.relu(self.cdown(x))
            h = F.relu(self.csec(h))
            h = h + self.shortcut(x, down=True)

        else:
            #h = F.relu(self.bnpara(self.cpara(x)))
            h = F.relu(self.cpara(x))
            h = F.relu(self.csec(h))
            h = h + self.shortcut(x, down=False)

        return h

class Discriminator(Chain):
    def __init__(self, base=64, ch_list = [512, 512, 256, 256, 128, 64]):
        super(Discriminator, self).__init__()
        blocks = chainer.ChainList()
        ins = chainer.ChainList()
        w = initializers.GlorotUniform()
        for i in range(1, len(ch_list)):
            blocks.add_link(CBR_Dis(ch_list[i], ch_list[i-1]))
            ins.add_link(CBR_Dis(3, ch_list[i], down=False))

        with self.init_scope():
            self.blocks = blocks
            self.ins = ins
            self.outs = CBR_Dis(ch_list[0] + 1, ch_list[0])
            self.c5 = L.Linear(None, 1, initialW=w)

    def __call__(self, x, stage):
        h = self.ins[stage - 1](x)
        for i, block in enumerate(self.blocks.children()):
            h = self.blocks[stage - i - 1](h)
            if i ==  stage - 1:
                break

        h = minibatch_std(h)
        h = self.outs(h)
        h = self.c5(h)
        
        return h