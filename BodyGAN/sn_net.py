import chainer
import chainer.functions as F
import chainer.links as L
from chainer import optimizers, cuda, serializers, Variable, initializers, Chain
from chainer.functions.connection import convolution_2d
from chainer.links.connection.convolution_2d import Convolution2D
from chainer.functions.connection import deconvolution_2d
from chainer.links.connection.deconvolution_2d import Deconvolution2D
from chainer.functions.connection import linear
from chainer.links.connection.linear import Linear
import numpy as np

xp = cuda.cupy

def _l2normalize(v, eps = 1e-12):
    return v / (((v**2).sum())**0.5 + eps)

def max_singular_value(W, u = None, Ip = 1):
    if u is None:
        u = xp.random.normal(size = (1, W.shape[0])).astype(xp.float32)
    _u = u

    for _ in range(Ip):
        _v = _l2normalize(xp.dot(_u, W.data),eps = 1e-12)
        _u = _l2normalize(xp.dot(_v, W.data.transpose()), eps = 1e-12)
    sigma = F.math.sum.sum(F.connection.linear.linear(_u, F.array.transpose.transpose(W))* _v)
    return sigma, _u, _v

class SNConvolution2D(Convolution2D):
    def __init__(self, in_channels, out_channels, ksize, stride = 1, pad = 0, nobias = True, initialW = None, initial_bias = None, use_gamma = False, Ip = 1):
         self.Ip = Ip
         self.u = None
         self.use_gamma = use_gamma
         super(SNConvolution2D,self).__init__(in_channels, out_channels, ksize, stride, pad, nobias, initialW, initial_bias)

    @property
    def W_bar(self):
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        sigma = F.array.broadcast.broadcast_to(sigma.reshape((1,1,1,1)), self.W.shape)
        self.u = _u
        return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNConvolution2D,self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1,1,1,1))

    def __call__(self,x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return convolution_2d.convolution_2d(x, self.W_bar, self.b, self.stride, self.pad)

class SNDeconvolution2D(Deconvolution2D):
    def __init__(self, in_channels, out_channels, ksize, stride = 1, pad = 0, nobias = True, initialW = None, initial_bias = None, use_gamma = False, Ip = 1):
         self.Ip = Ip
         self.u = None
         self.use_gamma = use_gamma
         super(SNDeconvolution2D,self).__init__(in_channels, out_channels, ksize, stride, pad, nobias, initialW, initial_bias)

    @property
    def W_bar(self):
        W_mat = self.W.reshape(self.W.shape[0], -1)
        sigma, _u, _ = max_singular_value(W_mat, self.u, self.Ip)
        sigma = F.array.broadcast.broadcast_to(sigma.reshape((1,1,1,1)), self.W.shape)
        self.u = _u
        return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNDeconvolution2D,self)._initialize_params(in_size)
        if self.use_gamma:
            W_mat = self.W.data.reshape(self.W.shape[0], -1)
            _, s, _ = np.linalg.svd(W_mat)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1,1,1,1))

    def __call__(self,x):
        if self.W.data is None:
            self._initialize_params(x.shape[1])
        return deconvolution_2d.deconvolution_2d(x, self.W_bar, self.b, self.stride, self.pad)

class SNLinear(Linear):
    def __init__(self, in_size, out_size, use_gamma = False, nobias = False, initialW = None, initial_bias = None, Ip = 1):
        self.Ip = Ip
        self.u = None
        self.use_gamma = use_gamma
        super(SNLinear, self).__init__(in_size, out_size, nobias, initialW, initial_bias)

    @property
    def W_bar(self):
        sigma, _u, _ = max_singular_value(self.W, self.u, self.Ip)
        sigma = F.array.broadcast.broadcast_to(sigma.reshape((1, 1)), self.W.shape)
        self.u = _u
        return self.W / sigma

    def _initialize_params(self, in_size):
        super(SNLinear, self)._initialize_params(in_size)
        if self.use_gamma:
            _, s, _ = np.linalg.svd(self.W.data)
            with self.init_scope():
                self.gamma = chainer.Parameter(s[0], (1, 1))

    def __call__(self,x):
        if self.W.data is None:
            self._initialize_params(x.size // x.shape[0])
        return linear.linear(x, self.W_bar, self.b)

class Generator(Chain):
    def __init__(self, n_hidden = 100, bottom_width = 4, ch = 512, wscale = 0.02):
        super(Generator, self).__init__()
        self.n_hidden = n_hidden
        self.ch = ch
        self.bottom_width = bottom_width
        
        with self.init_scope():
            w = chainer.initializers.Normal(wscale)
            self.l0 = L.Linear(self.n_hidden, bottom_width*bottom_width*ch, initialW = w)
            self.dc1 = L.Deconvolution2D(ch, ch//2, 4,2,1,initialW = w)
            self.dc2 = L.Deconvolution2D(ch//2, ch//4, 4,2,1,initialW = w)
            self.dc3 = L.Deconvolution2D(ch//4, ch//8, 4,2,1,initialW = w)
            self.dc4 = L.Deconvolution2D(ch//8, 3, 4,2,1,initialW = w)

            self.bn0 = L.BatchNormalization(bottom_width*bottom_width*ch)
            self.bn1 = L.BatchNormalization(ch//2)
            self.bn2 = L.BatchNormalization(ch//4)
            self.bn3 = L.BatchNormalization(ch//8)

    def __call__(self, z):
        h = F.reshape(F.relu(self.bn0(self.l0(z))),(len(z),self.ch,self.bottom_width,self.bottom_width))
        h = F.relu(self.bn1(self.dc1(h)))
        h = F.relu(self.bn2(self.dc2(h)))
        h = F.relu(self.bn3(self.dc3(h)))
        h = F.tanh(self.dc4(h))
        return h

class Discriminator(Chain):
    def __init__(self, bottom_width = 4, ch = 512, wscale = 0.02, output_dim = 1):
        w = initializers.Normal(wscale)
        super(Discriminator, self).__init__()
        with self.init_scope():
            self.c0_0 = SNConvolution2D(3, ch//8, 4,2,1,initialW = w)
            self.c0_1 = SNConvolution2D(ch//8, ch//4, 4,2,1,initialW = w)
            self.c1_1 = SNConvolution2D(ch//4, ch//2, 4,2,1,initialW = w)
            self.c2_1 = SNConvolution2D(ch//2, ch//1, 4,2,1,initialW = w)
            self.l4 = SNLinear(bottom_width * bottom_width * ch, output_dim, initialW = w)

    def __call__(self, x):
        h = F.leaky_relu(self.c0_0(x))
        h = F.leaky_relu(self.c0_1(h))
        h = F.leaky_relu(self.c1_1(h))
        h = F.leaky_relu(self.c2_1(h))
        return self.l4(h)