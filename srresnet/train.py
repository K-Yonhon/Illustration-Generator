import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Chain, Variable, serializers, optimizers]
import numpy as np
import os
import pylab
import argparse
from model import Generator, Discriminator]

xp = cuda.cupy

def set_optimizer(model, alpha=0.002, beta=0.5):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimizer.setup(model)

    return optimizer

parser = argparse.ArgumentParser(description="SRResNet")
parser.add_argument("--epoch", default=1000, type=int, help="the number of epochs")
parser.add_argument("--batchsize", default=100, type=int, help="batch size")
parser.add_argument("--interval", default=10, type=int, help="the interval of snapshot")
parser.add_argument("--lam1", default=10.0, type=float, help="the weight of gradient penalty")

args = parser.parse_args()
epochs = args.epoch
batchsize = args.batchsize
interval = args.interval
lambda1 = args.lam1

generator = Generator()
generator.to_gpu()
gen_opt = set_optimizer(generator)

discriminator = Discriminator()
discriminator.to_gpu()
dis_opt = set_optimizer(discriminator)