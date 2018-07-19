import numpy as np
import os
import math
import pylab
import chainer
import chainer.links as L
import chainer.functions as F
from chainer import cuda, Variable, initializers, serializers, Chain, optimizers
from pggan_net import Generator, Discriminator

xp = cuda.cupy
cuda.get_device(0).use()

image_out_dir = './output'
if not os.path.exists(image_out_dir):
    os.mkdir(image_out_dir)

def set_optimizer(model, alpha, beta):
    optimizer = optimizers.Adam(alpha = alpha, beta1 = beta)
    optimizer.setup(model)
    return optimizer

x_train = np.load('data.npy').astype(np.float32)[0:5000]

epochs = 1000
batchsize = 10
interval = 10
initial_stage = 0
stage_interval = 300000
lambda1 = 10.0
gamma = 750.0
resolution = 64
Ntrain = x_train.shape[0]

counter = math.ceil(initial_stage*stage_interval)

gen_model = Generator()
dis_model = Discriminator()

gen_model.to_gpu()
dis_model.to_gpu()

gen_opt = set_optimizer(gen_model, 0.001, 0.0)
dis_opt = set_optimizer(dis_model, 0.001, 0.0)

for epoch in range(epochs):
    sum_gen_loss = 0
    sum_dis_loss = 0
    for batch in range(0, Ntrain, batchsize):
        x_dis = np.zeros((batchsize,3,resolution,resolution), dtype=np.float32)
        for j in range(batchsize):
            rnd = np.random.randint(Ntrain)
            x_dis[j,:,:,:] = x_train[rnd]
        x_dis = Variable(cuda.to_gpu(x_dis))

        stage = counter / stage_interval

        if math.floor(stage)%2 == 0:
            reso = min(resolution, 4*2**(((math.floor(stage) + 1)//2)))
            scale = max(1, resolution//reso)
            if scale > 1:
                x_dis = F.average_pooling_2d(x_dis, scale, scale, 0)
        else:
            alpha = stage - math.floor(stage)
            reso_low = min(resolution, 4*2**(((math.floor(stage))//2)))
            reso_high = min(resolution, 4*2**(((math.floor(stage) + 1)//2)))
            scale_low = max(1, resolution//reso_low)
            scale_high = max(1, resolution//reso_high)
            if scale_low > 1:
                x_dis_low = F.average_pooling_2d(x_dis, scale_low, scale_low, 0)
                x_real_low = F.unpooling_2d(x_dis_low, 2,2,0, outsize = (reso_high, reso_high))
                x_real_high = F.average_pooling_2d(x_dis, scale_high, scale_high, 0)
                x_dis = (1 - alpha) * x_real_low + alpha * x_real_high
        y_dis = dis_model(x_dis, stage = stage)

        z = Variable(xp.asarray(gen_model.make_hidden(batchsize)))
        x_fake = gen_model(z, stage = stage)
        y_fake = dis_model(x_fake, stage = stage)

        x_fake.unchain_backward()

        eps = xp.random.uniform(0,1,size = batchsize).astype(xp.float32)[:,None,None,None]
        x_mid = eps * x_dis + (1.0 - eps) * x_fake

        x_mid_v = Variable(x_mid.data)
        y_mid = F.sum(dis_model(x_mid_v, stage = stage))
        grad,  = chainer.grad([y_mid], [x_mid_v], enable_double_backprop=True)
        grad = F.sqrt(F.sum(grad*grad, axis=(1,2,3)))
        loss_gp = lambda1 * F.mean_squared_error(grad, gamma*xp.ones_like(grad.data)) * (1.0 / gamma**2)

        loss_dis = F.sum(-y_dis) / batchsize
        loss_dis += F.sum(y_fake) / batchsize
        loss_dis += 0.001 * F.sum(y_dis**2) / batchsize
        loss_dis += loss_gp

        dis_model.cleargrads()
        loss_dis.backward()
        dis_opt.update()
        loss_dis.unchain_backward()

        z = Variable(xp.asarray(gen_model.make_hidden(batchsize)))
        x_fake = gen_model(z, stage = stage)
        y_fake = dis_model(x_fake, stage = stage)
        loss_gen = F.sum(-y_fake) / batchsize
        gen_model.cleargrads()
        loss_gen.backward()
        gen_opt.update()
        loss_gen.unchain_backward()

        sum_dis_loss += loss_dis.data.get()
        sum_gen_loss += loss_gen.data.get()

        if epoch%interval==0 and batch ==0:
            pylab.rcParams['figure.figsize'] = (16.0,16.0)
            pylab.clf()
            z = Variable(xp.asarray(gen_model.make_hidden(batchsize)))
            with chainer.using_config('train',False):
                x = gen_model(z,stage)
            x = x.data.get()
            for i_ in range(batchsize):
                tmp = (np.clip((x[i_,:,:,:]*127.5 + 127.5), 0.0, 255.0)).transpose(1,2,0).astype(np.uint8)
                pylab.subplot(2,5,i_+1)
                pylab.imshow(tmp)
                pylab.axis('off')
                pylab.savefig('%s/visualize_%d_stage%d.png'%(image_out_dir, epoch,stage))
        counter += batchsize
    print('epoch:{} stage:{} dis_loss:{} gen_loss:{}'.format(epoch, int(stage), sum_dis_loss/Ntrain, sum_gen_loss/Ntrain))

serializers.save_npz('discriminator.model',dis_model)
serializers.save_npz('generator.model',gen_model)