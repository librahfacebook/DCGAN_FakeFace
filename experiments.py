# -*- coding:utf-8 -*-
# @Time: 2020/5/4 21:50
# @Author: libra
# @Site: train the DCGAN network and test
# @File: experiments.py
# @Software: PyCharm

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torchvision.utils as vutils
from collections import OrderedDict
from dataloader import get_dataloader
from networks.generator import Generator
from networks.discriminator import Discriminator
from IPython.display import HTML

# set random seed
manualSeed = random.randint(1,10000)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

data_root = '/home/librah/workspace/video_dataset/img_align_celeba/'
num_epochs = 5
batch_size = 128
image_size = 64
nz = 100
ngpu = 4
learning_rate = 0.0002
beta1=0.5
beta2=0.999
device = torch.device("cuda:0" if torch.cuda.is_available() and ngpu > 0 else "cpu")


discriminator_weights = 'weights/discriminator.pth'
generator_weights = 'weights/generator.pth'

g_losses_np = "data/g_losses.npy"
d_losses_np = "data/d_losses.npy"


def weights_init(model):
    """
    weights initialization called on Generator and Discriminator
    :param model: network
    """
    className = model.__class__.__name__
    if className.find('Conv') != -1:
        nn.init.normal_(model.weight.data,0.0,0.02)
    elif className.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data,1.0,0.02)
        nn.init.constant_(model.bias.data,0)

def show_losses_versus():
    g_losses = np.load(g_losses_np)
    d_losses = np.load(d_losses_np)

    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def load_weights():
    netG = Generator(nz=100, ngf=64).to(device)
    state_dict = torch.load(generator_weights)
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove 'module.'
        new_state_dict[name] = v
    netG.load_state_dict(new_state_dict)

    return netG

def generate_fake(fake_nums=1):
    """
    Generate fake images
    """

    fixed_noise = torch.randn(fake_nums, nz, 1, 1, device=device)
    netG = load_weights()

    with torch.no_grad():
        fake = netG(fixed_noise).detach().cpu()
    if fake_nums==1:
        fake_images = fake[0]
    else:
        fake_images = vutils.make_grid(fake, padding=2, normalize=True)

    fake_images = np.transpose(fake_images,(1,2,0))
    return fake_images

def show_fake_images(fake_nums=1):
    fake_images = generate_fake(fake_nums)
    plt.figure(figsize=(16,16))
    plt.axis('off')
    plt.title('Fake Images')
    plt.imshow(fake_images)
    plt.show()

def visualize_progression(img_list):
    fig = plt.figure(figsize=(8,8))
    plt.axis("off")
    imgs = [[plt.imshow(np.transpose(i,(1,2,0)),animated=True)] for i in img_list]
    ani = animation.ArtistAnimation(fig,imgs,interval=1000,repeat_delay=1000,blit=True)
    HTML(ani.to_jshtml())

def show_real_fake(img_list):
    dataloader = get_dataloader(data_root, image_size, batch_size)
    real_batch = next(iter(dataloader))
    plt.figure(figsize=(15,15))
    plt.subplot(1,2,1)
    plt.axis('off')
    plt.title('Real Images')
    plt.imshow(np.transpose(vutils.make_grid(real_batch.to(device)[:64],
                                             padding=5,normalize=True).cpu(),(1,2,0)))

    plt.subplot(1,2,2)
    plt.axis('off')
    plt.title('Fake Images')
    plt.imshow(np.transpose(img_list[-1],(1,2,0)))
    plt.show()

def train():
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    dataloader = get_dataloader(data_root, image_size, batch_size)

    # build the Generator and Discriminator network
    netG = Generator(nz=100,ngf=64).to(device)
    netD = Discriminator(ndf=64).to(device)
    if device.type == 'cuda' and ngpu>1:
        netG = nn.DataParallel(netG,list(range(ngpu)))
        netD = nn.DataParallel(netD,list(range(ngpu)))
    netG.apply(weights_init)
    netD.apply(weights_init)

    # setup the loss function
    criterion = nn.BCELoss()
    # create batch of latent vectors
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    real_label = 1
    fake_label = 0

    # setup Adam optimizers
    optimizerD = optim.Adam(netD.parameters(), lr = learning_rate, betas=(beta1,beta2))
    optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(beta1, beta2))

    for epoch in range(num_epochs):
        for i, data in enumerate(dataloader, 0):
            # update Discriminator network: maximize log(D(x)) + log(1-D(G(z)))

            # train with all-real batch
            netD.zero_grad()
            data_input = data.to(device)
            # print(data_input.size())
            label = torch.full((batch_size,),real_label,device=device)
            output = netD(data_input).view(-1)
            errD_real = criterion(output,label)
            errD_real.backward()
            D_x = output.mean().item()

            # train with all-fake batch
            noise = torch.randn(batch_size,nz,1,1,device=device)
            # generate fake image
            fake = netG(noise)
            label.fill_(fake_label)
            output = netD(fake.detach()).view(-1)
            errD_fake = criterion(output,label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            # Add the gradients from all-real and all-fake batches
            errD = errD_real+errD_fake
            optimizerD.step()

            # update Generator network: maximize log(D(G(z))
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake).view(-1)
            errG = criterion(output,label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizerG.step()

            if i%50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, num_epochs, i, len(dataloader),
                         errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                # print(errD_fake.item(),errD_real.item())
            G_losses.append(errG.item())
            D_losses.append(errD.item())

            # check the generator's effect
            if iters%500 == 0 or (epoch == num_epochs-1 and i == len(dataloader)-1):
                with torch.no_grad():
                    fake = netG(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

    # save weights of all networks
    torch.save(netD.state_dict(),discriminator_weights)
    torch.save(netG.state_dict(),generator_weights)

    # save losses
    np.save(d_losses_np,np.array(D_losses))
    np.save(g_losses_np,np.array(G_losses))

    print('Ok!!!')

if __name__ == '__main__':
    # train()
    # show_losses_versus()
    show_fake_images(fake_nums=1)