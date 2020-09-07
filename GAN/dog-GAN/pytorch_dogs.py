import shutil
import numpy as np
import os
import zipfile

import matplotlib.pyplot as plt
import random
import cv2
from tqdm import tqdm
import platform
import argparse

from scipy.stats import truncnorm
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter

from config import *
from LBscore import calc_score


parser = argparse.ArgumentParser()
parser.add_argument('sess', default='0',
                    type=str, help='session id')
parser.add_argument('--train', default='T',
                    choices=('T', 'F'), help='to train')
parser.add_argument('--eval', default='T',
                    choices=('T', 'F'), help='to eval')
args = parser.parse_args()

sess = 'output_' + args.sess
LOG_DIR = os.path.join('logs', sess)
print('log dir: ', LOG_DIR)

MODEL_PATH = os.path.join('logs', sess, 'models')
OUTPUT_PATH = os.path.join('logs', sess, 'images')
LOG_IMAGE_PATH = os.path.join('logs', sess, 'log_images')

PREPROCESS_HOME = os.path.join(DATA_HOME, 'preprocess')


class DogDataset(Dataset):
    def __init__(self, img_dir, custom_transforms=None):

        self.img_dir = img_dir

        real_data = np.load(img_dir)
        self.real_labels = real_data['labels']
        self.imgs = real_data['dogs']  # (H,W,C), RGB, 0-255
        # self.imgs = real_data['dogs'].astype(np.float32) / 255.0
        self.custom_transforms = custom_transforms

    def rotateImage(self, image, angle):
        center = tuple(np.array(image.shape[0:2])/2)
        rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, image.shape[0:2], flags=cv2.INTER_LINEAR)

    def __getitem__(self, index):
        img = self.imgs[index]

        if random.random() < 0.3:
            img = cv2.flip(img, 1)
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img = self.rotateImage(img, angle)
        # if random.random() < 0.5:
        #     img = img + np.random.randn(*img.shape)

        img = self.custom_transforms(img)
        # img = torch.as_tensor(img, dtype=torch.float32)
        # img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)

        return img

    def __len__(self):
        return len(self.imgs)


custom_transforms = transforms.Compose([transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_dataset = DogDataset(img_dir=os.path.join(
    PREPROCESS_HOME, 'dogs.npz'), custom_transforms=custom_transforms)

print('train_dataset size: ', len(train_dataset))


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          #   drop_last=True,
                          num_workers=NUM_WORKERS)

# ----------------------------------------------------------------------------
# Pixelwise feature vector normalization.
# reference: https://github.com/tkarras/progressive_growing_of_gans/blob/master/networks.py#L120
# ----------------------------------------------------------------------------


class PixelwiseNorm(nn.Module):
    def __init__(self):
        super(PixelwiseNorm, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the module
        :param x: input activations volume
        :param alpha: small number for numerical stability
        :return: y => pixel normalized activations
        """
        y = x.pow(2.).mean(dim=1, keepdim=True).add(alpha).sqrt()  # [N1HW]
        y = x / y  # normalize the input x volume
        return y


class MinibatchStdDev(nn.Module):
    """
    Minibatch standard deviation layer for the discriminator
    """

    def __init__(self):
        """
        derived class constructor
        """
        super(MinibatchStdDev, self).__init__()

    def forward(self, x, alpha=1e-8):
        """
        forward pass of the layer
        :param x: input activation volume
        :param alpha: small number for numerical stability
        :return: y => x appended with standard deviation constant map
        """
        batch_size, _, height, width = x.shape
        # [B x C x H x W] Subtract mean over batch.
        y = x - x.mean(dim=0, keepdim=True)
        # [1 x C x H x W]  Calc standard deviation over batch
        y = torch.sqrt(y.pow(2.).mean(dim=0, keepdim=False) + alpha)

        # [1]  Take average over feature_maps and pixels.
        y = y.mean().view(1, 1, 1, 1)

        # [B x 1 x H x W]  Replicate over group and pixels.
        y = y.repeat(batch_size, 1, height, width)

        # [B x C x H x W]  Append as new feature_map.
        y = torch.cat([x, y], 1)
        # return the computed values:
        return y


class Generator(nn.Module):
    def __init__(self, nz, nfeats, nchannels):
        super(Generator, self).__init__()

        # input is Z, going into a convolution
        self.conv1 = spectral_norm(nn.ConvTranspose2d(
            nz, nfeats * 8, 4, 1, 0, bias=False))
        #self.bn1 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 4 x 4

        self.conv2 = spectral_norm(nn.ConvTranspose2d(
            nfeats * 8, nfeats * 8, 4, 2, 1, bias=False))
        #self.bn2 = nn.BatchNorm2d(nfeats * 8)
        # state size. (nfeats*8) x 8 x 8

        self.conv3 = spectral_norm(nn.ConvTranspose2d(
            nfeats * 8, nfeats * 4, 4, 2, 1, bias=False))
        #self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 16 x 16

        self.conv4 = spectral_norm(nn.ConvTranspose2d(
            nfeats * 4, nfeats * 2, 4, 2, 1, bias=False))
        #self.bn4 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats * 2) x 32 x 32

        self.conv5 = spectral_norm(nn.ConvTranspose2d(
            nfeats * 2, nfeats, 4, 2, 1, bias=False))
        #self.bn5 = nn.BatchNorm2d(nfeats)
        # state size. (nfeats) x 64 x 64

        self.conv6 = spectral_norm(nn.ConvTranspose2d(
            nfeats, nchannels, 3, 1, 1, bias=False))
        # state size. (nchannels) x 64 x 64
        self.pixnorm = PixelwiseNorm()

    def forward(self, x):
        #x = F.leaky_relu(self.bn1(self.conv1(x)))
        #x = F.leaky_relu(self.bn2(self.conv2(x)))
        #x = F.leaky_relu(self.bn3(self.conv3(x)))
        #x = F.leaky_relu(self.bn4(self.conv4(x)))
        #x = F.leaky_relu(self.bn5(self.conv5(x)))
        x = F.leaky_relu(self.conv1(x))
        x = F.leaky_relu(self.conv2(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv3(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv4(x))
        x = self.pixnorm(x)
        x = F.leaky_relu(self.conv5(x))
        x = self.pixnorm(x)
        x = torch.tanh(self.conv6(x))

        return x


class Discriminator(nn.Module):
    def __init__(self, nchannels, nfeats):
        super(Discriminator, self).__init__()

        # input is (nchannels) x 64 x 64
        self.conv1 = nn.Conv2d(nchannels, nfeats, 4, 2, 1, bias=False)
        # state size. (nfeats) x 32 x 32

        self.conv2 = spectral_norm(
            nn.Conv2d(nfeats, nfeats * 2, 4, 2, 1, bias=False))
        self.bn2 = nn.BatchNorm2d(nfeats * 2)
        # state size. (nfeats*2) x 16 x 16

        self.conv3 = spectral_norm(
            nn.Conv2d(nfeats * 2, nfeats * 4, 4, 2, 1, bias=False))
        self.bn3 = nn.BatchNorm2d(nfeats * 4)
        # state size. (nfeats*4) x 8 x 8

        self.conv4 = spectral_norm(
            nn.Conv2d(nfeats * 4, nfeats * 8, 4, 2, 1, bias=False))
        self.bn4 = nn.MaxPool2d(2)
        # state size. (nfeats*8) x 4 x 4
        self.batch_discriminator = MinibatchStdDev()
        self.pixnorm = PixelwiseNorm()
        self.conv5 = spectral_norm(
            nn.Conv2d(nfeats * 8 + 1, 1, 2, 1, 0, bias=False))
        # self.dropout = nn.Dropout(0.3)
        # state size. 1 x 1 x 1

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        # x = self.dropout(x)
       # x = self.pixnorm(x)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        # x = self.dropout(x)
       # x = self.pixnorm(x)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        # x = self.dropout(x)
       # x = self.pixnorm(x)
        x = self.batch_discriminator(x)
        x = torch.sigmoid(self.conv5(x))
        #x= self.conv5(x)
        return x.view(-1, 1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
netG = Generator(noise_dim, 32, 3).to(device)
# print('netG: ')
# print(netG)
netD = Discriminator(3, 48).to(device)
# print('netD: ')
# print(netD)


def weights_init(m):
    """
    Takes as input a neural network m that will initialize all its weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def generate_and_save_images(epoch, writer, gene_number=16):
    images_tensor, images = generate_images(netG, noise_dim, device, 16)

    fig = plt.figure(figsize=(8, 8))
    nrows = int(np.ceil(images.shape[0] / 4))
    for i in range(images.shape[0]):
        img = images[i, :, :, :]
        plt.subplot(nrows, 4, i+1)
        plt.imshow(img)
        plt.axis('off')

    plt.savefig(os.path.join(LOG_IMAGE_PATH,
                             'epoch_{:05d}.png'.format(epoch)))
    plt.close()

    for i in range(4):
        writer.add_image('generated_images' + str(i), images_tensor[i], epoch)


def generate_images(model, noise_dim, device, gene_number=16):
    # noise = torch.randn(gene_number, noise_dim, 1, 1, device=device)
    noise = truncnorm.rvs(-1, 1, size=(gene_number, noise_dim, 1, 1))
    noise = torch.from_numpy(noise).float().to(device)

    images_tensor = model(noise).to("cpu").clone().detach()
    images_tensor = (images_tensor + 1)/2.0

    images_numpy = images_tensor.numpy().transpose((0, 2, 3, 1))

    # images = (images * 255).astype(np.uint8)
    # images = np.clip(images, 0, 255)
    return images_tensor, images_numpy


def smooth_labels(b_size):
    # class=1 to [0.7, 1.2]
    labels = torch.full((b_size, 1), 1)
    labels = labels - 0.3 + torch.rand_like(labels) * 0.5
    return labels


def noise_labels(labels, p):
    n_select = int(p * len(labels))
    flip_index = np.random.choice([i for i in range(len(labels))], size=n_select)
    labels[flip_index] = 1 - labels[flip_index]
    labels[labels < 0 ] = 0
    return labels


# netG = netG.apply(weights_init)
# netD = netD.apply(weights_init)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=0.0002,
                        betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002,
                        betas=(0.5, 0.999))

update_steps = int(np.ceil(EPOCHS/200))
lr_schedulerD = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerD,
                                                                     T_0=update_steps, eta_min=0.00005)
lr_schedulerG = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizerG,
                                                                     T_0=update_steps, eta_min=0.00005)



def train():
    remakedirs(MODEL_PATH)
    remakedirs(OUTPUT_PATH)
    remakedirs(LOG_IMAGE_PATH)

    writer = SummaryWriter(LOG_DIR)
    step = 1
    writer_step = 1
    loss_Ds = []
    loss_Gs = []
    D_xs = []
    D_G_z1s = []
    D_G_z2s = []
    # tbar = tqdm(total= (len(train_loader) * EPOCHS))
    for epoch in tqdm(range(EPOCHS)):
        for i, real_images in enumerate(train_loader):
            ############################
            # (1) Update D network
            ###########################
            netD.zero_grad()
            # add noise to inputs
            real_images = real_images + torch.randn_like(real_images) * (1/(100*(epoch+1)))
            real_images = real_images.to(device)
            b_size = real_images.shape[0]

            # Use Soft and Noisy Labels
            labels = smooth_labels(b_size)
            labels = noise_labels(labels, 0.1 / (1+epoch)).to(device)

            D_out_real = netD(real_images)
            loss_D_real = criterion(D_out_real, labels)
            loss_D_real.backward()
            D_x = D_out_real.mean().item()

            noise = torch.randn(b_size, noise_dim, 1, 1, device=device)
            fake = netG(noise)
            labels.fill_(0) + np.random.uniform(0, 0.3)

            D_out_fake = netD(fake.detach())
            loss_D_fake = criterion(D_out_fake, labels)
            loss_D_fake.backward()

            D_G_z1 = D_out_fake.mean().item()
            loss_D = loss_D_real + loss_D_fake

            optimizerD.step()

            # loss_D = (torch.mean((D_out_real - torch.mean(D_out_fake) - labels) ** 2) +
            #           torch.mean((D_out_fake - torch.mean(D_out_real) + labels) ** 2))/2
            # loss_D.backward(retain_graph=True)
            # optimizerD.step()

            ############################
            # (2) Update G network
            ###########################
            netG.zero_grad()
            G_out_fake = netD(fake)
            # labels = smooth_labels(b_size)
            labels.fill_(1)

            loss_G = criterion(G_out_fake, labels)
            # loss_G = (torch.mean((D_out_real - torch.mean(G_out_fake) + labels) ** 2) +
            #           torch.mean((G_out_fake - torch.mean(D_out_real) - labels) ** 2))/2
            loss_G.backward()
            optimizerG.step()
            D_G_z2 = G_out_fake.mean().item()

            if step % 10 == 0:
                loss_Ds.append(loss_D.item())
                loss_Gs.append(loss_G.item())

                D_xs.append(D_x)
                D_G_z1s.append(D_G_z1)
                D_G_z2s.append(D_G_z2)

            if step % 200 == 0:
                # print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                #     % (epoch + 1, EPOCHS, i, len(train_loader),
                #         loss_D.item(), loss_G.item(), D_x, D_G_z1, D_G_z2))

                writer.add_scalars('loss', {
                    'D': np.mean(loss_Ds),
                    'G': np.mean(loss_Gs)
                }, writer_step)

                writer.add_scalars('output', {
                    'D_real_whenupdateD': np.mean(D_xs),
                    'D_G_fake_whenupdateD': np.mean(D_G_z1s),
                    'D_G_fake_whenupdateG': np.mean(D_G_z2s),
                }, writer_step)

                loss_Ds = []
                loss_Gs = []
                D_xs = []
                D_G_z1s = []
                D_G_z2s = []

                generate_and_save_images(writer_step, writer)
                writer_step += 1

            step += 1
            lr_schedulerG.step(epoch)
            lr_schedulerD.step(epoch)

    writer.close()

    torch.save(netD.state_dict(), os.path.join(
        MODEL_PATH, 'netD_state_dict.pth'))
    torch.save(netG.state_dict(), os.path.join(
        MODEL_PATH, 'netG_state_dict.pth'))

if args.train == 'T':
    train()


def save_zip(model, save_path=OUTPUT_PATH):
    im_batch_size = 100
    n_images = 10000
    # with zipfile.ZipFile(osp.join(LOG_DIR, 'images.zip')) as myZip:
    for i_batch in range(0, n_images, im_batch_size):
        images_tensor, images_numpy = generate_images(
            netG, noise_dim, device, im_batch_size)

        # images = (images * 255).astype(np.uint8)
        # images = np.clip(images, 0, 255)

        for i in range(images_tensor.shape[0]):
            file_name = os.path.join(save_path, f'image_{i_batch + i:05d}.png')
            # cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # cv2.imwrite(file_name, images[i, :, :, ::-1])
            # myZip.write(file_name)
            # os.remove(file_name)
            vutils.save_image((images_tensor[i, :, :, :]), file_name)

    # shutil.make_archive('images', 'zip', save_path)

def evaluate():
    print('start to evaluate')

    netG = Generator(noise_dim, 32, 3)
    netG.load_state_dict(torch.load(
        os.path.join(MODEL_PATH, 'netG_state_dict.pth')))
    netG = netG.to(device)
    netG.eval()

    save_zip(model=netG)

    images_path = [OUTPUT_PATH, os.path.join(DATA_HOME, 'all-dogs')]
    public_path = 'classify_image_graph_def.pb'

    # user_images_unzipped_path = '../output_images'
    # images_path = [user_images_unzipped_path,'../all-dogs/all-dogs/']
    # model_path = '../input/dog-face-generation-competition-kid-metric-input/classify_image_graph_def.pb'
    calc_score(images_path, public_path)


    # todo:
    # 8.Use stability tricks from RL
    # 9.Use SGD for discriminator and ADAM for generator
    # 13: Add noise to inputs, decay over time
if args.eval == 'T':
    evaluate()