# -*- encoding: utf-8 -*-
'''
@File        :main.py
@Date        :2021/04/14 16:05
@Author      :Wentong Liao, Kai Hu
@Email       :liao@tnt.uni-hannover.de
@Version     :0.1
@Description : Implementation of SSA-GAN
'''
from __future__ import print_function
import multiprocessing

import os
import io
import sys
import time
import errno
import random
import pprint
import datetime
import dateutil.tz
import argparse

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import torchvision.utils as vutils
from tensorboardX import SummaryWriter
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from miscc.utils import mkdir_p
from miscc.utils import imagenet_deprocess_batch
from miscc.config import cfg, cfg_from_file
from miscc.losses import DAMSM_loss
from sync_batchnorm import DataParallelWithCallback
#from datasets_everycap import TextDataset
from datasets import TextDataset
from datasets import prepare_data
from DAMSM import RNN_ENCODER, CNN_ENCODER
from model import NetG, NetD

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), './.')))
sys.path.append(dir_path)

multiprocessing.set_start_method('spawn', True)


UPDATE_INTERVAL = 200


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def sampling(text_encoder, netG, dataloader, ixtoword, device):
    model_dir = cfg.TRAIN.NET_G
    text_encoder_dir = model_dir.replace('netG', 'text_encoder')
    istart = cfg.TRAIN.NET_G.rfind('_') + 1
    iend = cfg.TRAIN.NET_G.rfind('.')
    start_epoch = int(cfg.TRAIN.NET_G[istart:iend])

    '''
    for path_count in range(11):
        if path_count > 0:
            current_epoch = next_epoch
        else:
            current_epoch = start_epoch
        next_epoch = start_epoch + path_count * 10
        model_dir = model_dir.replace(str(current_epoch), str(next_epoch))
        text_encoder_dir = text_encoder_dir.replace(str(current_epoch), str(next_epoch))
        
    '''

    for num_epoch in [600]:
        model_dir = model_dir.replace(str(start_epoch), str(num_epoch))
        text_encoder_dir = text_encoder_dir.replace(str(start_epoch), str(num_epoch))
        start_epoch = num_epoch

        #split_dir = 'valid'
        split_dir = 'test_every'
        # Build and load the generator
        netG.load_state_dict(torch.load(model_dir))
        netG.eval()
        text_encoder.load_state_dict(torch.load(text_encoder_dir))
        text_encoder.eval()

        batch_size = cfg.TRAIN.BATCH_SIZE
        #s_tmp = model_dir
        s_tmp = model_dir[:model_dir.rfind('.pth')]
        s_tmp_dir = s_tmp
        img_save_dir = '%s/%s' % (s_tmp, split_dir)
        mkdir_p(img_save_dir)
        #cap_save_dir = '%s/%s' % (s_tmp, 'caps')
        # mkdir_p(cap_save_dir)
        idx = 0
        cnt = 0
        for i in range(1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
            for step, data in enumerate(dataloader, 0):
                imags, captions, cap_lens, class_ids, keys = prepare_data(data)
                cnt += batch_size
                if step % 100 == 0:
                    print('step: ', step)
                # if step > 50:
                #     break
                hidden = text_encoder.init_hidden(batch_size)
                # words_embs: batch_size x nef x seq_len
                # sent_emb: batch_size x nef
                words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
                words_embs, sent_emb = words_embs.detach(), sent_emb.detach()

                # code for generating captions
                #cap_imgs = cap2img(ixtoword, captions, cap_lens, s_tmp_dir)

                #######################################################
                # (2) Generate fake images
                ######################################################
                with torch.no_grad():
                    noise = torch.randn(batch_size, 100)
                    noise = noise.to(device)
                    fake_imgs, _ = netG(noise, sent_emb)
                for j in range(batch_size):
                    #s_tmp = '%s/single/%s' % (save_dir, keys[j])
                    s_tmp = '%s/single' % (img_save_dir)
                    folder = s_tmp[:s_tmp.rfind('/')]
                    if not os.path.isdir(folder):
                        print('Make a new folder: ', folder)
                        mkdir_p(folder)
                    im = fake_imgs[j].data.cpu().numpy()
                    # [-1, 1] --> [0, 255]
                    im = (im + 1.0) * 127.5
                    im = im.astype(np.uint8)
                    im = np.transpose(im, (1, 2, 0))
                    im = Image.fromarray(im)
                    idx += 1
                    #fullpath = '%s_%3d.png' % (s_tmp,i)
                    fullpath = '%s_s%d.png' % (s_tmp, idx)
                    im.save(fullpath)


def cap2img(ixtoword, caps, cap_lens, save_dir=None):
    imgs = []
    if save_dir is not None:
        f = open(os.path.join(save_dir, 'captions.txt'), 'a')
    else:
        f = open('captions.txt', 'a')

    for cap, cap_len in zip(caps, cap_lens):
        idx = cap[:cap_len].cpu().numpy()
        caption = []
        caption_line = []
        for i, index in enumerate(idx, start=1):
            caption.append(ixtoword[index])
            caption_line.append(ixtoword[index])
            if i % 4 == 0 and i > 0:
                caption.append("\n")
        caption_line.append("\n")
        caption = " ".join(caption)
        caption_line = " ".join(caption_line)
        f.writelines(caption_line)
        fig = plt.figure(figsize=(2.5, 1.5))
        plt.axis("off")
        plt.text(0.5, 0.5, caption)
        plt.xlim(0, 10)
        plt.ylim(0, 10)
        buf = io.BytesIO()
        plt.savefig(buf, format="png")
        plt.close(fig)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        img = transforms.ToTensor()(img)
        imgs.append(img)
    f.close()
    imgs = torch.stack(imgs, dim=0)
    assert imgs.dim() == 4, "image dimension must be 4D"
    return imgs


def write_images_losses(writer, imgs, fake_imgs, errD, d_loss, DAMSM_D, errG, DAMSM_G, epoch):
    index = epoch
    writer.add_scalar('errD/d_loss', errD, index)
    writer.add_scalar('errD/MAGP', d_loss, index)
    writer.add_scalar('errD/DAMSM', DAMSM_D, index)
    writer.add_scalar('errG/g_loss', errG, index)
    writer.add_scalar('errG/DAMSM', DAMSM_G, index)
    imgs_print = imagenet_deprocess_batch(imgs)
    #imgs_64_print = imagenet_deprocess_batch(fake_imgs[0])
    #imgs_128_print = imagenet_deprocess_batch(fake_imgs[1])
    imgs_256_print = imagenet_deprocess_batch(fake_imgs)
    writer.add_image('images/img1_pred', torchvision.utils.make_grid(imgs_256_print, normalize=True, scale_each=True), index)
    #writer.add_image('images/img2_caption', torchvision.utils.make_grid(cap_imgs, normalize=True, scale_each=True), index)
    writer.add_image('images/img3_real', torchvision.utils.make_grid(imgs_print, normalize=True, scale_each=True), index)


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def prepare_labels(batch_size):
    # Kai: real_labels and fake_labels have data type: torch.float32
    # match_labels has data type: torch.int64
    real_labels = Variable(torch.FloatTensor(batch_size).fill_(1))
    fake_labels = Variable(torch.FloatTensor(batch_size).fill_(0))
    match_labels = Variable(torch.LongTensor(range(batch_size)))
    if cfg.CUDA:
        real_labels = real_labels.cuda()
        fake_labels = fake_labels.cuda()
        match_labels = match_labels.cuda()
    return real_labels, fake_labels, match_labels


def train(dataloader, ixtoword, netG, netD, text_encoder, image_encoder,
          optimizerG, optimizerD, optimizerEncoder, state_epoch, batch_size, device):
    base_dir = os.path.join('tmp', cfg.CONFIG_NAME, str(cfg.TRAIN.NF))

    if not cfg.RESTORE:
        writer = SummaryWriter(os.path.join(base_dir, 'writer'))
    else:
        writer = SummaryWriter(os.path.join(base_dir, 'writer_new'))

    mkdir_p('%s/models' % base_dir)
    real_labels, fake_labels, match_labels = prepare_labels(batch_size)

    # Build and load the generator and discriminator
    if cfg.RESTORE:
        model_dir = cfg.TRAIN.NET_G
        netG.load_state_dict(torch.load(model_dir))
        model_dir_D = model_dir.replace('netG', 'netD')
        netD.load_state_dict(torch.load(model_dir_D))
        model_dir_text_encoder = model_dir.replace('netG', 'text_encoder')
        text_encoder.load_state_dict(torch.load(model_dir_text_encoder))
        model_dir_image_encoder = model_dir.replace('netG', 'image_encoder')
        image_encoder.load_state_dict(torch.load(model_dir_image_encoder))
        netG.train()
        netD.train()
        text_encoder.train()
        image_encoder.train()
        istart = cfg.TRAIN.NET_G.rfind('_') + 1
        iend = cfg.TRAIN.NET_G.rfind('.')
        state_epoch = int(cfg.TRAIN.NET_G[istart:iend])

    for epoch in tqdm(range(state_epoch + 1, cfg.TRAIN.MAX_EPOCH + 1)):
        data_iter = iter(dataloader)
        # for step, data in enumerate(dataloader, 0):
        for step in tqdm(range(len(data_iter))):
            data = data_iter.next()

            imags, captions, cap_lens, class_ids, keys = prepare_data(data)
            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs_de, sent_emb_de = words_embs.detach(), sent_emb.detach()

            imgs = imags[0].to(device)
            real_features = netD(imgs)
            output = netD.module.COND_DNET(real_features, sent_emb_de)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()

            output = netD.module.COND_DNET(real_features[:(batch_size - 1)], sent_emb_de[1:batch_size])
            errD_mismatch = torch.nn.ReLU()(1.0 + output).mean()

            # synthesize fake images
            noise = torch.randn(batch_size, 100)
            noise = noise.to(device)
            fake, _ = netG(noise, sent_emb_de)

            # update encoder
            DAMSM_D = DAMSM_loss(image_encoder, imgs, real_labels, words_embs,
                                 sent_emb, match_labels, cap_lens, class_ids)
            optimizerEncoder.zero_grad()
            DAMSM_D.backward()
            optimizerEncoder.step()

            # G does not need update with D
            fake_features = netD(fake.detach())

            errD_fake = netD.module.COND_DNET(fake_features, sent_emb_de)
            errD_fake = torch.nn.ReLU()(1.0 + errD_fake).mean()

            errD = errD_real + (errD_fake + errD_mismatch) / 2.0
            optimizerD.zero_grad()
            errD.backward()
            optimizerD.step()

            # MA-GP
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb_de.data).requires_grad_()
            features = netD(interpolated)
            out = netD.module.COND_DNET(features, sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated, sent_inter),
                                        grad_outputs=torch.ones(out.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0, grad1), dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD.zero_grad()
            d_loss.backward()
            optimizerD.step()

            # update G
            features = netD(fake)
            output = netD.module.COND_DNET(features, sent_emb_de)
            errG = - output.mean()
            DAMSM_G = 0.1 * DAMSM_loss(image_encoder, fake, real_labels, words_embs_de,
                                       sent_emb_de, match_labels, cap_lens, class_ids)
            errG_total = errG + DAMSM_G
            optimizerG.zero_grad()
            errG_total.backward()
            optimizerG.step()

        #cap_imgs = cap2img(ixtoword, captions, cap_lens)
        #write_images_losses(writer, cap_imgs, imgs, fake, errD, d_loss, DAMSM_D, errG, DAMSM_G, epoch)
        write_images_losses(writer, imgs, fake, errD, d_loss, DAMSM_D, errG, DAMSM_G, epoch)

        if (epoch >= cfg.TRAIN.WARMUP_EPOCHS) and (epoch % cfg.TRAIN.GSAVE_INTERVAL == 0) and (epoch % 10 != 0):
            torch.save(netG.state_dict(), '%s/models/netG_%03d.pth' % (base_dir, epoch))
            torch.save(text_encoder.state_dict(), '%s/models/text_encoder_%03d.pth' % (base_dir, epoch))
        if (epoch >= cfg.TRAIN.WARMUP_EPOCHS) and (epoch % cfg.TRAIN.DSAVE_INTERVAL == 0):
            torch.save(netD.state_dict(), '%s/models/netD_%03d.pth' % (base_dir, epoch))
            torch.save(image_encoder.state_dict(), '%s/models/image_encoder_%03d.pth' % (base_dir, epoch))
    count = 0
    return count


if __name__ == "__main__":
    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        #args.manualSeed = random.randint(1, 10000)
    print("seed now is : ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    ##########################################################################
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
        (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    # Kai: i don't want to specify a gpu id
    # torch.cuda.set_device(cfg.GPU_ID)

    cudnn.benchmark = True

    # Get data loader ##################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        dataset = TextDataset(cfg.DATA_DIR, 'test',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        ixtoword = dataset.ixtoword
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:
        dataset = TextDataset(cfg.DATA_DIR, 'train',
                              base_size=cfg.TREE.BASE_SIZE,
                              transform=image_transform)
        ixtoword = dataset.ixtoword
        print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # # validation data #

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, 100).to(device)
    netD = NetD(cfg.TRAIN.NF).to(device)
    netG = DataParallelWithCallback(netG)
    netD = nn.DataParallel(netD)

    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()

    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
    state_dict = \
        torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    image_encoder.cuda()

    # get parameters from text_encoder and image_encoder
    if not cfg.B_VALIDATION:
        encoder_parameters = list(text_encoder.parameters())
        for v in image_encoder.parameters():
            if v.requires_grad:
                encoder_parameters.append(v)
        optimizerEncoder = torch.optim.Adam(encoder_parameters, lr=0.00004, betas=(0.0, 0.9))

    state_epoch = 0

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD = torch.optim.Adam(netD.parameters(), lr=0.0004, betas=(0.0, 0.9))

    if cfg.B_VALIDATION:
        count = sampling(text_encoder, netG, dataloader, ixtoword, device)  # generate images for the whole valid dataset
        print('state_epoch:  %d' % (state_epoch))
    else:

        count = train(dataloader, ixtoword, netG, netD, text_encoder, image_encoder, optimizerG, optimizerD, optimizerEncoder, state_epoch, batch_size, device)
