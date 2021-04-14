import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import io
import errno

import numpy as np
from PIL import Image
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

sys.path.append('..')
from model import NetG,NetD
from DAMSM import RNN_ENCODER

cudnn.benchmark = True
root = '.'

def parse_args():
    parser = argparse.ArgumentParser(description='Playing with the well trained model')
    parser.add_argument('--json', dest='json_file',
                        help='dataset json file',
                        default='./dataset_coco.json', type=str)
    parser.add_argument('--use_gpu', dest='use_gpu', type=bool, default=False)
    parser.add_argument('--model_path', type=str,
                       default='%s/tmp/coco/64/models/netG_120.pth'%root)
    parser.add_argument('--rnn_encoder', type=str,
                       default='%s/tmp/coco/64/models/text_encoder_120.pth'%root)
                       #default='../../DAMSMencoders/pretrained/coco/text_encoder100.pth')
    args = parser.parse_args()
    return args

def get_caption_idx(dataset_json, caption):
    caption = caption.split(' ')
    word2idx = dataset_json['word2idx']
    caption_idx = []
    for word in caption:
        assert word in word2idx, word + ' is not in word dictionary, please try another word!'
        word_idx = word2idx[word]
        caption_idx.append(word_idx)
        
    caption = np.asarray(caption_idx).astype('int64')
    if (caption == 0).sum() > 0:
        print('ERROR: do not need END (0) token', caption)
    num_words = len(caption)
    # pad with 0s (i.e., '<end>')
    x = np.zeros((dataset_json['max_words_per_cap'], 1), dtype='int64')
    x_len = num_words
    if num_words <= dataset_json['max_words_per_cap']:
        x[:num_words, 0] = caption
    else:
        ix = list(np.arange(num_words))  # 1, 2, 3,..., maxNum
        np.random.shuffle(ix)
        ix = ix[:dataset_json['max_words_per_cap']]
        ix = np.sort(ix)
        x[:, 0] = caption[ix]
        x_len = dataset_json['max_words_per_cap']
    return x, x_len

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
            
def cap2img(ixtoword, caps, cap_lens):
    imgs = []
    for cap, cap_len in zip(caps, cap_lens):
        idx = cap[:cap_len].cpu().numpy()
        caption = []
        for i, index in enumerate(idx, start=1):
            caption.append(ixtoword[index])
            if i % 4 == 0 and i > 0:
                caption.append("\n")
        caption = " ".join(caption)
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
    imgs = torch.stack(imgs, dim=0)
    assert imgs.dim() == 4, "image dimension must be 4D"
    return imgs

def main(args):
    ## manualSeed to control the noise
    manualSeed = 100
    random.seed(manualSeed)
    np.random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    
    with open(args.json_file, 'r') as f:
        dataset_json = json.load(f)
    
    
    ## load rnn encoder
    text_encoder = RNN_ENCODER(dataset_json['n_words'], nhidden=dataset_json['text_embed_dim'])
    text_encoder_dir = args.rnn_encoder
    state_dict = torch.load(text_encoder_dir, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    
    ## load netG
    state_dict = torch.load(args.model_path, map_location=torch.device('cpu'))
    #netG = NetG(int(dataset_json['n_channels']), int(dataset_json['cond_dim']))
    netG = NetG(64, int(dataset_json['cond_dim']))
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`nvidia
        new_state_dict[name] = v
    model_dict = netG.state_dict()
    pretrained_dict = {k: v for k, v in new_state_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    netG.load_state_dict(model_dict)
    
    ## use gpu or not, change model to evaluation mode
    if args.use_gpu:
        text_encoder.cuda()
        netG.cuda()
        caption_idx.cuda()
        caption_len.cuda()
        noise.cuda()
        
    text_encoder.eval()
    netG.eval()
        
    ## generate noise
    num_noise = 100
    noise = torch.FloatTensor(num_noise, 100)
    
    ## cub bird captions
    #caption = 'this small bird has a light yellow breast and brown wings'
    #caption = 'this small bird has a short beak a light gray breast a darker gray crown and black wing tips'
    #caption = 'this small bird has wings that are gray and has a white belly'
    #caption = 'this bird has a yellow throat belly abdomen and sides with lots of brown streaks on them'
    #caption = 'this little bird has a yellow belly and breast with a gray wing with white wingbars'
    #caption = 'this bird has a white belly and breast wit ha blue crown and nape'
    #caption = 'a bird with brown and black wings red crown and throat and the bill is short and pointed'
    #caption = 'this small bird has a yellow crown and a white belly'
    #caption = 'this bird has a blue crown with white throat and brown secondaries'
    #caption = 'this bird has wings that are black and has a white belly'
    #caption = 'a yellow bird has wings with dark stripes and small eyes'
    #caption = 'a black bird has wings with dark stripes and small eyes'
    #caption = 'a red bird has wings with dark stripes and small eyes'
    #caption = 'a white bird has wings with dark stripes and small eyes'
    #caption = 'a blue bird has wings with dark stripes and small eyes'
    #caption = 'a pink bird has wings with dark stripes and small eyes'
    #caption = 'this is a white and grey bird with black wings and a black stripe by its eyes'
    #caption = 'a small bird with an orange bill and grey crown and breast'
    #caption = 'a small bird with black gray and white wingbars'
    #caption = 'this bird is white and light orange in color with a black beak'
    #caption = 'a small sized bird that has tones of brown and a short pointed bill' # beak?
    
    ## MS coco captions
    #caption = 'two men skiing down a snow covered mountain in the evening'
    #caption = 'a man walking down a grass covered mountain'
    #caption = 'a close up of a boat on a field under a sunset'
    #caption = 'a close up of a boat on a field with a clear sky'  
    #caption = 'a herd of black and white cattle standing on a field'
    #caption = 'a herd of black and white sheep standing on a field'
    #caption = 'a herd of black and white dogs standing on a field'
    #caption = 'a herd of brown cattle standing on a field'
    #caption = 'a herd of black and white cattle standing in a river'   
    #caption = 'some horses in a field of green grass with a sky in the background'
    #caption = 'some horses in a field of yellow grass with a sky in the background'
    caption = 'some horses in a field of green grass with a sunset in the background'
        
        
    ## convert caption to index    
    caption_idx, caption_len = get_caption_idx(dataset_json, caption) 
    caption_idx = torch.LongTensor(caption_idx)
    caption_len = torch.LongTensor([caption_len])
    caption_idx = caption_idx.view(1, -1)
    caption_len = caption_len.view(-1)
    
    ## use rnn encoder to get caption embedding
    hidden = text_encoder.init_hidden(1)
    words_embs, sent_emb = text_encoder(caption_idx, caption_len, hidden)


    ## generate fake image
    noise.data.normal_(0, 1)
    sent_emb = sent_emb.repeat(num_noise, 1)
    words_embs = words_embs.repeat(num_noise, 1, 1)
    with torch.no_grad():
        fake_imgs, fusion_mask = netG(noise,sent_emb)


        ## create path to save image, caption and mask
        cap_number = 10000
        main_path = 'result/mani/cap_%s_0_coco_ch64' %(str(cap_number))
        img_save_path = '%s/image' % main_path 
        mask_save_path = '%s/mask_' % main_path
        mkdir_p(img_save_path)
        for i in range(7):
            mkdir_p(mask_save_path+str(i))

        ## save caption as image    
        ixtoword = {v:k for k, v in dataset_json['word2idx'].items()}    
        cap_img = cap2img(ixtoword, caption_idx, caption_len) 
        im = cap_img[0].data.cpu().numpy()
        im = (im + 1.0) * 127.5
        im = im.astype(np.uint8)
        im = np.transpose(im, (1, 2, 0))
        im = Image.fromarray(im)
        full_path = '%s/caption.png' % main_path
        im.save(full_path)

        ## save generated images and masks
        for i in tqdm(range(num_noise)):  
            full_path = '%s/image_%d.png' % (img_save_path, i)
            im = fake_imgs[i].data.cpu().numpy()
            im = (im + 1.0) * 127.5
            im = im.astype(np.uint8)
            im = np.transpose(im, (1, 2, 0))
            im = Image.fromarray(im)
            im.save(full_path)

            for j in range(7):
                full_path = '%s%1d/mask_%d.png' % (mask_save_path, j, i)
                im = fusion_mask[j][i][0].data.cpu().numpy()
                im = im * 255
                im = im.astype(np.uint8)
                im = Image.fromarray(im)
                im.save(full_path)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    
    
