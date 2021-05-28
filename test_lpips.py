'''test diversity'''
import torch
import numpy as np
import lpips
from PIL import Image
import os
import torchvision.transforms as T
import argparse
from tqdm import tqdm


def main(args):

    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]
    transform = [T.ToTensor()]
    transform.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))
    im_transform = T.Compose(transform)

    orig_images = os.listdir(args.orig_image_path)
    N = len(orig_images)
    print(N)
    net = lpips.LPIPS(net='alex')
    net = net.cuda()
    net.eval()
    scores = []
    with torch.no_grad():

        for i in tqdm(range(N)):
            orig_image = im_transform(Image.open(os.path.join(args.orig_image_path, orig_images[i])).convert('RGB'))
            orig_image = orig_image.cuda()
            orig_image = orig_image.unsqueeze(0)
            for j in range(args.generated_image_number):
                generated_image = im_transform(Image.open(os.path.join(args.generated_image_path, orig_images[i][:-4] + '_numb_' + str(j) + '.jpg')).convert('RGB'))
                generated_image = generated_image.cuda()
                generated_image = generated_image.unsqueeze(0)
                score = net(orig_image, generated_image).squeeze()
                scores.append(score.cpu().numpy())
    scores_all = np.asarray(scores)
    scores_mean = np.mean(scores_all)
    scores_std = np.std(scores_all)
    print('mean diversity scores = %4.2f%% +- %4.2f%%' % (scores_mean, scores_std))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--orig_image_path', type=str, default="samples/tmp/coco/128/val")
    parser.add_argument('--generated_image_path', type=str, default="samples/tmp/graph/coco_no_geo/G105/128_5")
    parser.add_argument('--generated_image_number', type=int, default=5)
    args = parser.parse_args()

    main(args)
