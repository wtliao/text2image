# Code adapted from
# https://github.com/openai/improved-gan/blob/master/inception_score/model.py
# which was in turn derived from
# tensorflow/tensorflow/models/image/imagenet/classify_image.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import sys
import tarfile

import numpy as np
from six.moves import urllib
import tensorflow as tf
import glob
from scipy.misc import imread, imresize
# from skimage.transform import resize as imresize
# from skimage.io import imread
import math
import sys
from tqdm import tqdm, trange

parser = argparse.ArgumentParser()
parser.add_argument('--input_npy_file', default=None)
parser.add_argument('--input_image_dir', default='samples/tmp/app/vg/G40/128_5')#app/coco_no_geo/G180/128_
parser.add_argument('--input_image_dir_list', default=None)
parser.add_argument('--input_image_superdir', default=None)
parser.add_argument('--image_size', default=128, type=int)

# Most papers use 50k samples and 10 splits but I don't have that much
# data so I'll use 3 splits for everything
parser.add_argument('--num_splits', default=3, type=int)
parser.add_argument('--tensor_layout', default='NHWC', choices=['NHWC', 'NCHW'])

IMAGE_EXTS = ['.png', '.jpg', '.jpeg']


def main(args):
    got_npy_file = args.input_npy_file is not None
    got_image_dir = args.input_image_dir is not None
    got_image_dir_list = args.input_image_dir_list is not None
    got_image_superdir = args.input_image_superdir is not None
    inputs = [got_npy_file, got_image_dir, got_image_dir_list, got_image_superdir]
    if sum(inputs) != 1:
        raise ValueError('Must give exactly one input type')

    if args.input_npy_file is not None:
        images = np.load(args.input_npy_file)
        images = np.split(images, images.shape[0], axis=0)
        images = [img[0] for img in images]
        mean, std = get_inception_score(args, images)
        print('Inception mean: ', mean)
        print('Inception std: ', std)
    elif args.input_image_dir is not None:
        images = load_images(args, args.input_image_dir)
        mean, std = get_inception_score(args, images)
        print('Inception mean: ', mean)
        print('Inception std: ', std)
    elif got_image_dir_list:
        with open(args.input_image_dir_list, 'r') as f:
            dir_list = [line.strip() for line in f]
        for image_dir in dir_list:
            images = load_images(args, image_dir)
            mean, std = get_inception_score(args, images)
            print('Inception mean: ', mean)
            print('Inception std: ', std)
            print()
    elif got_image_superdir:
        for fn in sorted(os.listdir(args.input_image_superdir)):
            if not fn.startswith('result'): continue
            image_dir = os.path.join(args.input_image_superdir, fn, 'images')
            if not os.path.isdir(image_dir): continue
            images = load_images(args, image_dir)
            mean, std = get_inception_score(args, images)
            print('Inception mean: ', mean)
            print('Inception std: ', std)
            print()


def load_images(args, image_dir):
    print('Loading images from ', image_dir)
    images = []
    args.tensor_layout = 'NHWC'
    for fn in os.listdir(image_dir):
        ext = os.path.splitext(fn)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        img_path = os.path.join(image_dir, fn)
        img = imread(img_path)
        # img = PIL.ImageOps.mirror(img_path)
        # print(img.shape)
        if len(img.shape) != 3 or img.shape[2] != 3:
            print('skip one channel image')
            continue
        if args.image_size is not None:
            img = imresize(img, (args.image_size, args.image_size))
        images.append(img)
    print('Found %d images' % len(images))
    return images


MODEL_DIR = './tmp/imagenet'
DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
softmax = None


# Call this function with list of images. Each of elements should be a
# numpy array with values ranging from 0 to 255.
def get_inception_score(args, images):
    splits = args.num_splits
    layout = args.tensor_layout

    assert (type(images) == list)
    assert (type(images[0]) == np.ndarray)
    assert (len(images[0].shape) == 3)
    print(images[0].min(), images[0].max(), images[0].dtype)
    # assert(np.max(images[0]) > 10)
    # assert(np.min(images[0]) >= 0.0)
    inps = []
    for img in images:
        img = img.astype(np.float32)
        inps.append(np.expand_dims(img, 0))
    bs = 1
    with tf.Session() as sess:
        preds = []
        n_batches = int(math.ceil(float(len(inps)) / float(bs)))
        n_preds = 0
        for i in trange(n_batches, bar_format="{desc:<5}{percentage:3.0f}%|{bar:10}{r_bar}"):
            sys.stdout.write(".")
            sys.stdout.flush()
            inp = inps[(i * bs):min((i + 1) * bs, len(inps))]
            inp = np.concatenate(inp, 0)
            if layout == 'NCHW':
                inp = inp.transpose(0, 2, 3, 1)
            pred = sess.run(softmax, {'ExpandDims:0': inp})
            preds.append(pred)
            n_preds += pred.shape[0]
            # print('Ran %d / %d images' % (n_preds, len(images)))
        preds = np.concatenate(preds, 0)
        scores = []
        for i in range(splits):
            part = preds[(i * preds.shape[0] // splits):((i + 1) * preds.shape[0] // splits), :]
            kl = part * (np.log(part) - np.log(np.expand_dims(np.mean(part, 0), 0)))
            kl = np.mean(np.sum(kl, 1))
            scores.append(np.exp(kl))
        return np.mean(scores), np.std(scores)


# This function is called automatically.
def _init_inception():
    global softmax
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(MODEL_DIR, filename)
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (
                filename, float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()

        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Succesfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)
    with tf.gfile.FastGFile(os.path.join(
            MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')
    # Works with an arbitrary minibatch size.
    with tf.Session() as sess:
        pool3 = sess.graph.get_tensor_by_name('pool_3:0')
        ops = pool3.graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o.set_shape(tf.TensorShape(new_shape))
        w = sess.graph.get_operation_by_name("softmax/logits/MatMul").inputs[1]
        logits = tf.matmul(tf.squeeze(pool3, [1, 2]), w)
        softmax = tf.nn.softmax(logits)


if softmax is None:
    _init_inception()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
    print("test on {} Done".format(args.input_image_dir))
