import torch
import os
import random
import numpy as np
import argparse

def rotate(images):
    x = images
    x_90 = x.transpose(2,3)
    x_180 = x.flip(2,3)
    x_270 = x.transpose(2,3).flip(2,3)
    images = torch.cat((x, x_90, x_180, x_270), 0)
    return images

def seed_torch(seed=1337):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def time_output(sec):
    sec = sec % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return "%02dh:%02dm:%02ds" % (hour, min, sec)

def save_checkpoint(points, path, name='checkpoint.pth.tar'):
    if not os.path.exists(path):
        os.makedirs(path)
    torch.save(points, os.path.join(path, name))

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x

def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

import errno
def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise

def convert_to_img(im):
    from PIL import Image
    # [-1, 1] --> [0, 255]
    im = (im + 1.0) * 127.5
    im = im.astype(np.uint8)
    im = np.transpose(im, (1, 2, 0))
    im = Image.fromarray(im)
    return im
