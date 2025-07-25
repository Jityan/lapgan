import os
import sys
import numpy as np
import pandas as pd
from PIL import Image
import numpy.random as random
if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import clip as clip


def prepare_data_full(data, text_encoder, device):
    imgs, captions, CLIP_tokens, keys = data
    new_imgs = []
    for i in range(len(imgs)):
        new_imgs.append(imgs[i].to(device))
    CLIP_tokens_list = []
    sent_emb_list = []
    words_embs_list = []
    for t in CLIP_tokens:
        t = t.to(device)
        CLIP_tokens_list.append(t)
        sent_emb, words_embs = encode_tokens(text_encoder, t)
        sent_emb_list.append(sent_emb)
        words_embs_list.append(words_embs)
    return new_imgs, captions, CLIP_tokens_list, sent_emb_list, words_embs_list, keys


def encode_tokens(text_encoder, caption):
    # encode text
    with torch.no_grad():
        sent_emb,words_embs = text_encoder(caption)
        sent_emb,words_embs = sent_emb.detach(), words_embs.detach()
    return sent_emb, words_embs 


def get_imgs(img_path, imsize=None, bbox=None, transform=None, normalize=None, args=None):
    img = Image.open(img_path).convert('RGB')
    width, height = img.size
    if bbox is not None:
        r = int(np.maximum(bbox[2], bbox[3]) * 0.75)
        center_x = int((2 * bbox[0] + bbox[2]) / 2)
        center_y = int((2 * bbox[1] + bbox[3]) / 2)
        y1 = np.maximum(0, center_y - r)
        y2 = np.minimum(height, center_y + r)
        x1 = np.maximum(0, center_x - r)
        x2 = np.minimum(width, center_x + r)
        img = img.crop([x1, y1, x2, y2])
    if transform is not None:
        img = transform(img)
    ret = []
    for i in range(args.stage):
        if i < (args.stage - 1):
            re_img = transforms.Resize(imsize[i])(img)
        else:
            re_img = img
        if normalize is not None:
            ret.append(normalize(re_img))
    return ret


def get_caption(cap_path, clip_info):
    eff_captions = []
    with open(cap_path, "r") as f:
        captions = f.read().encode('utf-8').decode('utf8').split('\n')
    for cap in captions:
        if len(cap) != 0:
            eff_captions.append(cap)
    tokens = []
    for c in eff_captions:
        tokens.append(clip.tokenize(c,truncate=True)[0])
    return eff_captions, tokens


################################################################
#                    Dataset
################################################################
class FullTextImgDataset(data.Dataset):
    def __init__(self, split, transform=None, args=None):
        self.args = args
        self.transform = transform
        self.clip4text = args.clip4text
        self.data_dir = args.data_dir
        self.dataset_name = args.dataset_name
        self.base_img_size = int(args.imsize / args.stage)
        self.imsize = []
        for i in range(args.stage):
            self.imsize.append(self.base_img_size)
            self.base_img_size = self.base_img_size * 2
        self.norm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
        self.split=split
        
        if self.data_dir.find('birds') != -1:
            self.bbox = self.load_bbox()
        else:
            self.bbox = None
        self.split_dir = os.path.join(self.data_dir, split)
        self.filenames = self.load_filenames(self.data_dir, split)
        self.number_example = len(self.filenames)

    def load_bbox(self):
        data_dir = self.data_dir
        bbox_path = os.path.join(data_dir, 'CUB_200_2011/bounding_boxes.txt')
        df_bounding_boxes = pd.read_csv(bbox_path,
                                        delim_whitespace=True,
                                        header=None).astype(int)
        #
        filepath = os.path.join(data_dir, 'CUB_200_2011/images.txt')
        df_filenames = \
            pd.read_csv(filepath, delim_whitespace=True, header=None)
        filenames = df_filenames[1].tolist()
        #print('Total filenames: ', len(filenames), filenames[0])
        #
        filename_bbox = {img_file[:-4]: [] for img_file in filenames}
        numImgs = len(filenames)
        for i in range(0, numImgs):
            # bbox = [x-left, y-top, width, height]
            bbox = df_bounding_boxes.iloc[i][1:].tolist()
            key = filenames[i][:-4]
            filename_bbox[key] = bbox
        return filename_bbox

    def load_filenames(self, data_dir, split):
        filepath = '%s/%s/filenames.pickle' % (data_dir, split)
        if os.path.isfile(filepath):
            with open(filepath, 'rb') as f:
                filenames = pickle.load(f)
            print('Load filenames from: %s (%d)' % (filepath, len(filenames)))
        else:
            filenames = []
        return filenames

    def __getitem__(self, index):
        #
        key = self.filenames[index]
        data_dir = self.data_dir
        #
        if self.bbox is not None:
            bbox = self.bbox[key]
        else:
            bbox = None
        #
        if self.dataset_name.lower().find('coco') != -1:
            if self.split=='train':
                img_name = '%s/images/train2014/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
            else:
                img_name = '%s/images/val2014/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key)
        elif self.dataset_name.lower().find('cc3m') != -1:
            if self.split=='train':
                img_name = '%s/images/train/%s.jpg' % (data_dir, key)
                text_name = '%s/text/train/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/test/%s.jpg' % (data_dir, key)
                text_name = '%s/text/test/%s.txt' % (data_dir, key.split('_')[0])
        elif self.dataset_name.lower().find('cc12m') != -1:
            if self.split=='train':
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
            else:
                img_name = '%s/images/%s.jpg' % (data_dir, key)
                text_name = '%s/text/%s.txt' % (data_dir, key.split('_')[0])
        elif self.dataset_name.lower().find('birds') != -1:
            img_name = '%s/CUB_200_2011/images/%s.jpg' % (data_dir, key)
            text_name = '%s/text/%s.txt' % (data_dir, key)
        else:
            img_name = '%s/%s.jpg' % (data_dir, key)
            text_name = '%s/text/%s.txt' % (data_dir, key.split('/')[-1])
        #
        imgs = get_imgs(img_name, self.imsize, bbox, self.transform, normalize=self.norm, args=self.args)
        caps,tokens = get_caption(text_name,self.clip4text)
        return imgs, caps, tokens, key

    def __len__(self):
        return len(self.filenames)

import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--save-path', type=str, default='./saved_model', help='save path')
    parser.add_argument('--dataset-name', type=str, default='coco', help='dataset name', choices=['flowers', 'birds', 'coco'])
    parser.add_argument('--imsize', type=int, default=128, help='image dimensions')
    parser.add_argument('--stage', type=int, default=2, help='number of stages')
    args = parser.parse_args()
    args.clip4text = {'src':"clip", 'type':'ViT-B/32'}
    args.clip4trn = {'src':"clip", 'type':'ViT-B/32'}
    args.clip4evl = {'src':"clip", 'type':'ViT-B/32'}

    if args.dataset_name == 'birds':
        args.data_dir = '../data/birds'
    elif args.dataset_name == 'flowers':
        args.data_dir = '../data/flowers'
    else:
        args.data_dir = '../data/coco'

    image_transform = transforms.Compose([
        transforms.Resize(int(args.imsize * 76 / 64)),
        transforms.RandomCrop(args.imsize),
        transforms.RandomHorizontalFlip(),
    ])
    
    dataset = FullTextImgDataset(split='test', transform=image_transform, args=args)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=0, shuffle='False')
    for sample in dataloader:
        imgs, captions, tokens, keys = sample
        print(len(imgs), captions, len(captions), len(tokens), keys)
        for c in captions:
            print(c[0])
        exit()
