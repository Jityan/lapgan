import argparse
import numpy as np
import os
import torch
from torch import nn
import datetime
import time
from model import G_NET, D_NET64_LA, D_NET128_LA, CLIP_IMG_ENCODER, CLIP_TXT_ENCODER, load_clip, copy_G_params, weights_init, load_params, KL_loss
from util import time_output, save_checkpoint, seed_torch, set_gpu, mkdir_p, convert_to_img, rotate

from prepare_datasets import prepare_dataloaders
from datasets import get_fix_data, prepare_data
from datasets_full import prepare_data_full
import torchvision

from vgg_model import VGGNet

def prepare_imgs_grad(imgs):
    new_imgs = []
    for i in range(len(imgs)):
        new_imgs.append(imgs[i].requires_grad_())
    return new_imgs

def prepare_imgs_rot(imgs):
    new_imgs = []
    for i in range(len(imgs)):
        new_imgs.append(rotate(imgs[i]))
    return new_imgs

def prepare_models(args):
    device = args.device
    CLIP4trn = load_clip(args.clip4trn, device).eval()
    CLIP4evl = load_clip(args.clip4evl, device).eval()
    # image encoder
    CLIP_img_enc = CLIP_IMG_ENCODER(CLIP4trn).to(device)
    for p in CLIP_img_enc.parameters():
        p.requires_grad = False
    CLIP_img_enc.eval()
    # text encoder
    CLIP_txt_enc = CLIP_TXT_ENCODER(CLIP4trn).to(device)
    for p in CLIP_txt_enc.parameters():
        p.requires_grad = False
    CLIP_txt_enc.eval()
    # GAN models
    disc = [
            D_NET64_LA().cuda(),
            D_NET128_LA().cuda()]
    gen = G_NET(branch=len(disc)).cuda()
    # initialize weight
    for i in range(len(disc)):
        disc[i].apply(weights_init)
    gen.apply(weights_init)

    style_loss = VGGNet().cuda()
    for p in style_loss.parameters():
        p.requires_grad = False
    print("Load the style loss model")
    style_loss.eval()
    
    return CLIP4trn, CLIP4evl, CLIP_img_enc, CLIP_txt_enc, gen, disc, style_loss

class Trainer(object):
    def __init__(self, args):
        self.args = args
        self.beta1 = 0.5
        self.num_epochs = args.epochs
        # models
        self.CLIP4trn, self.CLIP4evl, self.image_encoder, self.text_encoder, self.gen, self.disc, self.style_loss = prepare_models(args)
        # optimizer
        self.optimD = []
        for i in range(len(self.disc)):
            self.optimD.append(
                torch.optim.Adam(self.disc[i].parameters(), lr=self.args.lr, betas=(self.beta1, 0.999))
                )
        self.optimG = torch.optim.Adam(self.gen.parameters(), lr=self.args.lr, betas=(self.beta1, 0.999))

        self.train_dataloader, self.valid_dataloader, self.test_dataloader = prepare_dataloaders(args)

    def train(self):
        start_epoch = 0
        # prepare metrices
        trlog = {
            'args': self.args,
            'hist_d': [],
            'hist_dr': [],
            'hist_df': [],
            'hist_g': [],
            'hist_gb': [],
        }
        # check resume point
        checkpoint_file = os.path.join(self.args.save_path, self.args.exp_num, 'checkpoint.pth.tar')
        if os.path.isfile(checkpoint_file):
            checkpoint = torch.load(checkpoint_file)
            trlog = checkpoint['trlog']
            start_epoch = checkpoint['start_epoch'] + 1
            self.gen.load_state_dict(checkpoint['netG_state_dict'])
            self.disc[0].load_state_dict(checkpoint['netD1_state_dict'])
            self.disc[1].load_state_dict(checkpoint['netD2_state_dict'])
            self.optimG.load_state_dict(checkpoint['optimG'])
            self.optimD[0].load_state_dict(checkpoint['optimD1'])
            self.optimD[1].load_state_dict(checkpoint['optimD2'])
            print("Resume from epoch {} ...".format(start_epoch+1))

        avg_param_G = copy_G_params(self.gen)
        fixed_img, fixed_sent, fixed_words, fixed_z, fixed_captions = get_fix_data(self.train_dataloader, self.valid_dataloader, self.text_encoder, args)
        l2_loss = nn.MSELoss().cuda()
        l1_loss = nn.L1Loss().cuda()

        for epoch in range(start_epoch, self.num_epochs):
            time1 = time.time()
            for i in range(len(self.disc)):
                self.disc[i].train()
            self.gen.train()
            self.image_encoder.train()
            # prepare metrics
            temp_log = {
                'd_loss': [],
                'd_rloss': [],
                'd_floss': [],
                'g_loss': [],
                'g_bloss': [],
            }

            for idx, sample in enumerate(self.train_dataloader):
                # image
                real_images, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(sample, self.text_encoder, self.args.device)
                real_images = prepare_imgs_grad(real_images)
                sent_emb = sent_emb.requires_grad_()
                # generate image
                noise = torch.FloatTensor(real_images[0].size(0), self.args.noise_dim).cuda()
                noise.data.normal_(0,1)
                fake_images, mu, logvar = self.gen(noise, sent_emb)
                _, fake_emb = self.image_encoder(fake_images[-1])
                # rotation
                bs = sent_emb.size(0)
                real_images = prepare_imgs_rot(real_images)
                fake_images = prepare_imgs_rot(fake_images)
                sent_emb_ext = sent_emb.repeat(4, 1)
                # rot label
                real_labels = torch.zeros(4*bs,).cuda()
                for i in range(4*bs):
                    if i < bs:
                        real_labels[i] = 0
                    elif i < 2*bs:
                        real_labels[i] = 1
                    elif i < 3*bs:
                        real_labels[i] = 2
                    else:
                        real_labels[i] = 3
                # create label
                real_labels = real_labels * 2
                fake_labels = real_labels + 1
                real_labels = nn.functional.one_hot(real_labels.to(torch.int64), 8).float()
                fake_labels = nn.functional.one_hot(fake_labels.to(torch.int64), 8).float()
                # train discriminator
                total_dloss = 0.0
                total_drloss = 0.0
                total_dfloss = 0.0
                for i in range(len(self.disc)):
                    self.disc[i].zero_grad()
                    d_rloss, d_floss = self.disc_loss(
                        i, real_images[i], fake_images[i], sent_emb_ext, real_labels, fake_labels)
                    d_loss = d_rloss + d_floss
                    d_loss.backward()
                    self.optimD[i].step()
                    total_dloss += d_loss.data.cpu().mean()
                    total_drloss += d_rloss.data.cpu().mean()
                    total_dfloss += d_floss.data.cpu().mean()
                # train generator
                self.gen.zero_grad()
                g_loss, g_bloss = self.gen_loss(
                    fake_images, real_images, sent_emb_ext, real_labels, l1_loss, l2_loss)
                g_loss += (args.ca_coef * KL_loss(mu, logvar))
                text_img_sim = torch.cosine_similarity(fake_emb, sent_emb).mean()
                g_loss += (args.sim_w*text_img_sim)
                g_loss.backward()
                self.optimG.step()

                temp_log['d_loss'].append(total_dloss)
                temp_log['g_loss'].append(g_loss.data.cpu().mean())
                temp_log['d_rloss'].append(total_drloss)
                temp_log['d_floss'].append(total_dfloss)
                temp_log['g_bloss'].append(g_bloss.data.cpu().mean())
                
                for p, avg_p in zip(self.gen.parameters(), avg_param_G):
                    avg_p.mul_(0.999).add_(0.001, p.data)
            time2 = time.time()
            # update 1 epoch loss
            print("Epoch: {}/{}, d_loss={:.4f} - g_loss={:.4f} [{} total {}]".format(
                (epoch+1),
                self.num_epochs,
                np.array(temp_log['d_loss']).mean(),
                np.array(temp_log['g_loss']).mean(),
                datetime.datetime.now(pytz.timezone('Asia/Kuala_Lumpur')).strftime("%H:%M"),
                time_output(time2-time1)
                )
            )

            trlog['hist_d'].append(np.array(temp_log['d_loss']).mean())
            trlog['hist_dr'].append(np.array(temp_log['d_rloss']).mean())
            trlog['hist_df'].append(np.array(temp_log['d_floss']).mean())
            trlog['hist_g'].append(np.array(temp_log['g_loss']).mean())
            trlog['hist_gb'].append(np.array(temp_log['g_bloss']).mean())

            temp_log['d_loss'] = []
            temp_log['d_rloss'] = []
            temp_log['d_floss'] = []
            temp_log['g_loss'] = []
            temp_log['g_bloss'] = []
            
            backup_para = copy_G_params(self.gen)
            load_params(self.gen, avg_param_G)
            save_checkpoint({
                'start_epoch': epoch,
                'netG_state_dict': self.gen.state_dict(),
                'netD1_state_dict': self.disc[0].state_dict(),
                'netD2_state_dict': self.disc[1].state_dict(),
                'optimG': self.optimG.state_dict(),
                'optimD1': self.optimD[0].state_dict(),
                'optimD2': self.optimD[1].state_dict(),
                'trlog': trlog
            }, os.path.join(self.args.save_path, self.args.exp_num))
            if (epoch+1) % self.args.iter == 0:
                save_checkpoint({
                    'start_epoch': epoch,
                    'netG_state_dict': self.gen.state_dict(),
                    'trlog': trlog
                }, os.path.join(self.args.save_path, self.args.exp_num), name='epoch'+str(epoch+1)+'.pth.tar')
                self.sample_one_batch(fixed_z, fixed_sent, epoch+1)
            load_params(self.gen, backup_para)
    
    def disc_loss(self, idx, right_image, fake_image, embed, real_labels, fake_labels):
        embed = embed.detach()
        mis_embed = torch.cat((embed[1:], embed[0:1]), dim=0).detach()
        # obtain logits
        _, r_cond_logit, r_logit = self.disc[idx](right_image, embed)
        _, w_cond_logit, _ = self.disc[idx](right_image, mis_embed)
        _, f_cond_logit, f_logit = self.disc[idx](fake_image.detach(), embed)
        # conditional loss
        real_cond_loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(
            input=r_cond_logit,
            target=real_labels
        ))
        wrong_cond_loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(
            input=w_cond_logit,
            target=fake_labels
        ))
        fake_cond_loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(
            input=f_cond_logit,
            target=fake_labels
        ))
        # unconditional loss
        real_loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(
            input=r_logit,
            target=real_labels
        ))
        fake_loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(
            input=f_logit,
            target=fake_labels
        ))
        # label augmented image realism loss
        d_rloss = (real_loss + fake_loss) * args.img_coef
        # label augmented semantic consistency loss
        d_floss = real_cond_loss + (fake_cond_loss + wrong_cond_loss) / 2.0
        return d_rloss, d_floss
    
    def gen_loss(self, fake_images, right_images, embed, real_labels, l1_loss, l2_loss):
        total_gloss = 0.0
        perceptual_loss = 0.0
        embed = embed.detach()
        # obtain image feature
        for i in range(len(self.disc)):
            ffeat, f_cond_logit, f_logit = self.disc[i](fake_images[i], embed)
            rfeat, _, _ = self.disc[i](right_images[i], embed)
            # l2
            activation_fake = torch.mean(ffeat, 0)
            activation_real = torch.mean(rfeat, 0)
            # conditional loss
            g_cond_loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(
                input=f_cond_logit,
                target=real_labels
            ))
            # unconditional loss
            g_uncond_loss = torch.sum(nn.functional.binary_cross_entropy_with_logits(
                input=f_logit,
                target=real_labels
            ))
            # final loss
            g_bloss = g_cond_loss + (g_uncond_loss * args.img_coef_gen)
            total_gloss += g_bloss
            # L1-distance loss
            total_gloss += (self.args.gamma * l1_loss(fake_images[i], right_images[i]))
            # feature matching loss
            total_gloss += (self.args.beta * l2_loss(activation_fake, activation_real.detach()))
            # perceptual loss
            rfp = self.style_loss(right_images[i])[0]
            ffp = self.style_loss(fake_images[i])[0]
            perceptual_loss += nn.functional.mse_loss(rfp, ffp)
        total_gloss += (perceptual_loss / len(self.disc)) * args.pl
        return total_gloss, g_bloss
    
    def sample_one_batch(self, noise, sent, epoch):
        self.gen.eval()
        with torch.no_grad():
            B = noise.size(0)
            fixed_results_train, _, _ = self.gen(noise[:B//2], sent[:B//2])
            fixed_results_train = fixed_results_train[-1].cpu()
            torch.cuda.empty_cache()
            fixed_results_test, _, _ = self.gen(noise[B//2:], sent[B//2:])
            fixed_results_test = fixed_results_test[-1].cpu()
            torch.cuda.empty_cache()
            fixed_results = torch.cat((fixed_results_train, fixed_results_test), dim=0)
        img_name = 'samples_epoch_%03d.png'%(epoch)
        img_save_path = os.path.join(self.args.save_path, self.args.exp_num, img_name)
        torchvision.utils.save_image(fixed_results.data, img_save_path, nrow=6, value_range=(-1, 1), normalize=True)

    def sampling(self, target='checkpoint.pth.tar'):
        import re
        targetfilename = target
        tfn = targetfilename.split('.')[0]
        checkpoint_file = os.path.join(self.args.save_path, args.exp_num, targetfilename)
        if not os.path.isfile(checkpoint_file):
            print("Pretrained model not found...")
            return False
            
        checkpoint = torch.load(checkpoint_file)
        self.gen.load_state_dict(checkpoint['netG_state_dict'])
        print("Model loaded...")
        self.gen.eval()

        for itr in range(self.args.samp_iter):
            for step, data in enumerate(self.valid_dataloader, 0):
                ######################################################
                # (1) Prepare_data
                ######################################################
                real_images, captions, CLIP_tokens, sent_emb, words_embs, keys = prepare_data(data, self.text_encoder, self.args.device)
                ######################################################
                # (2) Generate fake images
                ######################################################
                batch_size = sent_emb.size(0)
                with torch.no_grad():
                    noise = torch.FloatTensor(batch_size, self.args.noise_dim).to(self.args.device)
                    noise.data.normal_(0,1)
                    fake_images, _, _ = self.gen(noise, sent_emb)
                    batch_img_name = 'e_%02d_step_%04d.png'%(itr, step)
                    batch_img_save_dir  = os.path.join(self.args.save_path, self.args.exp_num, 'batch', 'imgs')
                    batch_img_save_name = os.path.join(batch_img_save_dir, batch_img_name)
                    batch_txt_name = 'e_%02d_step_%04d.txt'%(itr, step)
                    batch_txt_save_dir  = os.path.join(self.args.save_path, self.args.exp_num, 'batch', 'txts')
                    batch_txt_save_name = os.path.join(batch_txt_save_dir, batch_txt_name)
                    mkdir_p(batch_img_save_dir)
                    torchvision.utils.save_image(fake_images[-1].data, batch_img_save_name, nrow=8, value_range=(-1, 1), normalize=True)
                    mkdir_p(batch_txt_save_dir)
                    txt = open(batch_txt_save_name,'w')
                    for cap in captions:
                        txt.write(cap+'\n')
                    txt.close()
                    for j in range(batch_size):
                        text = re.sub('[^0-9a-zA-Z]+', '_', captions[j])[:50]
                        single_img_name = 'e_%02d_step_%04d_%s.png'%(itr, step, text)
                        ######################################################
                        # (3) Save fake images
                        ######################################################
                        im = fake_images[-1][j].data.cpu().numpy()
                        im = convert_to_img(im)
                        single_img_save_dir  = os.path.join(self.args.save_path, self.args.exp_num, 'samp_fake_images_checkpoint', 'class')
                        single_img_save_name = os.path.join(single_img_save_dir, single_img_name)   
                        mkdir_p(single_img_save_dir)
                        im.save(single_img_save_name)
                        ######################################################
                        # (3) Save Real images
                        ######################################################
                        im = real_images[-1][j].data.cpu().numpy()
                        im = convert_to_img(im)
                        single_img_save_dir  = os.path.join(self.args.save_path, self.args.exp_num, 'samp_real_images_checkpoint', 'class')
                        single_img_save_name = os.path.join(single_img_save_dir, single_img_name)   
                        mkdir_p(single_img_save_dir)   
                        im.save(single_img_save_name)
                print('Iteration %d Step %d' % (itr, step))
    
    def sampling_alltext(self, target='checkpoint.pth.tar'):
        import re
        targetfilename = target
        tfn = targetfilename.split('.')[0]
        checkpoint_file = os.path.join(self.args.save_path, args.exp_num, targetfilename)
        if not os.path.isfile(checkpoint_file):
            print("Pretrained model not found...")
            return False
            
        checkpoint = torch.load(checkpoint_file)
        self.gen.load_state_dict(checkpoint['netG_state_dict'])
        print("Model loaded from {}...".format(targetfilename))
        self.gen.eval()

        #count = 0
        for step, data in enumerate(self.test_dataloader, 0):
            ######################################################
            # (1) Prepare_data
            ######################################################
            real_images, captions, _, sent_embs, _, _ = prepare_data_full(data, self.text_encoder, self.args.device)
            for idx, (caption, sent_emb) in enumerate(zip(captions, sent_embs), 0): # should have 10 times
                ######################################################
                # (2) Generate fake images
                ######################################################
                batch_size = sent_emb.size(0) # 1 sample only
                with torch.no_grad():
                    noise = torch.FloatTensor(batch_size, self.args.noise_dim).to(self.args.device)
                    noise.data.normal_(0,1)
                    fake_images, _, _ = self.gen(noise, sent_emb)
                    for j in range(batch_size):
                        text = re.sub('[^0-9a-zA-Z]+', '_', caption[0])[:50]
                        single_img_name = 'step_%06d_%02d_%s.png'%(step, idx, text)
                        ######################################################
                        # (3) Save fake images
                        ######################################################
                        im = fake_images[-1][j].data.cpu().numpy()
                        im = convert_to_img(im)
                        single_img_save_dir  = os.path.join(self.args.save_path, self.args.exp_num, 'alltext_fake_images_checkpoint', 'class')
                        single_img_save_name = os.path.join(single_img_save_dir, single_img_name)   
                        mkdir_p(single_img_save_dir)
                        im.save(single_img_save_name)
                        ######################################################
                        # (3) Save Real images
                        ######################################################
                        im = real_images[-1][j].data.cpu().numpy()
                        im = convert_to_img(im)
                        single_img_save_dir  = os.path.join(self.args.save_path, self.args.exp_num, 'alltext_real_images_checkpoint', 'class')
                        single_img_save_name = os.path.join(single_img_save_dir, single_img_name)   
                        mkdir_p(single_img_save_dir)   
                        im.save(single_img_save_name)
        print("Done...")

import pytz
import sys
def main(args):
    tz = pytz.timezone('Asia/Kuala_Lumpur')
    starttime = datetime.datetime.now(tz)
    print("=> Train start :", starttime)
    seed_torch(seed=args.seed)
    trainer = Trainer(args)
    if not args.is_test:
        trainer.train()
    else:
        trainer.sampling_alltext(target=args.target)
    print("=> Total executed time :", datetime.datetime.now(tz) - starttime)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Text2Img')
    parser.add_argument('--save-path', type=str, default='./saved_model', help='save path')
    parser.add_argument('--exp-num', type=str, default='', help='Name of the experiment folder')
    parser.add_argument('--target', type=str, default='checkpoint.pth.tar')
    parser.add_argument('--is-test', action='store_true') # default false
    parser.add_argument('--iter', type=int, default=50, help="round for generating model checkpoint")
    parser.add_argument('--samp-iter', type=int, default=10, help="round for generating fake samples")
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--gpu_id', type=str, default='0', help='gpu id')
    parser.add_argument('--dataset-name', type=str, default='birds', help='dataset name', choices=['flowers', 'birds', 'coco'])
    parser.add_argument('--imsize', type=int, default=128, help='image dimensions')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--stage', type=int, default=2, help='number of stages')
    parser.add_argument('--noise-dim', type=int, default=100, help='noise dimension')
    parser.add_argument('--lr', type=float, default=0.0002)
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--gamma', type=float, default=1.0, help="coefficient for L1-distance loss")
    parser.add_argument('--beta', type=float, default=1.0, help="coefficient for feature matching loss")
    parser.add_argument('--ca-coef', type=float, default=1.0, help="coefficient for conditioning augmentation loss")
    parser.add_argument('--sim-w', type=float, default=1.0, help="coefficient for text embedding similarity loss")
    parser.add_argument('--pl', type=float, default=0.01, help="coefficient for perceptual loss")
    parser.add_argument('--img-coef', type=float, default=0.8, help="coefficient for image realism loss in D")
    parser.add_argument('--img-coef-gen', type=float, default=1.0, help="coefficient for image realism loss in G")
    args = parser.parse_args()
    args.clip4text = {'src':"clip", 'type':'ViT-B/32'}
    args.clip4trn = {'src':"clip", 'type':'ViT-B/32'}
    args.clip4evl = {'src':"clip", 'type':'ViT-B/32'}

    if args.dataset_name == 'birds':
        args.data_dir = './data/birds'
    elif args.dataset_name == 'flowers':
        args.data_dir = './data/flowers'
    else:
        args.data_dir = './data/coco'
        args.epochs = 150
        args.samp_iter = 1
        args.iter = 10

    set_gpu(args.gpu_id)
    args.device = torch.device("cpu")
    if args.gpu_id:
        args.device = torch.device("cuda")
    main(args)
