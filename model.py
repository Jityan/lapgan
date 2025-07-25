
import torch
import torch.nn as nn
import torch.nn.functional as F

from util import set_gpu

def load_clip(clip_info, device):
    import clip as clip
    model = clip.load(clip_info['type'], device=device)[0]
    return model

class CLIP_TXT_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_TXT_ENCODER, self).__init__()
        self.define_module(CLIP)
        # print(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, CLIP):
        self.transformer = CLIP.transformer
        self.vocab_size = CLIP.vocab_size
        self.token_embedding = CLIP.token_embedding
        self.positional_embedding = CLIP.positional_embedding
        self.ln_final = CLIP.ln_final
        self.text_projection = CLIP.text_projection

    @property
    def dtype(self):
        return self.transformer.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        sent_emb = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection
        return sent_emb.type(torch.cuda.FloatTensor), x # convert into float32, original clip is float16


class CLIP_IMG_ENCODER(nn.Module):
    def __init__(self, CLIP):
        super(CLIP_IMG_ENCODER, self).__init__()
        model = CLIP.visual
        # print(model)
        self.define_module(model)
        for param in self.parameters():
            param.requires_grad = False

    def define_module(self, model):
        self.conv1 = model.conv1
        self.class_embedding = model.class_embedding
        self.positional_embedding = model.positional_embedding
        self.ln_pre = model.ln_pre
        self.transformer = model.transformer
        self.ln_post = model.ln_post
        self.proj = model.proj

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def transf_to_CLIP_input(self,inputs):
        device = inputs.device
        if len(inputs.size()) != 4:
            raise ValueError('Expect the (B, C, X, Y) tensor.')
        else:
            mean = torch.tensor([0.48145466, 0.4578275, 0.40821073])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            var = torch.tensor([0.26862954, 0.26130258, 0.27577711])\
                .unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)
            inputs = F.interpolate(inputs*0.5+0.5, size=(224, 224))
            inputs = ((inputs+1)*0.5-mean)/var
            return inputs

    def forward(self, img: torch.Tensor):
        x = self.transf_to_CLIP_input(img)
        x = x.type(self.dtype)
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        grid =  x.size(-1)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)
        # NLD -> LND
        x = x.permute(1, 0, 2)
        # Local features
        #selected = [1,4,7,12]
        selected = [1,4,8]
        local_features = []
        for i in range(12):
            x = self.transformer.resblocks[i](x)
            if i in selected:
                local_features.append(x.permute(1, 0, 2)[:, 1:, :].permute(0, 2, 1).reshape(-1, 768, grid, grid).contiguous().type(img.dtype))
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x[:, 0, :])
        if self.proj is not None:
            x = x @ self.proj
        return torch.stack(local_features, dim=1), x.type(img.dtype)


def KL_loss(mu, logvar):
    # -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.mean(KLD_element).mul_(-0.5)
    return KLD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        nn.init.orthogonal_(m.weight.data, 1.0)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def load_params(model, new_param):
    for p, new_p in zip(model.parameters(), new_param):
        p.data.copy_(new_p)

import copy
def copy_G_params(model):
    flatten = copy.deepcopy(list(p.data for p in model.parameters()))
    return flatten


class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc/2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


def conv3x3(in_planes, out_planes):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                     padding=1, bias=False)


# ############## G networks ################################################
# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


# Keep the spatial size
def Block3x3_relu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes * 2),
        nn.BatchNorm2d(out_planes * 2),
        GLU()
    )
    return block


class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num * 2),
            nn.BatchNorm2d(channel_num * 2),
            GLU(),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num)
        )


    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        return out


class CA_NET(nn.Module):
    # some code is modified from vae examples
    # (https://github.com/pytorch/examples/blob/master/vae/main.py)
    def __init__(self, embed_dim=128, is_cuda=True):
        super(CA_NET, self).__init__()
        self.t_dim = 512
        self.ef_dim = embed_dim
        self.fc = nn.Linear(self.t_dim, self.ef_dim * 4, bias=True) # change to 2 for 512 dimension
        self.relu = GLU()
        self.is_cuda = is_cuda

    def encode(self, text_embedding):
        x = self.relu(self.fc(text_embedding))
        mu = x[:, :self.ef_dim]
        logvar = x[:, self.ef_dim:]
        return mu, logvar

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if self.is_cuda:
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def forward(self, text_embedding):
        #print(text_embedding.shape)
        mu, logvar = self.encode(text_embedding)
        #print(mu.shape, logvar.shape)
        c_code = self.reparametrize(mu, logvar)
        #print(c_code.shape)
        return c_code, mu, logvar


class INIT_STAGE_G(nn.Module):
    def __init__(self, ngf, noise_dim=100, embed_dim=128):
        super(INIT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.in_dim = noise_dim + embed_dim
        self.define_module()

    def define_module(self):
        in_dim = self.in_dim
        ngf = self.gf_dim
        self.fc = nn.Sequential(
            nn.Linear(in_dim, ngf * 4 * 4 * 2, bias=False),
            nn.BatchNorm1d(ngf * 4 * 4 * 2),
            GLU())


        self.upsample1 = upBlock(ngf, ngf // 2)
        self.upsample2 = upBlock(ngf // 2, ngf // 4)
        self.upsample3 = upBlock(ngf // 4, ngf // 8)
        self.upsample4 = upBlock(ngf // 8, ngf // 16)

    def forward(self, z_code, c_code=None):
        in_code = torch.cat((c_code, z_code), 1)
        # state size 16ngf x 4 x 4
        out_code = self.fc(in_code)
        out_code = out_code.view(-1, self.gf_dim, 4, 4)
        # state size 8ngf x 8 x 8
        out_code = self.upsample1(out_code)
        # state size 4ngf x 16 x 16
        out_code = self.upsample2(out_code)
        # state size 2ngf x 32 x 32
        out_code = self.upsample3(out_code)
        # state size ngf x 64 x 64
        out_code = self.upsample4(out_code)

        return out_code


class NEXT_STAGE_G(nn.Module):
    def __init__(self, ngf, embed_dim=128, num_residual=2):
        super(NEXT_STAGE_G, self).__init__()
        self.gf_dim = ngf
        self.ef_dim = embed_dim
        self.num_residual = num_residual
        self.define_module()

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(self.num_residual):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = self.gf_dim
        efg = self.ef_dim

        self.jointConv = Block3x3_relu(ngf + efg, ngf)
        self.residual = self._make_layer(ResBlock, ngf)
        self.upsample = upBlock(ngf, ngf // 2)

    def forward(self, h_code, c_code):
        s_size = h_code.size(2)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, s_size, s_size)
        # state size (ngf+egf) x in_size x in_size
        h_c_code = torch.cat((c_code, h_code), 1)
        # state size ngf x in_size x in_size
        out_code = self.jointConv(h_c_code)
        out_code = self.residual(out_code)
        # state size ngf/2 x 2in_size x 2in_size
        out_code = self.upsample(out_code)

        return out_code


class GET_IMAGE_G(nn.Module):
    def __init__(self, ngf):
        super(GET_IMAGE_G, self).__init__()
        self.gf_dim = ngf
        self.img = nn.Sequential(
            conv3x3(ngf, 3),
            nn.Tanh()
        )

    def forward(self, h_code):
        out_img = self.img(h_code)
        return out_img


class G_NET(nn.Module):
    def __init__(self, ngf=64, branch=2):
        super(G_NET, self).__init__()
        self.gf_dim = ngf
        self.branch = branch
        self.define_module()

    def define_module(self):
        self.ca_net = CA_NET(embed_dim=256)

        if self.branch > 0:
            self.h_net1 = INIT_STAGE_G(self.gf_dim * 16, embed_dim=256)
            self.img_net1 = GET_IMAGE_G(self.gf_dim)
        if self.branch > 1:
            self.h_net2 = NEXT_STAGE_G(self.gf_dim, embed_dim=256)
            self.img_net2 = GET_IMAGE_G(self.gf_dim // 2)
        if self.branch > 2:
            self.h_net3 = NEXT_STAGE_G(self.gf_dim // 2, embed_dim=256)
            self.img_net3 = GET_IMAGE_G(self.gf_dim // 4)

    def forward(self, z_code, text_embedding=None):
        c_code, mu, logvar = self.ca_net(text_embedding)
        fake_imgs = []
        if self.branch > 0:
            h_code1 = self.h_net1(z_code, c_code)
            fake_img1 = self.img_net1(h_code1)
            fake_imgs.append(fake_img1)
        if self.branch > 1:
            h_code2 = self.h_net2(h_code1, c_code)
            fake_img2 = self.img_net2(h_code2)
            fake_imgs.append(fake_img2)
        if self.branch > 2:
            h_code3 = self.h_net3(h_code2, c_code)
            fake_img3 = self.img_net3(h_code3)
            fake_imgs.append(fake_img3)

        return fake_imgs, mu, logvar


# ############## D networks ################################################
def Block3x3_leakRelu(in_planes, out_planes):
    block = nn.Sequential(
        conv3x3(in_planes, out_planes),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 2
def downBlock(in_planes, out_planes):
    block = nn.Sequential(
        nn.Conv2d(in_planes, out_planes, 4, 2, 1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return block


# Downsale the spatial size by a factor of 16
def encode_image_by_16times(ndf):
    encode_img = nn.Sequential(
        # --> state size. ndf x in_size/2 x in_size/2
        nn.Conv2d(3, ndf, 4, 2, 1, bias=False),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 2ndf x x in_size/4 x in_size/4
        nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 2),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 4ndf x in_size/8 x in_size/8
        nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 4),
        nn.LeakyReLU(0.2, inplace=True),
        # --> state size 8ndf x in_size/16 x in_size/16
        nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
        nn.BatchNorm2d(ndf * 8),
        nn.LeakyReLU(0.2, inplace=True)
    )
    return encode_img


# For 64 x 64 images
class D_NET64(nn.Module):
    def __init__(self, df_dim=64, ef_dim=512, num_classes=0):
        super(D_NET64, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.num_classes = num_classes
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        if self.num_classes == 0:
            self.num_classes = ndf
        
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())
        
        self.linear = nn.Linear(in_features=ndf*8*4*4, out_features=self.num_classes)
        self.fc_rot = nn.Linear(in_features=self.num_classes, out_features=4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_var, c_code=None, ssl=False):
        x_code = self.img_code_s16(x_var)
        #print(c_code.shape, x_code.shape)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        #print(c_code.shape, x_code.shape)
        #exit()
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        
        # === SSL and Aux
        rot = None
        cs = None
        if ssl:
            x = x_code.view(x_code.size(0), self.df_dim*8*4*4)
            x = self.linear(x)
            cs = self.softmax(x)
            rot = self.fc_rot(x)
        return x_code, rot, cs, output.view(-1), out_uncond.view(-1)


# For 128 x 128 images
class D_NET128(nn.Module):
    def __init__(self, df_dim=64, ef_dim=512, num_classes=0):
        super(D_NET128, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.num_classes = num_classes
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        if self.num_classes == 0:
            self.num_classes = ndf
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())
        
        self.linear = nn.Linear(in_features=ndf*8*4*4, out_features=self.num_classes)
        self.fc_rot = nn.Linear(in_features=self.num_classes, out_features=4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_var, c_code=None, ssl=False):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        
        # === SSL and Aux
        rot = None
        cs = None
        if ssl:
            x = x_code.view(x_code.size(0), self.df_dim*8*4*4)
            x = self.linear(x)
            cs = self.softmax(x)
            rot = self.fc_rot(x)
        return x_code, rot, cs, output.view(-1), out_uncond.view(-1)


# For 256 x 256 images
class D_NET256(nn.Module):
    def __init__(self, df_dim=64, ef_dim=256, num_classes=0):
        super(D_NET256, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.num_classes = num_classes
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        if self.num_classes == 0:
            self.num_classes = ndf
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s64 = downBlock(ndf * 16, ndf * 32)
        self.img_code_s64_1 = Block3x3_leakRelu(ndf * 32, ndf * 16)
        self.img_code_s64_2 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
                nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
                nn.Sigmoid())
        
        self.linear = nn.Linear(in_features=ndf*8*4*4, out_features=self.num_classes)
        self.fc_rot = nn.Linear(in_features=self.num_classes, out_features=4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s64(x_code)
        x_code = self.img_code_s64_1(x_code)
        x_code = self.img_code_s64_2(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        # === SSL and Aux
        x = x_code.view(x_code.size(0), self.df_dim*8*4*4)
        x = self.linear(x)
        cs = self.softmax(x)
        rot = self.fc_rot(x)
        return x_code, rot, cs, output.view(-1), out_uncond.view(-1)





# For 64 x 64 images
class D_NET64_CONV(nn.Module):
    def __init__(self, df_dim=64, ef_dim=512, num_classes=0):
        super(D_NET64_CONV, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.num_classes = num_classes
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        if self.num_classes == 0:
            self.num_classes = ndf
        
        self.img_code_s16 = encode_image_by_16times(ndf)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())
        
        self.fc_rot = nn.Sequential(
            nn.Conv2d(ndf * 8, 4, kernel_size=4, stride=4),
        )

    def forward(self, x_var, c_code=None, ssl=False):
        x_code = self.img_code_s16(x_var)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        rot = None
        if ssl:
            rot = self.fc_rot(x_code)
            rot = rot.view(-1, 4)
        return x_code, rot, None, output.view(-1), out_uncond.view(-1)


# For 128 x 128 images
class D_NET128_CONV(nn.Module):
    def __init__(self, df_dim=64, ef_dim=512, num_classes=0):
        super(D_NET128_CONV, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.num_classes = num_classes
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        if self.num_classes == 0:
            self.num_classes = ndf
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)

        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())

        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Sigmoid())
        
        self.fc_rot = nn.Sequential(
            nn.Conv2d(ndf * 8, 4, kernel_size=4, stride=4),
        )

    def forward(self, x_var, c_code=None, ssl=False):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)

        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)

        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        
        rot = None
        if ssl:
            rot = self.fc_rot(x_code)
            rot = rot.view(-1, 4)
        return x_code, rot, None, output.view(-1), out_uncond.view(-1)






# Label Augmentation
# For 64 x 64 images
class D_NET64_LA(torch.nn.Module):
    def __init__(self, df_dim=64, ef_dim=512):
        super(D_NET64_LA, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.define_module()
    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.logits = torch.nn.Sequential(
            torch.nn.Conv2d(ndf * 8, 8, kernel_size=4, stride=4),)
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = torch.nn.Sequential(
            torch.nn.Conv2d(ndf * 8, 8, kernel_size=4, stride=4),)
    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return x_code, output.view(-1, 8), out_uncond.view(-1,8)

# For 128 x 128 images
class D_NET128_LA(nn.Module):
    def __init__(self, df_dim=64, ef_dim=512):
        super(D_NET128_LA, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.define_module()
    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 8, kernel_size=4, stride=4),)
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 8, kernel_size=4, stride=4),)

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return x_code, output.view(-1, 8), out_uncond.view(-1, 8)

# For 64 x 64 images
class D_NET64_LA2(torch.nn.Module):
    def __init__(self, df_dim=64, ef_dim=512):
        super(D_NET64_LA2, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.define_module()
    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.logits = torch.nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Softmax())
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = torch.nn.Sequential(
            nn.Conv2d(ndf * 8, 8, kernel_size=4, stride=4))

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return x_code, output.view(-1), out_uncond.view(-1,8)

# For 128 x 128 images
class D_NET128_LA2(nn.Module):
    def __init__(self, df_dim=64, ef_dim=512):
        super(D_NET128_LA2, self).__init__()
        self.df_dim = df_dim
        self.ef_dim = ef_dim
        self.define_module()

    def define_module(self):
        ndf = self.df_dim
        efg = self.ef_dim
        self.img_code_s16 = encode_image_by_16times(ndf)
        self.img_code_s32 = downBlock(ndf * 8, ndf * 16)
        self.img_code_s32_1 = Block3x3_leakRelu(ndf * 16, ndf * 8)
        self.logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=4),
            nn.Softmax())
        self.jointConv = Block3x3_leakRelu(ndf * 8 + efg, ndf * 8)
        self.uncond_logits = nn.Sequential(
            nn.Conv2d(ndf * 8, 8, kernel_size=4, stride=4))

    def forward(self, x_var, c_code=None):
        x_code = self.img_code_s16(x_var)
        x_code = self.img_code_s32(x_code)
        x_code = self.img_code_s32_1(x_code)
        c_code = c_code.view(-1, self.ef_dim, 1, 1)
        c_code = c_code.repeat(1, 1, 4, 4)
        # state size (ngf+egf) x 4 x 4
        h_c_code = torch.cat((c_code, x_code), 1)
        # state size ngf x in_size x in_size
        h_c_code = self.jointConv(h_c_code)
        output = self.logits(h_c_code)
        out_uncond = self.uncond_logits(x_code)
        return x_code, output.view(-1), out_uncond.view(-1, 8)




if __name__ == "__main__":
    set_gpu('1')
    gen = G_NET(branch=3).cuda()
    disc1 = D_NET64().cuda()
    disc2 = D_NET128().cuda()
    disc3 = D_NET256().cuda()
    
    noise = torch.FloatTensor(128, 100).cuda()
    noise.data.normal_(0, 1)
    embed = torch.rand(128, 1024).cuda()
    fimgs, mu, logvar = gen(noise, embed)
    print(fimgs[0].shape, fimgs[1].shape, fimgs[2].shape)
    s10, xx, ss, s11, s12 = disc1(fimgs[0], mu)
    s20, xx, ss, s21, s22 = disc2(fimgs[1], mu)
    s30, xx, ss, s31, s32 = disc3(fimgs[2], mu)