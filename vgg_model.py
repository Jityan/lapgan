from __future__ import division
from torchvision import models
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

class VGGNet(nn.Module):
    def __init__(self):
        super(VGGNet, self).__init__()
        self.select = ['8'] ## relu2_2 

        model = models.vgg16()
        url = 'https://download.pytorch.org/models/vgg16-397923af.pth'
        model.load_state_dict(model_zoo.load_url(url))

        for param in model.parameters():
            param.resquires_grad = False

        print('Load pretrained model from ', url)
        self.vgg = model.features
        
    def forward(self, x):
        features = []
        for name, layer in self.vgg._modules.items():
            x = layer(x)
            if name in self.select:
                features.append(x)
        return features

if __name__ == '__main__':
    perceptual_loss = 0
    style_loss = VGGNet()
    for p in style_loss.parameters():
        p.requires_grad = False
    print("Load the style loss model")
    style_loss.eval()
    imgs = torch.rand(2, 3, 128, 128)
    real_features = style_loss(imgs)[0]
    fake_features = style_loss(imgs)[0]
    print(real_features.shape)
    perceptual_loss += torch.nn.functional.mse_loss(real_features, fake_features)
    print(perceptual_loss)