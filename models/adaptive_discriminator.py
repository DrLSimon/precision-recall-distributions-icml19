import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        self.mean = nn.Parameter(mean.clone().detach().view(1,-1, 1, 1))
        self.std = nn.Parameter(std.clone().detach().view(1,-1, 1, 1))

    def forward(self, img):
        return (img - self.mean) / self.std


def get_vgg19():
    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406])
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225])
    vgg = models.vgg19(pretrained=True).features
    vgg = vgg[:30]
    vgg = nn.Sequential(Normalization(cnn_normalization_mean,cnn_normalization_std),vgg)
    return vgg


class AlexDiscriminator(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.nc = nc
        self.input = nn.Sequential(
                Normalization(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
                nn.AdaptiveAvgPool2d(224)
                )

        self.alex = models.alexnet(pretrained=True)
        self.alex.classifier = self.alex.classifier[:-1]

    def forward(self, x):
        out = self.input(x)
        out = self.alex(out)
        return out

class VGGDiscriminator(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.nc = nc
        self.input = nn.Sequential(
                Normalization(mean=torch.Tensor([0.485, 0.456, 0.406]), std=torch.Tensor([0.229, 0.224, 0.225])),
                nn.AdaptiveAvgPool2d(224)
                )

        self.vgg = models.vgg11(pretrained=True)
        self.vgg.classifier = self.vgg.classifier[:-1]

    def forward(self, x):
        out = self.input(x)
        out = self.vgg(out)
        return out

class AdaptiveDiscriminator(nn.Module):
    def __init__(self, resolution, nc=3):
        super().__init__()
        self.nc = nc
        nin = resolution*resolution*nc
        nh = 64
        self.fc = nn.Sequential(
                nn.Linear(nin, nh),
                nn.ReLU(inplace=True),
                nn.Linear(nh, 1)
                )
    
    def forward(self, x):
        batch_size = x.size(0)
        out = x.view(batch_size, -1)
        out = self.fc(out)

        return out.view(-1)



class RyanAdaptiveDiscriminator(nn.Module):
    def __init__(self, size,nc=3, nfilter=64, nfilter_max=512):
        super().__init__()
        s0 = self.s0 = 4
        nf = self.nf = nfilter
        nf_max = self.nf_max = nfilter_max

        pow2 = max(2, math.ceil(np.log2(size)))
        self.input_pool = nn.AdaptiveAvgPool2d(2**pow2)
        size = 2**pow2

        # Submodules
        nlayers = int(np.log2(size / s0))
        self.nf0 = min(nf_max, int(nf * 2**nlayers))

        blocks = [
            ResnetBlock(nf, nf)
        ]

        for i in range(nlayers):
            nf0 = min(nf * 2**i, nf_max)
            nf1 = min(nf * 2**(i+1), nf_max)
            blocks += [
                nn.AvgPool2d(3, stride=2, padding=1),
                ResnetBlock(nf0, nf1),
            ]

        self.conv_img = nn.Conv2d(nc, 1*nf, 3, padding=1)
        self.resnet = nn.Sequential(*blocks)
        self.fc = nn.Linear(int(self.nf0*s0*s0), 1)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.input_pool(x)
        out = self.conv_img(out)
        out = self.resnet(out)
        out = out.view(batch_size, -1)
        out = self.fc(actvn(out))

        return out.view(-1)
    
    
class ResnetBlock(nn.Module):
    def __init__(self, fin, fout, fhidden=None, is_bias=True):
        super().__init__()
        # Attributes
        self.is_bias = is_bias
        self.learned_shortcut = (fin != fout)
        self.fin = fin
        self.fout = fout
        if fhidden is None:
            self.fhidden = min(fin, fout)
        else:
            self.fhidden = fhidden

        # Submodules
        self.conv_0 = nn.Conv2d(self.fin, self.fhidden, 3, stride=1, padding=1)
        self.conv_1 = nn.Conv2d(self.fhidden, self.fout, 3, stride=1, padding=1, bias=is_bias)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(self.fin, self.fout, 1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_s = self._shortcut(x)
        dx = self.conv_0(actvn(x))
        dx = self.conv_1(actvn(dx))
        out = x_s + 0.1*dx

        return out

    def _shortcut(self, x):
        if self.learned_shortcut:
            x_s = self.conv_s(x)
        else:
            x_s = x
        return x_s


def actvn(x):
    out = F.leaky_relu(x, 2e-1)
    return out

