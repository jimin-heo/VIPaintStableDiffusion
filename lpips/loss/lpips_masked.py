"""Stripped version of https://github.com/richzhang/PerceptualSimilarity/tree/master/models"""

from collections import namedtuple

import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from ..util import get_ckpt_path


class LPIPS(nn.Module):
    # Learned perceptual metric
    def __init__(self, use_dropout=True):
        super().__init__()
        self.scaling_layer = ScalingLayer()
        self.chns = [64, 128, 256, 512, 512]  # vg16 features
        self.net = vgg16(pretrained=True, requires_grad=False)
        self.lin0 = NetLinLayer(self.chns[0], use_dropout=use_dropout)
        self.lin1 = NetLinLayer(self.chns[1], use_dropout=use_dropout)
        self.lin2 = NetLinLayer(self.chns[2], use_dropout=use_dropout)
        self.lin3 = NetLinLayer(self.chns[3], use_dropout=use_dropout)
        self.lin4 = NetLinLayer(self.chns[4], use_dropout=use_dropout)
        self.load_from_pretrained()
        for param in self.parameters():
            param.requires_grad = False

    def load_from_pretrained(self, name="vgg_lpips"):
        ckpt = get_ckpt_path(name, "sgm/modules/autoencoding/lpips/loss")
        self.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        print("loaded pretrained LPIPS loss from {}".format(ckpt))

    @classmethod
    def from_pretrained(cls, name="vgg_lpips"):
        if name != "vgg_lpips":
            raise NotImplementedError
        model = cls()
        ckpt = get_ckpt_path(name)
        model.load_state_dict(
            torch.load(ckpt, map_location=torch.device("cpu")), strict=False
        )
        return model


    def forward(self, input, target, mask=None, downsample=False):
        # input, target: (B,3,H,W); mask: (B,H,W) or (B,1,H,W), values in {0,1}
        in0, in1 = self.scaling_layer(input), self.scaling_layer(target)
        if not downsample:
            in1 = in1 * mask + in0.detach() * (1 - mask)

        outs0, outs1 = self.net(in0), self.net(in1)

        lins = [self.lin0, self.lin1, self.lin2, self.lin3, self.lin4]
        res = []

        for kk, lin in enumerate(lins):
            f0 = normalize_tensor(outs0[kk])
            f1 = normalize_tensor(outs1[kk])
            diff = (f0 - f1).pow(2)
            out_feat = lin.model(diff)  # (B,1,hk,wk)

            if mask is not None:
                # ensure mask has shape (B,1,H,W)
                m = mask.unsqueeze(1) if mask.dim()==3 else mask
                # downsample mask to feature map size
                m_k = F.interpolate(m.float(),
                                    size=out_feat.shape[2:],
                                    mode='bilinear',)
                # weighted sum over spatial dims
                weighted = (out_feat * m_k).sum(dim=[2,3], keepdim=True)
                # normalize by number of masked pixels
                norm = m_k.sum(dim=[2,3], keepdim=True).clamp(min=1e-6)
                res.append(weighted / norm)
            else:
                res.append(spatial_average(out_feat, keepdim=True))

        # sum losses from all layers
        return torch.stack(res, dim=0).sum(dim=0)


class ScalingLayer(nn.Module):
    def __init__(self):
        super(ScalingLayer, self).__init__()
        self.register_buffer(
            "shift", torch.Tensor([-0.030, -0.088, -0.188])[None, :, None, None]
        )
        self.register_buffer(
            "scale", torch.Tensor([0.458, 0.448, 0.450])[None, :, None, None]
        )

    def forward(self, inp):
        return (inp - self.shift) / self.scale


class NetLinLayer(nn.Module):
    """A single linear layer which does a 1x1 conv"""

    def __init__(self, chn_in, chn_out=1, use_dropout=False):
        super(NetLinLayer, self).__init__()
        layers = (
            [
                nn.Dropout(),
            ]
            if (use_dropout)
            else []
        )
        layers += [
            nn.Conv2d(chn_in, chn_out, 1, stride=1, padding=0, bias=False),
        ]
        self.model = nn.Sequential(*layers)


class vgg16(torch.nn.Module):
    def __init__(self, requires_grad=False, pretrained=True):
        super(vgg16, self).__init__()
        vgg_pretrained_features = models.vgg16(pretrained=pretrained).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        self.N_slices = 5
        for x in range(4):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(4, 9):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(9, 16):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(16, 23):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(23, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_2 = h
        h = self.slice2(h)
        h_relu2_2 = h
        h = self.slice3(h)
        h_relu3_3 = h
        h = self.slice4(h)
        h_relu4_3 = h
        h = self.slice5(h)
        h_relu5_3 = h
        vgg_outputs = namedtuple(
            "VggOutputs", ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3"]
        )
        out = vgg_outputs(h_relu1_2, h_relu2_2, h_relu3_3, h_relu4_3, h_relu5_3)
        return out


def normalize_tensor(x, eps=1e-10):
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)


def spatial_average(x, keepdim=True):
    return x.mean([2, 3], keepdim=keepdim)
