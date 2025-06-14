import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck

from .backbones.beit import (
    _make_pretrained_beitl16_512,
    _make_pretrained_beitl16_384,
    _make_pretrained_beitb16_384,
    forward_beit,
)
from .backbones.swin_common import (
    forward_swin,
)
from .backbones.swin2 import (
    _make_pretrained_swin2l24_384,
    _make_pretrained_swin2b24_384,
    _make_pretrained_swin2t16_256,
)
from .backbones.swin import (
    _make_pretrained_swinl12_384,
)
from .backbones.levit import (
    _make_pretrained_levit_384,
    forward_levit,
)
from .backbones.vit import (
    _make_pretrained_vitb_rn50_384,
    _make_pretrained_vitl16_384,
    _make_pretrained_vitb16_384,
    forward_vit,
)

import antialiased_cnns

from .geffnet.gen_efficientnet import tf_efficientnet_lite3


def _calc_same_pad(i, k, s, d):
    """Added by Chris
    """
    return max((-(i // -s) - 1) * s + (k - 1) * d + 1 - i, 0)


def conv2d_same(x, weight, bias=None, stride=(1, 1), padding=(0, 0), dilation=(1, 1), groups=1):
    """Added by Chris
    """
    ih, iw = x.size()[-2:]
    kh, kw = weight.size()[-2:]
    pad_h = _calc_same_pad(ih, kh, stride[0], dilation[0])
    pad_w = _calc_same_pad(iw, kw, stride[1], dilation[1])
    x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
    return F.conv2d(x, weight, bias, stride, (0, 0), dilation, groups)


class Conv2dSame(nn.Conv2d):
    """Added by Chris
    Tensorflow like 'SAME' convolution wrapper for 2D convolutions
    """
    # pylint: disable=unused-argument
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv2dSame, self).__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)

    def forward(self, x):
        """Added by Chris
        """
        return conv2d_same(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def _make_encoder(backbone, features, use_pretrained, groups=1, expand=False, exportable=True, hooks=None,
                  use_vit_only=False, use_readout="ignore", in_features=[96, 256, 512, 1024], in_chan=3, group_width=8, aa=False):
    """Added by Chris: in_chan argument, which is used by _make_pretrained_resnext101_wsl and _make_pretrained_efficientnet_lite3
    """
    if backbone == "beitl16_512":
        pretrained = _make_pretrained_beitl16_512(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # BEiT_512-L (backbone)
    elif backbone == "beitl16_384":
        pretrained = _make_pretrained_beitl16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # BEiT_384-L (backbone)
    elif backbone == "beitb16_384":
        pretrained = _make_pretrained_beitb16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # BEiT_384-B (backbone)
    elif backbone == "swin2l24_384":
        pretrained = _make_pretrained_swin2l24_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [192, 384, 768, 1536], features, groups=groups, expand=expand
        )  # Swin2-L/12to24 (backbone)
    elif backbone == "swin2b24_384":
        pretrained = _make_pretrained_swin2b24_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [128, 256, 512, 1024], features, groups=groups, expand=expand
        )  # Swin2-B/12to24 (backbone)
    elif backbone == "swin2t16_256":
        pretrained = _make_pretrained_swin2t16_256(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # Swin2-T/16 (backbone)
    elif backbone == "swinl12_384":
        pretrained = _make_pretrained_swinl12_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [192, 384, 768, 1536], features, groups=groups, expand=expand
        )  # Swin-L/12 (backbone)
    elif backbone == "next_vit_large_6m":
        from .backbones.next_vit import _make_pretrained_next_vit_large_6m
        pretrained = _make_pretrained_next_vit_large_6m(hooks=hooks)
        scratch = _make_scratch(
            in_features, features, groups=groups, expand=expand
        )  # Next-ViT-L on ImageNet-1K-6M (backbone)
    elif backbone == "levit_384":
        pretrained = _make_pretrained_levit_384(
            use_pretrained, hooks=hooks
        )
        scratch = _make_scratch(
            [384, 512, 768], features, groups=groups, expand=expand
        )  # LeViT 384 (backbone)
    elif backbone == "vitl16_384":
        pretrained = _make_pretrained_vitl16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [256, 512, 1024, 1024], features, groups=groups, expand=expand
        )  # ViT-L/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb_rn50_384":
        pretrained = _make_pretrained_vitb_rn50_384(
            use_pretrained,
            hooks=hooks,
            use_vit_only=use_vit_only,
            use_readout=use_readout,
        )
        scratch = _make_scratch(
            [256, 512, 768, 768], features, groups=groups, expand=expand
        )  # ViT-H/16 - 85.0% Top1 (backbone)
    elif backbone == "vitb16_384":
        pretrained = _make_pretrained_vitb16_384(
            use_pretrained, hooks=hooks, use_readout=use_readout
        )
        scratch = _make_scratch(
            [96, 192, 384, 768], features, groups=groups, expand=expand
        )  # ViT-B/16 - 84.6% Top1 (backbone)

    elif backbone == "resnext101_wsl":
        pretrained = _make_pretrained_resnext101_wsl(use_pretrained, in_chan=in_chan, group_width=group_width)
        scratch = _make_scratch([256, 512, 1024, 2048], features, groups=groups, expand=expand)

    elif backbone == "efficientnet_lite3":
        pretrained = _make_pretrained_efficientnet_lite3(use_pretrained, exportable=exportable, in_chan=in_chan)
        scratch = _make_scratch([32, 48, 136, 384], features, groups=groups, expand=expand)  # efficientnet_lite3
    else:
        print(f"Backbone '{backbone}' not implemented")
        assert False
        
    return pretrained, scratch


def _make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    if len(in_shape) >= 4:
        out_shape4 = out_shape

    if expand:
        out_shape1 = out_shape
        out_shape2 = out_shape*2
        out_shape3 = out_shape*4
        if len(in_shape) >= 4:
            out_shape4 = out_shape*8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0], out_shape1, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1], out_shape2, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2], out_shape3, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
    )
    if len(in_shape) >= 4:
        scratch.layer4_rn = nn.Conv2d(
            in_shape[3], out_shape4, kernel_size=3, stride=1, padding=1, bias=False, groups=groups
        )

    return scratch


def _make_pretrained_efficientnet_lite3(use_pretrained, exportable=False, in_chan=3):
    """Modified by Chris to add in_chan
    """
    #efficientnet = torch.hub.load(
    #    "rwightman/gen-efficientnet-pytorch",
    #    "tf_efficientnet_lite3",
    #    pretrained=use_pretrained,
    #    exportable=exportable
    #)

    efficientnet = tf_efficientnet_lite3()

    if in_chan != 3:
        efficientnet.conv_stem = Conv2dSame(in_chan, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)

    return _make_efficientnet_backbone(efficientnet)


def _make_efficientnet_backbone(effnet):
    pretrained = nn.Module()

    pretrained.layer1 = nn.Sequential(
        effnet.conv_stem, effnet.bn1, effnet.act1, *effnet.blocks[0:2]
    )
    pretrained.layer2 = nn.Sequential(*effnet.blocks[2:3])
    pretrained.layer3 = nn.Sequential(*effnet.blocks[3:5])
    pretrained.layer4 = nn.Sequential(*effnet.blocks[5:9])

    return pretrained
    

def _make_resnet_backbone(resnet):
    pretrained = nn.Module()
    pretrained.layer1 = nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1
    )

    pretrained.layer2 = resnet.layer2
    pretrained.layer3 = resnet.layer3
    pretrained.layer4 = resnet.layer4

    return pretrained

def _resnext(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    #state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
    #model.load_state_dict(state_dict)
    return model

def resnext101_32x8d_wsl(progress=True, **kwargs):
    """Constructs a ResNeXt-101 32x8 model pre-trained on weakly-supervised data
    and finetuned on ImageNet from Figure 5 in
    `"Exploring the Limits of Weakly Supervised Pretraining" <https://arxiv.org/abs/1805.00932>`_

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr.
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnext('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], True, progress, **kwargs)


def _make_pretrained_resnext101_wsl(use_pretrained, in_chan=3, group_width=8, aa=False):
    """Modified by Chris to take in_chan
    """

    if aa:
        if group_width != 8:
            print("group_width must be 8 when using antialiased resnext101_32x8d_wsl, ignoring group_width")
            
        resnet = antialiased_cnns.resnext101_32x8d_wsl(pretrained=use_pretrained)
    else:
        #resnet = torch.hub.load("facebookresearch/WSL-Images", f"resnext101_32x{group_width}d_wsl")
        resnet = resnext101_32x8d_wsl()
    if in_chan != 3:
        resnet.conv1 = torch.nn.Conv2d(in_chan, 64, 7, 2, 3, bias=False)

    return _make_resnet_backbone(resnet)



class Interpolate(nn.Module):
    """Interpolation module.
    """

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.

        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners
        )

        return x


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock, self).__init__()

        self.resConfUnit1 = ResidualConvUnit(features)
        self.resConfUnit2 = ResidualConvUnit(features)

    def forward(self, *xs):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            output += self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        output = nn.functional.interpolate(
            output, scale_factor=2, mode="bilinear", align_corners=True
        )

        return output




class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

        # return out + x


class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, deconv=False, bn=False, expand=False, align_corners=True, size=None):
        """Init.

        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups=1

        self.expand = expand
        out_features = features
        if self.expand==True:
            out_features = features//2
        
        self.out_conv = nn.Conv2d(features, out_features, kernel_size=1, stride=1, padding=0, bias=True, groups=1)

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)
        
        self.skip_add = nn.quantized.FloatFunctional()

        self.size=size

    def forward(self, *xs, size=None):
        """Forward pass.

        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if (size is None) and (self.size is None):
            modifier = {"scale_factor": 2}
        elif size is None:
            modifier = {"size": self.size}
        else:
            modifier = {"size": size}

        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=self.align_corners
        )

        output = self.out_conv(output)

        return output

