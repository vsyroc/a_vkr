import torch
from torch import nn
from torch.nn import functional as F

from .utils import _SimpleSegmentationModel

__all__ = ["DeepLabV3"]


class DeepLabV3(_SimpleSegmentationModel):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
            The backbone should return an OrderedDict[Tensor], with the key being
            "out" for the last feature map used, and "aux" if an auxiliary classifier
            is used.
        classifier (nn.Module): module that takes the "out" element returned from
            the backbone and returns a dense prediction.
        aux_classifier (nn.Module, optional): auxiliary classifier used during training
    """
    pass


class DeepLabHeadV3Plus(nn.Module):
    def __init__(self, in_channels, middle_level_channels, low_level_channels, num_classes, aspp_dilate=[2, 4, 8, 10, 12, 16, 24, 38]):
        super(DeepLabHeadV3Plus, self).__init__()
        self.project = nn.Sequential(
            nn.Conv2d(low_level_channels, 16, 1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
        )

        self.aspp = ASPP(in_channels, aspp_dilate)
        self.aspp_middle = ASPPMiddle(middle_level_channels)

        self.classifier = nn.Sequential(
            nn.Conv2d(232, 128, 5, padding=2, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),
            nn.Conv2d(128, 64, 3, padding=1, bias=False),

            #nn.Conv2d(256, 256, kernel_size=5, stride=1, padding=2, bias=False, groups=256),
            #nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            #nn.Dropout(0.1),
            nn.Conv2d(64, 16, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, num_classes, 1)
        )
        ###self.classifier = convert_to_separable_conv(self.classifier)
        self._init_weight()

    def forward(self, feature):
        low_level_feature = self.project(feature['low_level'])
        output_feature = self.aspp(feature['out'])
        middle_feature = self.aspp_middle(feature['middle_level'])
        output_feature = F.interpolate(output_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        middle_feature = F.interpolate(middle_feature, size=low_level_feature.shape[2:], mode='bilinear',
                                       align_corners=False)
        return self.classifier(torch.cat([low_level_feature,middle_feature, output_feature], dim=1))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class DeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes, aspp_dilate=[2, 4, 6, 8, 10, 12]):
        super(DeepLabHead, self).__init__()

        self.classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),

            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )
        self._init_weight()

    def forward(self, feature):
        return self.classifier(feature['out'])

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class AtrousSeparableConvolution(nn.Module):
    """ Atrous Separable Convolution
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, bias=True):
        super(AtrousSeparableConvolution, self).__init__()
        self.body = nn.Sequential(
            # Separable Conv
            nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, bias=bias, groups=in_channels),
            # PointWise Conv
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=bias),
        )

        self._init_weight()

    def forward(self, x):
        return self.body(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels =80
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3, rate4, rate5, rate6, rate7, rate8 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPConv(in_channels, out_channels, rate4))
        modules.append(ASPPConv(in_channels, out_channels, rate5))
        modules.append(ASPPConv(in_channels, out_channels, rate6))
        modules.append(ASPPConv(in_channels, out_channels, rate7))
        modules.append(ASPPConv(in_channels, out_channels, rate8))


        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(9 * out_channels, 2 * out_channels, 1, bias=False),
            nn.BatchNorm2d(2 * out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )
        self.convs = convert_to_separable_conv(self.convs)
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class ASPPMiddle(nn.Module):
    def __init__(self, in_channels):
        super(ASPPMiddle, self).__init__()
        out_channels = 28
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))


        modules.append(ASPPConv(in_channels, out_channels, 2))
        modules.append(ASPPConv(in_channels, out_channels, 6))
        modules.append(ASPPConv(in_channels, out_channels, 8))
        modules.append(ASPPConv(in_channels, out_channels, 10))
        modules.append(ASPPConv(in_channels, out_channels, 14))
        modules.append(ASPPConv(in_channels, out_channels, 18))

        #modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(7 * out_channels, 2 * out_channels, 1, bias=False),
            nn.BatchNorm2d(2 * out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )
        self.convs = convert_to_separable_conv(self.convs)
    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)




def convert_to_separable_conv(module):
    new_module = module
    if isinstance(module, nn.Conv2d) and module.kernel_size[0] > 1:
        new_module = AtrousSeparableConvolution(module.in_channels,
                                                module.out_channels,
                                                module.kernel_size,
                                                module.stride,
                                                module.padding,
                                                module.dilation,
                                                module.bias)
    for name, child in module.named_children():
        new_module.add_module(name, convert_to_separable_conv(child))
    return new_module
