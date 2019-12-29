import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from layers import *
from data import voc, coco
import os

class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class IBN(nn.Module):
    def __init__(self, out_planes, bn=True):
        super(IBN, self).__init__()
        self.out_channels = out_planes
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None

    def forward(self, x):
        if self.bn is not None:
            x = self.bn(x)
        return x
		
class DS_module1(nn.Module):
    def __init__(self,):
        super(DS_module1, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.pool4 = nn.Conv2d(64, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.bn = nn.BatchNorm2d(512, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x_pool1)
        x_pool3 = self.pool3(x_pool2)
        x_pool4 = self.pool4(x_pool3)
        x_pool5 = self.bn(x_pool4)
        return x_pool5
		
class DS_module2(nn.Module):
    def __init__(self,):
        super(DS_module2, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool4 = nn.Conv2d(128, 1024, kernel_size=(3, 3), stride=1, padding=1)
        self.bn = nn.BatchNorm2d(1024, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x_pool1)
        x_pool3 = self.pool3(x_pool2)
        x_pool4 = self.pool4(x_pool3)
        x_pool5 = self.bn(x_pool4)
        return x_pool5
		
class DS_module3(nn.Module):
    def __init__(self,):
        super(DS_module3, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=2, padding=0)
        self.pool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.pool5 = nn.Conv2d(128, 512, kernel_size=(3, 3), stride=1, padding=1)
        self.bn = nn.BatchNorm2d(512, eps=1e-5, momentum=0.01, affine=True)

    def forward(self, x):
        x_pool1 = self.pool1(x)
        x_pool2 = self.pool2(x_pool1)
        x_pool3 = self.pool3(x_pool2)
        x_pool4 = self.pool4(x_pool3)
        x_pool5 = self.pool5(x_pool4)
        x_pool6 = self.bn(x_pool5)
        return x_pool6
		
		
class Relu_Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(Relu_Conv, self).__init__()
        self.out_channels = out_planes
        self.relu = nn.ReLU(inplace=False)
        self.single_branch = nn.Sequential(
            ConvBlock(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=1)
        )

    def forward(self, x):
        x = self.relu(x)
        out = self.single_branch(x)
        return out
		
		
class Ds_Conv(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, padding=(1, 1)):
        super(Ds_Conv, self).__init__()
        self.out_channels = out_planes
        self.single_branch = nn.Sequential(
            ConvBlock(in_planes, out_planes, kernel_size=(3, 3), stride=stride, padding=padding, relu=False)
        )

    def forward(self, x):
        out = self.single_branch(x)
        return out


class SSD(nn.Module):
    """Single Shot Multibox Architecture
    The network is composed of a base VGG network followed by the
    added multibox conv layers.  Each multibox layer branches into
        1) conv2d for class conf scores
        2) conv2d for localization predictions
        3) associated priorbox layer to produce default bounding
           boxes specific to the layer's feature map size.
    See: https://arxiv.org/pdf/1512.02325.pdf for more details.

    Args:
        phase: (string) Can be "test" or "train"
        size: input image size
        base: VGG16 layers for input, size of either 300 or 500
        extras: extra layers that feed to multibox loc and conf layers
        head: "multibox head" consists of loc and conf conv layers
    """

    def __init__(self, phase, size, base, extras, head, num_classes):
        super(SSD, self).__init__()
        self.phase = phase
        self.num_classes = num_classes
        self.cfg = (coco, voc)[num_classes == 21]
        self.priorbox = PriorBox(self.cfg)
        self.priors = Variable(self.priorbox.forward(), volatile=True)
        self.size = size
        self.ibn = IBN(512, bn=True)
        self.ibn2 = IBN(1024, bn=True)
        self.ds1 = DS_module1()
        self.ds2 = DS_module2()
        self.ds3 = DS_module3()
		
        self.Norm1 = Relu_Conv(512, 512, stride=1)
        self.Norm2 = Relu_Conv(1024, 1024, stride=1)
        self.Norm3 = Relu_Conv(512, 512, stride=1)
        self.Norm4 = Relu_Conv(256, 256, stride=1)
		
        # convs with s=2 to downsample the features
        self.dsc1 = Ds_Conv(512, 1024, stride=2, padding=(1, 1))
        self.dsc2 = Ds_Conv(1024, 512, stride=2, padding=(1, 1))
        self.dsc3 = Ds_Conv(512, 256, stride=2, padding=(1, 1))
		
        # convs to reduce the feature dimensions of other levels
        self.proj1 = ConvBlock(1024, 128, kernel_size=1, stride=1)
        self.proj2 = ConvBlock(512, 128, kernel_size=1, stride=1)
        self.proj3 = ConvBlock(256, 128, kernel_size=1, stride=1)
		
        # convs to reduce the feature dimensions of current level
        self.agent1 = ConvBlock(512, 256, kernel_size=1, stride=1)
        self.agent2 = ConvBlock(1024, 512, kernel_size=1, stride=1)
        self.agent3 = ConvBlock(512, 256, kernel_size=1, stride=1)
		
        # convs to reduce the feature dimensions of other levels
        self.convert1 = ConvBlock(384, 256, kernel_size=1)
        self.convert2 = ConvBlock(256, 512, kernel_size=1)
        self.convert3 = ConvBlock(128, 256, kernel_size=1)
		
        # convs to merge the features of the current and higher level features
        self.merge1 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)
        self.merge2 = ConvBlock(1024, 1024, kernel_size=3, stride=1, padding=1)
        self.merge3 = ConvBlock(512, 512, kernel_size=3, stride=1, padding=1)

        # SSD network
        self.vgg = nn.ModuleList(base)
        # Layer learns to scale the l2 normalized features from conv4_3
        self.L2Norm = L2Norm(512, 20)
        self.extras = nn.ModuleList(extras)

        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

        if phase == 'test':
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(num_classes, 0, 200, 0.01, 0.45)

    def forward(self, x):
        """Applies network layers and ops on input image(s) x.

        Args:
            x: input image or batch of images. Shape: [batch,3,300,300].

        Return:
            Depending on phase:
            test:
                Variable(tensor) of output class label predictions,
                confidence score, and corresponding location predictions for
                each object detected. Shape: [batch,topk,7]

            train:
                list of concat outputs from:
                    1: confidence layers, Shape: [batch*num_priors,num_classes]
                    2: localization layers, Shape: [batch,num_priors*4]
                    3: priorbox layers, Shape: [2,num_priors*4]
        """
        sources = list()
        loc = list()
        conf = list()
        s1 = list()
        new_sources = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)
            if (k == 2 or k == 5 or k == 7):
                s1.append(x)
				
        conv4_3_bn = self.ibn(x)
        ds1 = self.ds1(s1[0])
        s = self.Norm1(conv4_3_bn * ds1)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        fc7_bn = self.ibn2(x)
        ds2 = self.ds2(s1[1])
        # print(fc7_bn)
        p = self.Norm2(0.8*self.dsc1(s) + fc7_bn * ds2)
        # x = self.vgg[34](x)
        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            # print(k, v)
            x = v(x)
            if k == 1:
                ds3 = self.ds3(s1[2])
                # print("ds3 : {}  p : {}  dsc2(p) : {}  x : {}".format(ds3.size(), p.size(), self.dsc2(p).size(), x.size()))
                w = self.Norm3(0.8*self.dsc2(p) + x * ds3)
            elif k == 3:
                q = self.Norm4(0.8*self.dsc3(w) + x)
                sources.append(q)
            elif k % 2 == 1:
                sources.append(x)
            else:
                pass
				
        # project the forward features into lower dimension.
        tmp1 = self.proj1(p)
        tmp2 = self.proj2(w)
        tmp3 = self.proj3(q)

        # The conv4_3 level
        proj1 = F.upsample(tmp1, size=(38, 38), mode='bilinear')
        proj2 = F.upsample(tmp2, size=(38, 38), mode='bilinear')
        proj3 = F.upsample(tmp3, size=(38, 38), mode='bilinear')
        proj = torch.cat([proj1, proj2, proj3], dim=1)

        agent1 = self.agent1(s)
        convert1 = self.convert1(proj)
        pred1 = torch.cat([agent1, convert1], dim=1)
        pred1 = self.merge1(pred1)
        new_sources.append(pred1)

        # The fc_7 level
        proj2 = F.upsample(tmp2, size=(19, 19), mode='bilinear')
        proj3 = F.upsample(tmp3, size=(19, 19), mode='bilinear')
        proj = torch.cat([proj2, proj3], dim=1)

        agent2 = self.agent2(p)
        convert2 = self.convert2(proj)
        pred2 = torch.cat([agent2, convert2], dim=1)
        pred2 = self.merge2(pred2)
        new_sources.append(pred2)

        # The conv8 level
        proj3 = F.upsample(tmp3, size=(10, 10), mode='bilinear')
        proj = proj3

        agent3 = self.agent3(w)
        convert3 = self.convert3(proj)
        pred3 = torch.cat([agent3, convert3], dim=1)
        pred3 = self.merge3(pred3)
        new_sources.append(pred3)

        for prediction in sources:
            new_sources.append(prediction)
      

        # apply multibox head to source layers
        for (x, l, c) in zip(new_sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        if self.phase == "test":
            output = self.detect(
                loc.view(loc.size(0), -1, 4),                   # loc preds
                self.softmax(conf.view(conf.size(0), -1,
                             self.num_classes)),                # conf preds
                self.priors.type(type(x.data))                  # default boxes
            )
        else:
            output = (
                loc.view(loc.size(0), -1, 4),
                conf.view(conf.size(0), -1, self.num_classes),
                self.priors
            )
        return output

    def load_weights(self, base_file):
        other, ext = os.path.splitext(base_file)
        if ext == '.pkl' or '.pth':
            print('Loading weights into state dict...')
            self.load_state_dict(torch.load(base_file,
                                 map_location=lambda storage, loc: storage))
            print('Finished!')
        else:
            print('Sorry only .pth and .pkl files supported.')


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
def vgg(cfg, i, batch_norm=False):
    layers = []
    in_channels = i
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, batch_norm=False):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k+1], kernel_size=(1,3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    # layers += [ConvBlock(256, 128, kernel_size=1,stride=1)]
    # layers += [ConvBlock(128, 256, kernel_size=3,stride=1)]
    # layers += [ConvBlock(256, 128, kernel_size=1,stride=1)]
    # layers += [ConvBlock(128, 256, kernel_size=3,stride=1)]
    return layers


def multibox(vgg, extra_layers, cfg, num_classes):
    loc_layers = []
    conf_layers = []
    vgg_source = [21, -2]
    for k, v in enumerate(vgg_source):
        loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                 cfg[k] * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(vgg[v].out_channels,
                        cfg[k] * num_classes, kernel_size=3, padding=1)]
    for k, v in enumerate(extra_layers[1::2], 2):
        loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                 * 4, kernel_size=3, padding=1)]
        conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                  * num_classes, kernel_size=3, padding=1)]
    return vgg, extra_layers, (loc_layers, conf_layers)


base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [],
}
extras = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [],
}
mbox = {
    '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
    '512': [],
}


def build_ssd(phase, size=300, num_classes=21):
    if phase != "test" and phase != "train":
        print("ERROR: Phase: " + phase + " not recognized")
        return
    if size != 300:
        print("ERROR: You specified size " + repr(size) + ". However, " +
              "currently only SSD300 (size=300) is supported!")
        return
    base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                     add_extras(extras[str(size)], 1024),
                                     mbox[str(size)], num_classes)
    return SSD(phase, size, base_, extras_, head_, num_classes)
