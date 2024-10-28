import math
from collections import OrderedDict

import torch
import torch.nn as nn

import os
os.environ['DISPLAY'] = ':0'

#---------------------------------------------------------------------#
#   残差结构
#   利用一个1x1卷积下降通道数，然后利用一个3x3卷积提取特征并且上升通道数
#   最后接上一个残差边
#---------------------------------------------------------------------#
class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes): # inplanes=1024, planes=[512, 1024]
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                               stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes[0])
        self.relu1 = nn.LeakyReLU(0.1)
        
        self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes[1])
        self.relu2 = nn.LeakyReLU(0.1)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out += residual
        return out


class DarkNet(nn.Module):
    def __init__(self, layers):
        super(DarkNet, self).__init__()
        self.inplanes = 32
        # 416,416,3 -> 416,416,32
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.LeakyReLU(0.1)

        # 416,416,32 -> 208,208,64
        self.layer1 = self._make_layer([32, 64], layers[0])    #layers[0] = 1
        # 208,208,64 -> 104,104,128
        self.layer2 = self._make_layer([64, 128], layers[1])   #layers[1] = 2
        # 104,104,128 -> 52,52,256
        self.layer3 = self._make_layer([128, 256], layers[2])  #layers[2] = 8
        # 52,52,256 -> 26,26,512
        self.layer4 = self._make_layer([256, 512], layers[3])  #layers[3] = 8
        # 26,26,512 -> 13,13,1024
        self.layer5 = self._make_layer([512, 1024], layers[4]) #layers[4] = 4

        self.layers_out_filters = [64, 128, 256, 512, 1024]

        #CTK 下面这几行代码，CTK屏蔽掉之后并没有什么影响
        # 进行权值初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    #---------------------------------------------------------------------#
    #   在每一个layer里面，首先利用一个步长为2的3x3卷积进行下采样
    #   然后进行残差结构的堆叠
    #---------------------------------------------------------------------#
    def _make_layer(self, planes, blocks):  # 对于残差4这个单元 self._make_layer([512, 1024], layers[4]) #layers[4] = 4
        layers = []
        # 下采样，步长为2，卷积核大小为3
        layers.append(("ds_conv", nn.Conv2d(self.inplanes, planes[1], kernel_size=3, stride=2, padding=1, bias=False)))  # self.inplanes=512， planes[1] = 1024
        layers.append(("ds_bn", nn.BatchNorm2d(planes[1])))
        layers.append(("ds_relu", nn.LeakyReLU(0.1)))

        # 加入残差结构
        self.inplanes = planes[1]
        for i in range(0, blocks):
            layers.append(("residual_{}".format(i), BasicBlock(self.inplanes, planes)))  # inplanes=1024, planes=[512, 1024]
        # 这里使用orderedDict是因为上面利用layer这一个list进行收集相关的层，经过查阅资料可以得知，在使用list收集
        # 相关层的时候，list并不会按照顺序进行保存，是一种乱序的模式。因此使用OrdererDict来是它按照收集的顺序保存。
        # 然后使用nn.sequential来嵌套OrderedDict里面的内容。
        return nn.Sequential(OrderedDict(layers))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        out3 = self.layer3(x)
        out4 = self.layer4(out3)
        out5 = self.layer5(out4)

        return out3, out4, out5

def darknet53(pretrained, **kwargs):
    model = DarkNet([1, 2, 8, 8, 4])
    # 下面加载预训练权重部分，不会被使用，
    if pretrained:
        if isinstance(pretrained, str): # CTK 判定所传入的pretrained参数是不是一个str类型
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception("darknet request a pretrained path. got [{}]".format(pretrained))
    return model
