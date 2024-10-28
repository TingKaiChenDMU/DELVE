from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

import os
os.environ['DISPLAY'] = ':0'

def conv2d(filter_in, filter_out, kernel_size):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=1, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(filters_list, in_filters, out_filter):
    # nn.ModuleList和nn.Sequential的区别 % https://blog.csdn.net/watermelon1123/article/details/89954224
    # https://blog.csdn.net/e01528/article/details/84397174
    # 总体而言，nn.ModuleList没有forward函数，所以他不可以进行前向传播，因此需要借助于 在YoloBody中所使用的enumarate函数
    # 逐个的遍历nn.ModuleList层，进行forward的传播
    m = nn.ModuleList([
        conv2d(in_filters, filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),
        conv2d(filters_list[0], filters_list[1], 3),
        conv2d(filters_list[1], filters_list[0], 1),

        conv2d(filters_list[0], filters_list[1], 3),
        nn.Conv2d(filters_list[1], out_filter, kernel_size=1,
                                        stride=1, padding=0, bias=True)
    ])
    return m

class YoloBody(nn.Module):
    def __init__(self, config, num_in_features):
        super(YoloBody, self).__init__()
        self.config = config
        #---------------------------------------------------#   
        #   生成darknet53的主干模型
        #   获得三个有效特征层，他们的shape分别是：
        #   52,52,256
        #   26,26,512
        #   13,13,1024
        #---------------------------------------------------#
        self.backbone = darknet53(None)

        # CTK修改
        self.num_in_features = num_in_features

        # out_filters : [64, 128, 256, 512, 1024]
        out_filters = self.backbone.layers_out_filters

        #------------------------------------------------------------------------#
        #   计算yolo_head的输出通道数，对于voc数据集而言
        #   final_out_filter0 = final_out_filter1 = final_out_filter2 = 75
        #------------------------------------------------------------------------#
        final_out_filter0 = len(config["yolo"]["anchors"][0]) * (5 + config["yolo"]["classes"])

        # print('final_out_filter0的大小为：%s\n'%(len(config["yolo"]["anchors"][0])))
        # print('final_out_filter0的大小为：%s\n'%(config["yolo"]["classes"]))
        # print('final_out_filter0的大小为：%s\n'%((5 + config["yolo"]["classes"])))
        # print('final_out_filter0的大小为：%s\n'%(final_out_filter0))

        self.last_layer0 = make_last_layers([512, 1024], out_filters[-1], final_out_filter0)      # last_layer0 的shape（b,75,13,13）

        final_out_filter1 = len(config["yolo"]["anchors"][1]) * (5 + config["yolo"]["classes"])
        self.last_layer1_conv = conv2d(512, 256, 1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer1 = make_last_layers([256, 512], out_filters[-2] + 256, final_out_filter1) # last_layer1 的shape（b,75,26,26）

        final_out_filter2 = len(config["yolo"]["anchors"][2]) * (5 + config["yolo"]["classes"])
        self.last_layer2_conv = conv2d(256, 128, 1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.last_layer2 = make_last_layers([128, 256], out_filters[-3] + 128, final_out_filter2) # last_layer2 的shape（b,75,52,52）

        # CTK 修改
        self.change_channel_Conv2d_13x13 = nn.Conv2d(in_channels=1024, out_channels=self.num_in_features, kernel_size=3, stride=1, bias=False, padding=1)
        self.change_channel_BN_13x13 = nn.BatchNorm2d(self.num_in_features)
        self.change_channel_ReLu_13x13 = nn.LeakyReLU(0.1)

        self.change_channel_Conv2d_26x26 = nn.Conv2d(in_channels=512,  out_channels=self.num_in_features, kernel_size=3, stride=1, bias=False, padding=1)
        self.change_channel_BN_26x26 = nn.BatchNorm2d(self.num_in_features)
        self.change_channel_ReLu_26x26 = nn.LeakyReLU(0.1)

        self.change_channel_Conv2d_52x52 = nn.Conv2d(in_channels=256,  out_channels=self.num_in_features, kernel_size=3, stride=1, bias=False, padding=1)
        self.change_channel_BN_52x52 = nn.BatchNorm2d(self.num_in_features)
        self.change_channel_ReLu_52x52 = nn.LeakyReLU(0.1)

    # CTK 修改
    def forward(self, x):
        def _branch(last_layer, layer_in):
            for i, e in enumerate(last_layer):  # 其中的i 代表第几个，而 e代表是什么结构
                layer_in = e(layer_in)
                if i == 4:
                    out_branch = layer_in
                if i == 5:
                    out_branch_embedding = layer_in
            return layer_in, out_branch, out_branch_embedding
        #---------------------------------------------------#   
        #   获得三个有效特征层，他们的shape分别是：
        #   x2：52,52,256；
        #   x1: 26,26,512；
        #   x0: 13,13,1024
        #---------------------------------------------------#
        x2, x1, x0 = self.backbone(x)

        #---------------------------------------------------#
        # CTK  x0 的shape是（bs,1024,13,13）

        #   第一个特征层
        #   out0 = (batch_size,255,13,13)
        #---------------------------------------------------#
        # 前5个卷积
        # 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512 -> 13,13,1024 -> 13,13,512
        # 最后2个卷积
        # 13,13,512 -> 13,13,1024 -> 13,13, 3 * (num_class + 1 + 4)
        # CTK  修改 out0的shape为 （13,13, 3 * (num_class + 1 + 4)）; out0_embedding的shape为 bs, 1024, 13,13
        out0, out0_branch, out0_embedding = _branch(self.last_layer0, x0)
        # 等号右侧：out0_embedding的shape的shape为 bs, 1024, 13,13,
        # 等号左侧：out0_embedding_13x13的shape为 bs, 128, 13,13,
        out0_embedding_Conv2d_13x13 = self.change_channel_Conv2d_13x13(out0_embedding)
        out0_embedding_BN_13x13 = self.change_channel_BN_13x13(out0_embedding_Conv2d_13x13)
        out0_embedding_ReLu_13x13 = self.change_channel_ReLu_13x13(out0_embedding_BN_13x13)

##################################################################################################

        # 13,13,512 -> 13,13,256 卷积
        x1_in = self.last_layer1_conv(out0_branch)
        # 13,13,256 -> 26,26,256 上采样
        x1_in = self.last_layer1_upsample(x1_in)

        # bs, 256, 26,26, + bs, 512, 26,26 -> bs, 768, 26,26
        x1_in = torch.cat([x1_in, x1], 1)


        #---------------------------------------------------#
        #   第二个特征层
        #   out1 = (batch_size,255,26,26)
        #---------------------------------------------------#
        # 前5个卷积
        # 26,26,768 -> 26,26,256 -> 26,26,512 -> 26,26,256 -> 26,26,512 -> 26,26,256
        # 后面2个卷积
        # 26, 26, 256 -> 26,26,512 -> 26,26,3 * （num_class + 1 + 4）
        out1, out1_branch, out1_embedding = _branch(self.last_layer1, x1_in)
        # 等号右侧：out1_embedding 的shape的shape为 bs, 512, 13,13,
        # 等号左侧：out1_embedding_26x26 的shape为 bs, 128, 13,13,
        out1_embedding_Conv2d_26x26 = self.change_channel_Conv2d_26x26(out1_embedding)
        out1_embedding_BN_26x26 = self.change_channel_BN_26x26(out1_embedding_Conv2d_26x26)
        out1_embedding_ReLu_26x26 = self.change_channel_ReLu_26x26(out1_embedding_BN_26x26)

##################################################################################################

        # 26,26,256 -> 26,26,128
        x2_in = self.last_layer2_conv(out1_branch)
        # 26,26,128 -> 52,52,128 上采样
        x2_in = self.last_layer2_upsample(x2_in)

        # 52,52,128 + 52,52,256 -> 52,52,384
        x2_in = torch.cat([x2_in, x2], 1)
        #---------------------------------------------------#
        #   第一个特征层
        #   out2 = (batch_size,255,52,52)
        #---------------------------------------------------#
        # 前面5个卷积
        # 52,52,384 -> 52,52,128 -> 52,52,256 -> 52,52,128 -> 52,52,256 -> 52,52,128
        # 后面2个卷积
        # 52, 52, 128 -> 52, 52, 256 ->  52, 52, 3 * (num_class + 1 +4)
        out2, _, out2_embedding = _branch(self.last_layer2, x2_in)
        # 等号右侧：out2_embedding 的shape的shape为 bs, 256, 13,13,
        # 等号左侧：out2_embedding_52x52 的shape为 bs, 128, 13,13,
        out2_embedding_Conv2d_52x52 = self.change_channel_Conv2d_52x52(out2_embedding)
        out2_embedding_BN_52x52 = self.change_channel_BN_52x52(out2_embedding_Conv2d_52x52)
        out2_embedding_ReLu_52x52 = self.change_channel_ReLu_52x52(out2_embedding_BN_52x52)

        # out0 = (batch_size,255,13,13)
        # out1 = (batch_size,255,26,26)
        # out2 = (batch_size,255,52,52)
        # out0_embedding_13x13 = （bs, 128,13,13）
        # out0_embedding_26x26 = （bs, 128,26,26）
        # out0_embedding_52x52 = （bs, 128,52,52）
        return out0, out1, out2, out0_embedding_ReLu_13x13, out1_embedding_ReLu_26x26, out2_embedding_ReLu_52x52





