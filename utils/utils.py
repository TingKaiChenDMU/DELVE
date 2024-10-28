from __future__ import division

import math
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable
from torchvision.ops import nms

import os
os.environ['DISPLAY'] = ':0'

# CTK  解码过程
class DecodeBox(nn.Module):
    def __init__(self, anchors, num_classes, img_size):
        super(DecodeBox, self).__init__()
        #-----------------------------------------------------------#
        # CTK 只能取下面的一组anchor boxes
        #   13x13的特征层对应的anchor是[116,90],[156,198],[373,326]
        #   26x26的特征层对应的anchor是[30,61],[62,45],[59,119]
        #   52x52的特征层对应的anchor是[10,13],[16,30],[33,23]
        #  num_classes = 5
        #  img_size=[416,416]

        #-----------------------------------------------------------#
        self.anchors = anchors           # self.anchors 的shape是（3，2）
        self.num_anchors = len(anchors)  # self.num_anchors的数值为 3
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.img_size = img_size

    def forward(self, input):
        #-----------------------------------------------#
        #   CTk  input的 shape为（bs, 255, 13, 13）
        #-----------------------------------------------#
        batch_size = input.size(0)
        input_height = input.size(2)
        input_width = input.size(3)

        #-----------------------------------------------#
        #   CTK 当输入 img_size 为416x416 且 input_height=13的时候，stride_h = stride_w = 32  （stride_h = stride_w 只能为 32、16、8 其中的一个）
        #-----------------------------------------------#
        stride_h = self.img_size[1] / input_height
        stride_w = self.img_size[0] / input_width

        #-------------------------------------------------#
        #   此时获得的scaled_anchors大小是相对于特征层的
        #-------------------------------------------------#
        scaled_anchors = [(anchor_width / stride_w, anchor_height / stride_h) for anchor_width, anchor_height in self.anchors]

        #-----------------------------------------------#
        #   输入的input 的shape为(bs, 255, 13, 13)
        #-----------------------------------------------#
        # 等号右侧 input的shape为（bs, 255, 13, 13）
        # CTK 等号左侧 prediction的shape是（b, 3, 13,13, (1+ 4+ num_class)）
        # 为什么要将(1+ 4+ num_class)放到最后一个维度，原因是这样做有利于后续中心位置x,y和宽高w,h进行操作；假定不在最后一个维度，例如（b, 3, (1+ 4+ num_class)，13,13,）
        # 这样子对中心位置x进行sigmoid操作的时候，需要这么写：torch.sigmoid(prediction[b,3,0,13,13])
        # 这样子对中心位置y进行sigmoid操作的时候，需要这么写：torch.sigmoid(prediction[b,3,1,13,13])
        # 很难看，也不好看
        # 在view操作之后，都会配有contiguous操作，是为了让其保持连续
        prediction = input.view(batch_size, self.num_anchors, self.bbox_attrs, input_height, input_width).permute(0, 1, 3, 4, 2).contiguous()

        # 先验框的中心位置的调整参数  x和y的shape均为[4, 3, 13, 13]
        x = torch.sigmoid(prediction[..., 0])  
        y = torch.sigmoid(prediction[..., 1])

        # 先验框的宽高调整参数  w和h的shape均为[4, 3, 13, 13]
        w = prediction[..., 2]
        h = prediction[..., 3]

        # 获得置信度，是否有物体 conf的shape均为[4, 3, 13, 13]
        conf = torch.sigmoid(prediction[..., 4])

        # 种类置信度 pred_cls的shape为[4, 3, 13, 13, 20]
        pred_cls = torch.sigmoid(prediction[..., 5:])


        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor  # 如果x是在GPU上运算，则将x转为 torch.cuda.FloatTensor， 否则转为torch.FloatTensor （32位浮点型）
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor     # 如果x是在GPU上运算，则将x转为 torch.cuda.LongTensor， 否则转为torch.LongTensor （64有符号整型）



        #----------------------------------------------------------#
        #   生成网格，先验框中心，网格左上角 
        #   batch_size,3,13,13
        #----------------------------------------------------------#
        # torch.linspace生成（1，13）,   即[0,1,2,3,4,5,6,7,8,9,10,11,12]
        # 经过第一个repeat变成（13，13）
        # 经过第二个repeat变成（batch_size * self.num_anchors, 13, 13）
        # 然后利用view函数，变成（b,3,13,13）
        grid_x = torch.linspace(0, input_width - 1, input_width).repeat(input_height, 1).repeat(
            batch_size * self.num_anchors, 1, 1).view(x.shape).type(FloatTensor)

        grid_y = torch.linspace(0, input_height - 1, input_height).repeat(input_width, 1).t().repeat(
            batch_size * self.num_anchors, 1, 1).view(y.shape).type(FloatTensor)

        #----------------------------------------------------------#
        #   CTK  按照网格格式生成先验框的宽高  # 可以参考  https://blog.csdn.net/g_blink/article/details/102854188
        #   anchor_w 和 anchor_h 的shape均为 （bs,3,13,13）
        #----------------------------------------------------------#
        anchor_w = FloatTensor(scaled_anchors).index_select(1, LongTensor([0]))
        anchor_h = FloatTensor(scaled_anchors).index_select(1, LongTensor([1]))
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(w.shape)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, input_height * input_width).view(h.shape)

        #----------------------------------------------------------#
        #   利用预测结果对先验框进行调整
        #   首先调整先验框的中心，从先验框中心向右下角偏移
        #   再调整先验框的宽高。
        #----------------------------------------------------------#
        pred_boxes = FloatTensor(prediction[..., :4].shape)  # pred_boxes 的shape为（b, 3, 13, 13, 4)
        pred_boxes[..., 0] = x.data + grid_x                 # pred_boxes[..., 0] 的shape为([b, 3, 13, 13])
        pred_boxes[..., 1] = y.data + grid_y                 # pred_boxes[..., 1] 的shape为([b, 3, 13, 13])
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w    # pred_boxes[..., 2] 的shape为([b, 3, 13, 13])
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h    # pred_boxes[..., 3] 的shape为([b, 3, 13, 13])

        #----------------------------------------------------------#
        #   将输出结果调整成相对于输入图像大小
        #----------------------------------------------------------#
        _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)  # 利用torch.Tensor创建Tensor，并且将这个Tensor转换为FloatTensor类型，其中_scale的形状为torch.Size([4])
        output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,   # pred_boxes.view(batch_size, -1, 4) * _scale得到（b,3*13*13,4）
                            conf.view(batch_size, -1, 1),                  # conf.view(batch_size, -1, 1)得到（b,3*13*13,1）
                            pred_cls.view(batch_size, -1, self.num_classes)), -1) # pred_cls.view(batch_size, -1, self.num_classes))得到（b,3*13*13, num_class）
                                                                                    # output的形状为（bs,3*13*13, 4+1+20）

        return output.data

# CTK 给图像的边缘加上合适的灰条，防止图片出现失真的现象
def letterbox_image(image, size):
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

# CTK 去掉灰条
# top_ymin, top_xmin, top_ymax, top_xmax的shape均为【12，1】 model_image_size[0]和 model_image_size[1]均为416; image_shape的大小为[422 640]
# ！！！！！！！！！！！！！！！！！！！！ 这个原理非常的简单 见2021-07-19 星期一
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def yolo_correct_boxes(top, left, bottom, right, input_shape, image_shape):
    # top为左上角x1
    # left为左上角y1
    # bottom为右下角x2
    # right为右下角y2
    # input_shape = [416,416]
    # image_shape = [422, 640]

    # new_shape = [422,640] * min[(416,416)/(422,640)] = [422,640] * min(0.98, 0.65) = [422,640] * 0.65 = (274, 416)
    new_shape = image_shape*np.min(input_shape/image_shape)

    offset = (input_shape-new_shape)/2./input_shape
    scale = input_shape/new_shape


    # 中心点坐标占【416，416】的比例，相当于归一化【0，1】之间。
    box_yx = np.concatenate(((top+bottom)/2, (left+right)/2), axis=-1)/input_shape
    # 高度和宽度占【416，416】的比例，相当于归一化【0，1】之间。
    box_hw = np.concatenate((bottom-top,right-left),axis=-1)/input_shape


    box_yx = (box_yx - offset) * scale
    box_hw *= scale

    box_mins = box_yx - (box_hw / 2.)
    box_maxes = box_yx + (box_hw / 2.)

    boxes =  np.concatenate([
        box_mins[:, 0:1],
        box_mins[:, 1:2],
        box_maxes[:, 0:1],
        box_maxes[:, 1:2]
    ],axis=-1)
    boxes *= np.concatenate([image_shape, image_shape],axis=-1)

    return boxes

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
        计算IOU
    """
    if not x1y1x2y2:
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)

    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * \
                 torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)
                 
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou


def non_max_suppression(prediction, num_classes, conf_thres=0.5, nms_thres=0.4):
    #----------------------------------------------------------#
    #   prediction 的样式为 [batch_size, num_anchors, 85]，其中85中的形式，表示为cx,cy,w,h, conf, class_1, class_2, class_3... class_80
    #----------------------------------------------------------#
    box_corner = prediction.new(prediction.shape)

    # 经过下面四行，我们将cx,cy,w,h变成左上角、右下角的形式，也就是x1,y1, x2,y2
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2

    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for image_i, image_pred in enumerate(prediction):
        #----------------------------------------------------------#
        #   对种类预测部分取max。
        #   class_conf  [num_anchors, 1]    种类置信度
        #   class_pred  [num_anchors, 1]    种类
        #----------------------------------------------------------#
        # https://www.jianshu.com/p/3ed11362b54f
        # torch.max 函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引
        class_conf, class_pred = torch.max(image_pred[:, 5:5 + num_classes], 1, keepdim=True)

        #----------------------------------------------------------#
        #   利用置信度进行第一轮筛选 框的置信度(image_pred) * 种类置信度(class_conf) = 置信度
        #----------------------------------------------------------#
        # 左侧 conf_mask的shape为（10647，），他是bool类型的数据
        conf_mask = (image_pred[:, 4] * class_conf[:, 0] >= conf_thres).squeeze()


        #----------------------------------------------------------#
        #   根据置信度进行预测结果的筛选
        #----------------------------------------------------------#
        image_pred = image_pred[conf_mask] # CTK 框的置信度
        class_conf = class_conf[conf_mask] # CTK 种类置信度
        class_pred = class_pred[conf_mask] # CTK 种类
        if not image_pred.size(0):
            continue
        #-------------------------------------------------------------------------#
        #   detections  [num_anchors, 7]
        #   7的内容为：x1, y1, x2, y2, obj_conf(框的置信度), class_conf（种类的置信度）, class_pred（种类）
        #-------------------------------------------------------------------------#
        detections = torch.cat((image_pred[:, :5], class_conf.float(), class_pred.float()), 1)

        #------------------------------------------#
        #   获得预测结果中包含的所有种类
        #------------------------------------------#
        # CTK https://blog.csdn.net/yangyuwen_yang/article/details/79193770
        # 对于一维数组或者列表，unique函数去除其中重复的元素，并按元素由大到小返回一个新的无元素重复的元组或者列表
        unique_labels = detections[:, -1].cpu().unique()

        if prediction.is_cuda:
            unique_labels = unique_labels.cuda()
            detections = detections.cuda()

        for c in unique_labels:
            #------------------------------------------#
            #   获得某一类得分筛选后全部的预测结果
            #------------------------------------------#
            detections_class = detections[detections[:, -1] == c]

            #------------------------------------------#
            #   使用官方自带的非极大抑制会速度更快一些！
            #------------------------------------------#
            keep = nms(
                detections_class[:, :4],
                detections_class[:, 4] * detections_class[:, 5],
                nms_thres
            )
            max_detections = detections_class[keep]
            
            # # 按照存在物体的置信度排序
            # _, conf_sort_index = torch.sort(detections_class[:, 4]*detections_class[:, 5], descending=True)
            # detections_class = detections_class[conf_sort_index]
            # # 进行非极大抑制
            # max_detections = []
            # while detections_class.size(0):
            #     # 取出这一类置信度最高的，一步一步往下判断，判断重合程度是否大于nms_thres，如果是则去除掉
            #     max_detections.append(detections_class[0].unsqueeze(0))
            #     if len(detections_class) == 1:
            #         break
            #     ious = bbox_iou(max_detections[-1], detections_class[1:])
            #     detections_class = detections_class[1:][ious < nms_thres]
            # # 堆叠
            # max_detections = torch.cat(max_detections).data
            
            # Add max detections to outputs
            output[image_i] = max_detections if output[image_i] is None else torch.cat(
                (output[image_i], max_detections))

    return output
