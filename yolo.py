#-------------------------------------#
#       创建YOLO类
#-------------------------------------#
import colorsys
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.autograd import Variable

from nets.yolo3 import YoloBody
from utils.config import Config
from utils.utils import (DecodeBox, bbox_iou, letterbox_image,
                         non_max_suppression, yolo_correct_boxes)
import os
os.environ['DISPLAY'] = ':0'

#--------------------------------------------#
#   使用自己训练好的模型预测需要修改2个参数
#   model_path和classes_path都需要修改！
#   如果出现shape不匹配，一定要注意
#   训练时的model_path和classes_path参数的修改
#--------------------------------------------#
class YOLO(object):
    print('进入YOLO啦')
    _defaults = {
        "model_path"        : 'logs/Epoch93-Total_Loss32.8033-Val_Loss214.2362.pth',
        # "model_path"        : 'logs/yolo_weights.pth',
        "classes_path"      : 'model_data/uod_classes.txt',
        # "classes_path"      : 'model_data/coco_classes.txt',
        "model_image_size"  : (608, 608, 3),
        "confidence"        : 0.5, # 在进行NMS时候用得到，confidence低于0.5，则认为预测的不准确，舍掉。
        "iou"               : 0.3, # 在进行NMS时候用得到，IOU大于0.3，则认为是预测的是同一个物体。
        "cuda"              : True
    }
    print(_defaults)
    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        print('进入YOLO的init函数啦')
        self.__dict__.update(self._defaults)  # CTK 使用本行后，_defaults中所包含的内容，直接使用self.【属性】即可进行调用
        self.class_names = self._get_class()   # 获得所有类别的名字
        self.config = Config
        self.generate()


    #---------------------------------------------------#
    #   CTK  获得所有的分类 名字
    #---------------------------------------------------#
    def _get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        # https://m.runoob.com/python/att-string-strip.html
        # # Python strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        # # 注意：该方法只能删除开头或是结尾的字符，不能删除中间部分的字符。
        class_names = [c.strip() for c in class_names]
        return class_names

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self):
        # 因为 config里面虽然设定了类别的数量，但是有时候我们在classes_path那里更改了类别的数量，这里先进行更新类别的数量
        self.config["yolo"]["classes"] = len(self.class_names)  # 获得共有类别的数量 20类
        #---------------------------------------------------#
        #   建立yolov3模型
        #---------------------------------------------------#
        self.net = YoloBody(self.config)

        #---------------------------------------------------#
        #   载入yolov3模型的权重
        #---------------------------------------------------#
        print('Loading weights into state dict...')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        state_dict = torch.load(self.model_path, map_location=device)
        self.net.load_state_dict(state_dict)
        self.net = self.net.eval()

        if self.cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)  # CTK 采用数据并行的方式来处理，可以设置多卡并行处理
            self.net = self.net.cuda()

        #---------------------------------------------------#
        #   建立三个特征层解码用的工具
        #---------------------------------------------------#
        self.yolo_decodes = []
        for i in range(3):
            # yolo_decodes[0]的形式为（b,3*13*13, 4+1+20），对应最大的anchors
            # yolo_decodes[1]的形式为（b,3*26*26, 4+1+20），对应中等的anchors
            # yolo_decodes[2]的形式为（b,3*52*52, 4+1+20），对应最小的anchors
            self.yolo_decodes.append(DecodeBox(self.config["yolo"]["anchors"][i], self.config["yolo"]["classes"], (self.model_image_size[1], self.model_image_size[0])))

        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        # 画框设置不同的颜色
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image):
        image_shape = np.array(np.shape(image)[0:2])

        #---------------------------------------------------------#
        #   CTK  给图像增加灰条，实现不失真的resize
        #---------------------------------------------------------#
        crop_img = np.array(letterbox_image(image, (self.model_image_size[1],self.model_image_size[0])))
        # 将图片转换为np.float32的形式，并且除以255代表归一化操作
        photo = np.array(crop_img, dtype = np.float32) / 255.0
        photo = np.transpose(photo, (2, 0, 1))
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        images = [photo]

        with torch.no_grad():
            images = torch.from_numpy(np.asarray(images))  # 将图片从numpy转为tensor形式，才能够网络使用
            if self.cuda:
                images = images.cuda()

            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            # 对于coco数据集，并且输入图片的大小是416*416，print(outputs[0].shape)的结果为torch.Size([1, 255, 13, 13])
            # 对于coco数据集，并且输入图片的大小是416*416，print(outputs[1].shape)的结果为torch.Size([1, 255, 26, 26])
            # 对于coco数据集，并且输入图片的大小是416*416，print(outputs[2].shape)的结果为torch.Size([1, 255, 52, 52])
            outputs = self.net(images)

            output_list = []
            for i in range(3):
                # output_list[0]的形式为（bs,3*13*13, 4+1+20）, 最后得到的4（也就是x,y,w,h）是相对于416*416大小的。
                # output_list[1]的形式为（bs,3*26*26, 4+1+20）, 最后得到的4（也就是x,y,w,h）是相对于416*416大小的。
                # output_list[2]的形式为（bs,3*52*52, 4+1+20）, 最后得到的4（也就是x,y,w,h）是相对于416*416大小的。
                output_list.append(self.yolo_decodes[i](outputs[i]))

            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            # # output的形状为 （b, 3*13*13 + 3*26*26 + 3*52*52, 4+1+20）
            output = torch.cat(output_list, 1)

            # 左侧 batch_detections的形式为【XXXX，7】，其中7的内容表示为：x1, y1, x2, y2, obj_conf(框的置信度), class_conf（种类的置信度）, class_pred（种类）
            batch_detections = non_max_suppression(output, self.config["yolo"]["classes"],
                                                    conf_thres=self.confidence,
                                                    nms_thres=self.iou)

            #---------------------------------------------------------#
            #   如果没有检测出物体，返回原图
            #---------------------------------------------------------#
            try :
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return image

            #---------------------------------------------------------#
            #   对预测框进行得分筛选
            #---------------------------------------------------------#
            # batch_detections的形式为 x1, y1, x2, y2, obj_conf(框的置信度), class_conf（种类的置信度）, class_pred（种类）
            # 有一句话说：框里面包含某物体的置信度为0.9，其实这里面有两层含义，第一层是框位置的置信度，第二层是某物体的置信度，两者相乘则为，包含框框中包含某物体的置信度为0.9
            # top_index的值为Bool类型，其值全为true.
            # top_index的shape为(AAAA,),其中AAAA的值，我也不知道，其跟经过NMS操作所保留下来的框的数量是一致的。
            top_index = batch_detections[:,4] * batch_detections[:,5] > self.confidence


            # # batch_detections的形式为 x1, y1, x2, y2, obj_conf(框的置信度), class_conf（种类的置信度）, class_pred（种类）
            # 得到最后的置信度 = 框框的置信度 * 种类的置信度
            # 其shape为(AAAA,),其中AAAA的值，我也不知道，其跟经过NMS操作所保留下来的框的数量是一致的。
            top_conf = batch_detections[top_index,4]*batch_detections[top_index,5]


            # 取出来种类，是海参？海胆？海星？还是贝壳？当然了，这里用0,1,2,3等数字来进行表示类别，同一个类别的数字是一致的。其shape为(AAAA,),其中AAAA的值，我也不知道，其跟经过NMS操作所保留下来的框的数量是一致的。
            top_label = np.array(batch_detections[top_index,-1],np.int32)

            # 取出来框框的位置。(AAAA,4),其中AAAA的值，我也不知道，其跟经过NMS操作所保留下来的框的数量是一致的。
            top_bboxes = np.array(batch_detections[top_index,:4])

            # top_xmin的shape为【12，1】；
            # top_ymin的shape为【12，1】；
            # top_xmax的shape为【12，1】
            # top_ymax的shape为【12，1】
            top_xmin, top_ymin, top_xmax, top_ymax = np.expand_dims(top_bboxes[:,0],-1),np.expand_dims(top_bboxes[:,1],-1),np.expand_dims(top_bboxes[:,2],-1),np.expand_dims(top_bboxes[:,3],-1)

            #-----------------------------------------------------------------#
            #   在图像传入网络预测前会进行letterbox_image给图像周围添加灰条
            #   因此生成的top_bboxes是相对于有灰条的图像的
            #   我们需要对其进行修改，去除灰条的部分。
            #-----------------------------------------------------------------#
            # CTK去掉灰条，从而所得到预测框框的坐标信息，是相对于原始图片大小的，即，[422 640]
            # top_ymin, top_xmin, top_ymax, top_xmax的shape均为【12，1】 model_image_size[0]和 model_image_size[1]均为416; image_shape的大小为[422 640]
            boxes = yolo_correct_boxes(top_ymin,top_xmin,top_ymax,top_xmax,np.array([self.model_image_size[0],self.model_image_size[1]]), image_shape)

        font = ImageFont.truetype(font='model_data/simhei.ttf',size=np.floor(3e-2 * np.shape(image)[1] + 0.5).astype('int32'))

        # 对于img/test.jpg来说，np.shape(image)为[422 640,3]
        # self.model_image_size[0]为416
        thickness = max((np.shape(image)[0] + np.shape(image)[1]) // self.model_image_size[0], 1)

        # 将类别、得分画在图片上面
        # 取出来种类，是海参？海胆？海星？还是贝壳？当然了，这里用0,1,2,3等数字来进行表示类别，同一个类别的数字是一致的。其shape为(AAAA,),其中AAAA的值，我也不知道，其跟经过NMS操作所保留下来的框的数量是一致的。
        for i, c in enumerate(top_label):

            # 获得类别（echinus, scallop, holothurian, starfish）
            predicted_class = self.class_names[c]

            # 得到最后的置信度 = 框框的置信度 * 种类的置信度  # 其shape为(AAAA,),其中AAAA的值，我也不知道，其跟经过NMS操作所保留下来的框的数量是一致的。
            score = top_conf[i]

            # 左上角x1 top, 左上角y1 left, 右下角x2 bottom, 右下角y2 right.
            top, left, bottom, right = boxes[i]
            top = top - 5
            left = left - 5
            bottom = bottom + 5
            right = right + 5

            # 将 top、left，bottom，right 与 0或者边界值进行比较，防止画出来的框框溢出图像自身，并且最后转换为int32类型。
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(np.shape(image)[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(np.shape(image)[1], np.floor(right + 0.5).astype('int32'))

            # 画框框
            label = '{} {:.2f}'.format(predicted_class, score) # 将类别和置信度放置到label变量
            draw = ImageDraw.Draw(image) # 画出来图像

            # https://blog.csdn.net/m0_46653437/article/details/112048730
            # 返回用指定字体对象显示给定字符串所需要的图像尺寸，CTK，这也就是说这里系统会根据label以及font的形式来自动的计算所需要的标签的宽度和高度大小，例如（220,40）
            label_size = draw.textsize(label, font) #

            label = label.encode('utf-8')
            print(label, top, left, bottom, right)

            # 所绘制label标签（类别名：置信度得分）左上角坐标。
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                #  绘制检测框
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[self.class_names.index(predicted_class)])
            # 绘制label标签的框框。
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[self.class_names.index(predicted_class)])
            # 在上面所画label标签的框框里面写上 “类别名：置信度得分”
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw
        return image

