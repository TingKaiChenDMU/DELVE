import torch
import torch.nn as nn
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
        #print('input.shape', input.shape)
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
        # _scale = torch.Tensor([stride_w, stride_h] * 2).type(FloatTensor)  # 利用torch.Tensor创建Tensor，并且将这个Tensor转换为FloatTensor类型，其中_scale的形状为torch.Size([4])

        # 等号右侧 pred_boxes的形式为（b,3,13,13,4）,最后4的形式为（x,y,w,h）,x，y,w,h都是以当前特征图（13*13，26*26，52*52中的某一个）为衡量
        pred_boxes = pred_boxes



        # output = torch.cat((pred_boxes.view(batch_size, -1, 4) * _scale,   # pred_boxes.view(batch_size, -1, 4) * _scale得到（b,3*13*13,4）
        #                     conf.view(batch_size, -1, 1),                  # conf.view(batch_size, -1, 1)得到（b,3*13*13,1）
        #                     pred_cls.view(batch_size, -1, self.num_classes)), -1) # pred_cls.view(batch_size, -1, self.num_classes))得到（b,3*13*13, num_class）
        #                                                                             # output的形状为（bs,3*13*13, 4+1+20）

        return pred_boxes