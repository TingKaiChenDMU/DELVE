Config = \
{   
    #-------------------------------------------------------------#
    #   训练前一定要修改classes参数
    #   anchors可以不修改，因为anchors的通用性较大
    #   而且大中小的设置非常符合yolo的特征层情况
    #-------------------------------------------------------------#
    "yolo": {
        "anchors": [[[116, 90], [156, 198], [373, 326]],
                    [[30, 61], [62, 45], [59, 119]],
                    [[10, 13], [16, 30], [33, 23]]],
        "classes": 5,
    },
    #-------------------------------------------------------------#
    #   img_h和img_w可以修改成608x608
    #-------------------------------------------------------------#
    "img_h": 416,
    "img_w": 416,
}
