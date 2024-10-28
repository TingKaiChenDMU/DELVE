import torch
import torchvision
#boxes = torch.rand(5, 4)
boxes = torch.Tensor([[0.6, 0.4, 0.9, 0.7]])

torch.manual_seed(10)
image = torch.rand(1, 1, 10, 10)
#print(image)


# roi_bin_grid_h = 2
# roi_bin_grid_w = 2
# count = 4
per_level_feature = image

pooled_height = 3
pooled_width = 3

roi_start_w = 0.6
roi_start_h = 0.4
roi_end_w = 0.9
roi_end_h = 0.7

# bin_size_w = (0.9 - 0.6)/3
# bin_size_h = ( 0.7-0.4)/3

roi_width = max((roi_end_w - roi_start_w), 1)
roi_height = max((roi_end_h - roi_start_h), 1)
bin_size_h = roi_height / pooled_height
bin_size_w = roi_width / pooled_width


for h in range(pooled_height):
    for w in range(pooled_width):
        res = 0
        for i in range(2):
            for j in range(2):
                point_x = roi_start_w + w * bin_size_w + (bin_size_w * (1 + 2 * i) / 4)
                #print('point_x', point_x)
                point_y = roi_start_h + h * bin_size_h + (bin_size_h * (1 + 2 * j) / 4)
                #print('point_y', point_y)

                val0 = per_level_feature[0][0][int(point_y)][int(point_x)]
                #print('val0', val0)
                val1 = per_level_feature[0][0][int(point_y)][int(point_x) + 1]
                #print('val1', val1)
                val2 = per_level_feature[0][0][int(point_y) + 1][int(point_x)]
                #print('val2', val2)
                val3 = per_level_feature[0][0][int(point_y) + 1][int(point_x) + 1]
                #print('val3', val3)

                x1 = point_x - int(point_x)
                #print('x1', x1)
                x2 = int(point_x) + 1 - point_x
                #print('x2', x2)
                y1 = point_y - int(point_y)
               # print('y1', y1)
                y2 = int(point_y) + 1 - point_y
                #print('y2', y2)

                area0 = x1 * y1
                area1 = x2 * y1
                area2 = x1 * y2
                area3 = x2 * y2

                res += (val0 * area3 + val1 * area2 + val2 * area1 + val3 * area0)

        print(res / 4)

pooled_regions = torchvision.ops.roi_align(image, [boxes], output_size=(pooled_height, pooled_width), sampling_ratio=2)
# check the size
print(pooled_regions)


#print((0.7-0.4)/3/4*3)