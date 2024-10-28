# -*- coding: utf-8 -*-
import torch
from torch.autograd import Variable
import torch.optim as optim

from recall import NMB

# https://www.jb51.net/article/167888.htm
x = Variable(torch.FloatTensor([1, 2, 3])).cuda()
y = Variable(torch.FloatTensor([4, 5])).cuda()


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(3, 5)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(5, 2)


    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)

        return x

# 定义模型
model_1 = MLP().cuda()
model_2 = NMB().cuda()

print('----------------开始打印参数----------------')
for _, param in enumerate(model_2.named_parameters()):
    print(param[0])
    print(param[1])
print('----------------打印参数结束-----------------')


loss_fn = torch.nn.MSELoss(size_average=False)
optimizer = optim.Adam([{'params':model_1.parameters()}, {'params':model_2.parameters()}], lr=0.001)
#optimizer = optim.Adam(model_1.parameters(), lr=0.001)

for t in range(3):
    y_pred_1 = model_1(x)
    y_pred_2 = model_2(y_pred_1)

    loss = loss_fn(y_pred_2, y)
    print('==============第%s迭代，训练损失为：%s'%(t, loss.data.item()))

    model_1.zero_grad()
    model_2.zero_grad()

    loss.backward()
    optimizer.step()


# 输入数据
#data = torch.randn((8,3)).cuda()

# 输出数据
#output1 = model_1(data)
#print('output:', output1)

# 输出数据
#output2 = model_2(output1)
#print('output:', output2)


# print('----------------开始打印 model_1 参数----------------\n')
# for _, param in enumerate(model_1.named_parameters()):
#     print(param[0])
#     #print(param[1])
# print('----------------结束打印 model_1 参数结束-----------------\n')
#
# print('----------------开始打印 model_2 参数----------------\n')
# for _, param in enumerate(model_2.named_parameters()):
#     print(param[0])
#     #print(param[1])
# print('----------------结束打印 model_2 参数结束-----------------\n')



#print(model_1(x))

# print(model.state_dict().keys())
# for i, j in model.named_parameters():
#     print(i)
#     print(j)