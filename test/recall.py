import torch

class NMB(torch.nn.Module):
    def __init__(self):
        super(NMB, self).__init__()
        self.coefficient = torch.nn.Parameter(torch.Tensor([0.0]))


        print('NMB初始化')

    def forward(self, x):
        xxx = self.coefficient * x



        print('coefficient值为:', self.coefficient.item())

        return xxx
