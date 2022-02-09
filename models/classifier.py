import torch.nn as nn

class MyClfModel(nn.Module):

    def __init__(self):
        super(MyClfModel, self).__init__()

        self.clf = nn.Sequential(nn.Linear(256, 15))

    def forward(self, x):
        x = self.clf(x)
        return x