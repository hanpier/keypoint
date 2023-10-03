import torch
import torch.nn as nn
class Unet_Head(nn.Module):
    def __init__(self, num_keypoint=16, mode='train'):
        super(Unet_Head, self).__init__()

        self.mode = mode

        self.init_weight()

        self.header_heatmaps = nn.Sequential(*[
            nn.Conv2d(16, num_keypoint, 1, 1, 0, bias=True),
            nn.Sigmoid()
        ])

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.01)
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):

        res = []
        h1 = self.header_heatmaps(x)
        res = h1
        return res
