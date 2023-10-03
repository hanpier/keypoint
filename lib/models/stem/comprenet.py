import torch
import torch.nn as nn
from torchsummary import summary

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, padding=0)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.batch_norm(x)
        residual = self.shortcut(residual)
        x += residual
        x = nn.functional.relu(x)
        return x


class CompressingNetwork(nn.Module):
    def __init__(self, num_residual_blocks=1, num_filters=16):
        super(CompressingNetwork, self).__init__()

        self.preprocess_conv = nn.Conv2d(3, num_filters, kernel_size=3, padding=1)
        self.conv7x7 = nn.Conv2d(num_filters, num_filters, kernel_size=7, stride=2, padding=3)

        self.residual_blocks = nn.ModuleList([
            ResidualBlock(num_filters, num_filters) for _ in range(num_residual_blocks)
        ])

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # m.weight.data.normal_(0, 0.01)
                torch.nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
                # torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.preprocess_conv(x)
        x = self.conv7x7(x)
        for block in self.residual_blocks:
            x = block(x)
        return x


# Create an instance of the CompressingNetwork
# num_residual_blocks = 1
# num_filters = 3
# compressor = CompressingNetwork(num_residual_blocks, num_filters).cuda()
#
# # Print model summary
# summary(compressor,(3, 704, 1280))
