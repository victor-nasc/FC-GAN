import torch.nn as nn

'''
VGG style discriminator
adapted from https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/archs/discriminator_arch.py
'''

class Critic(nn.Module):
    def __init__(self, channels=3, filters=64):
        super().__init__()

        self.conv0_0 = nn.Conv2d(channels, filters, 3, 1, 1, bias=True)
        self.conv0_1 = nn.Conv2d(filters, filters, 4, 2, 1, bias=False)
        self.bn0_1 = nn.BatchNorm2d(filters, affine=True)

        self.conv1_0 = nn.Conv2d(filters, filters * 2, 3, 1, 1, bias=False)
        self.bn1_0 = nn.BatchNorm2d(filters * 2, affine=True)
        self.conv1_1 = nn.Conv2d(filters * 2, filters * 2, 4, 2, 1, bias=False)
        self.bn1_1 = nn.BatchNorm2d(filters * 2, affine=True)

        self.conv2_0 = nn.Conv2d(filters * 2, filters * 4, 3, 1, 1, bias=False)
        self.bn2_0 = nn.BatchNorm2d(filters * 4, affine=True)
        self.conv2_1 = nn.Conv2d(filters * 4, filters * 4, 4, 2, 1, bias=False)
        self.bn2_1 = nn.BatchNorm2d(filters * 4, affine=True)

        self.conv3_0 = nn.Conv2d(filters * 4, filters * 8, 3, 1, 1, bias=False)
        self.bn3_0 = nn.BatchNorm2d(filters * 8, affine=True)
        self.conv3_1 = nn.Conv2d(filters * 8, filters * 8, 4, 2, 1, bias=False)
        self.bn3_1 = nn.BatchNorm2d(filters * 8, affine=True)

        self.conv4_0 = nn.Conv2d(filters * 8, filters * 8, 3, 1, 1, bias=False)
        self.bn4_0 = nn.BatchNorm2d(filters * 8, affine=True)
        self.conv4_1 = nn.Conv2d(filters * 8, filters * 8, 4, 2, 1, bias=False)
        self.bn4_1 = nn.BatchNorm2d(filters * 8, affine=True)

        self.linear1 = nn.Linear(filters * 8 * 4 * 4, 100)
        self.linear2 = nn.Linear(100, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        assert x.size(2) == 128, (f'Input size must be identical to 128, but received {x.size()}.')

        feat = self.lrelu(self.conv0_0(x))
        feat = self.lrelu(self.bn0_1(self.conv0_1(feat)))  # output spatial size: /2

        feat = self.lrelu(self.bn1_0(self.conv1_0(feat)))
        feat = self.lrelu(self.bn1_1(self.conv1_1(feat)))  # output spatial size: /4

        feat = self.lrelu(self.bn2_0(self.conv2_0(feat)))
        feat = self.lrelu(self.bn2_1(self.conv2_1(feat)))  # output spatial size: /8

        feat = self.lrelu(self.bn3_0(self.conv3_0(feat)))
        feat = self.lrelu(self.bn3_1(self.conv3_1(feat)))  # output spatial size: /16

        feat = self.lrelu(self.bn4_0(self.conv4_0(feat)))
        feat = self.lrelu(self.bn4_1(self.conv4_1(feat)))  # output spatial size: /32

        # spatial size: (4, 4)
        feat = feat.view(feat.size(0), -1)
        feat = self.lrelu(self.linear1(feat))
        out = self.linear2(feat)
        return out
