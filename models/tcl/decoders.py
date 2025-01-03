import torch.nn as nn

from models.builder import MODELS
from models.tcl.modules import ResConv

from clip.model import Transformer


@MODELS.register_module()
class GDecoder(nn.Module):
    def __init__(self, C, kernel_size, norm, act, double, n_layers=2, **kwargs):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(
                ResConv(
                    C, C,
                    kernel_size=kernel_size,
                    padding=kernel_size//2,
                    upsample=True,
                    norm=norm,
                    activ=act,
                    double=double,
                    gate=True
                )
            )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TextDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = Transformer(
            width=512,
            layers=2,
            heads=8,
        )

    def forward(self, middle_features):
        return self.transformer(middle_features[-1])
    
@MODELS.register_module()
class ImgFeatureDncoder(nn.Module):
    def __init__(self):
        super(ImgFeatureDncoder, self).__init__()
        self.conv1 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0)
        self.bn1 = nn.BatchNorm2d(512)
        self.relu1 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        return x

