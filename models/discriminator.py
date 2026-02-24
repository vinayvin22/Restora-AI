import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(CNNBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=False, padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super(Discriminator, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels * 2, features[0], 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
        )

        layers = []
        in_ch = features[0]
        for feature in features[1:]:
            layers.append(CNNBlock(in_ch, feature, stride=1 if feature == features[-1] else 2))
            in_ch = feature

        layers.append(nn.Conv2d(in_ch, 1, 4, 1, 1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
                if m.weight is not None:
                    nn.init.normal_(m.weight.data, 1.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x, y):
        # Pix2Pix Discriminator concatenates input and target (distorted and restored/clean)
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    y = torch.randn((1, 3, 256, 256))
    model = Discriminator(in_channels=3)
    preds = model(x, y)
    print(preds.shape)
