import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, down=True, dropout=False, use_dilated=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False, padding_mode="reflect"),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True)
            )
        else:
            # Using Bilinear Upsampling followed by Conv for stability and smoothness
            layers = [
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
                nn.InstanceNorm2d(out_ch, affine=True),
                nn.ReLU(inplace=True)
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)

class Generator(nn.Module):
    """
    Robust U-Net Restoration Network with Dilated Bottleneck for Inpainting.
    Optimized for clarity, structural fidelity, and filling missing regions.
    """
    def __init__(self):
        super().__init__()
        
        # Encoder: 256 -> 128 -> 64 -> 32 -> 16 -> 8
        self.e1 = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True)
        ) # 128
        self.e2 = UNetBlock(64, 128, down=True)  # 64
        self.e3 = UNetBlock(128, 256, down=True) # 32
        self.e4 = UNetBlock(256, 512, down=True) # 16
        self.e5 = UNetBlock(512, 512, down=True) # 8
        
        # Bottleneck: Deep and Dilated to fill blank regions (inpainting core)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, padding=2, dilation=2, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=4, dilation=4, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, 1, padding=1, bias=False),
            nn.InstanceNorm2d(512, affine=True),
            nn.ReLU(inplace=True)
        )
        
        # Decoder: 8 -> 16 -> 32 -> 64 -> 128 -> 256
        self.d1 = UNetBlock(512, 512, down=False, dropout=True) # 16
        self.d2 = UNetBlock(1024, 256, down=False, dropout=True) # 32
        self.d3 = UNetBlock(512, 128, down=False) # 64
        self.d4 = UNetBlock(256, 64, down=False) # 128
        
        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 256
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        
        b = self.bottleneck(e5)
        
        d1 = self.d1(b)
        d2 = self.d2(torch.cat([d1, e4], 1))
        d3 = self.d3(torch.cat([d2, e3], 1))
        d4 = self.d4(torch.cat([d3, e2], 1))
        
        return self.final(torch.cat([d4, e1], 1))

if __name__ == "__main__":
    x = torch.randn((1, 3, 256, 256))
    model = Generator()
    print(f"Output shape: {model(x).shape}")
