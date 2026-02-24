import torch
import sys
from generator import Generator

model = Generator(in_channels=3, features=64)
x = torch.randn((1, 3, 256, 256))

print("Step 1: initial_down")
d1 = model.initial_down(x)
print(f"d1: {d1.shape}")

print("Step 2: down1-6")
d2 = model.down1(d1)
print(f"d2: {d2.shape}")
d3 = model.down2(d2)
print(f"d3: {d3.shape}")
d4 = model.down3(d3)
print(f"d4: {d4.shape}")
d5 = model.down4(d4)
print(f"d5: {d5.shape}")
d6 = model.down5(d5)
print(f"d6: {d6.shape}")
d7 = model.down6(d6)
print(f"d7: {d7.shape}")

print("Step 3: bottleneck")
bn = model.bottleneck(d7)
print(f"bn: {bn.shape}")

print("Step 4: up1-7")
u1 = model.up1(bn)
print(f"u1: {u1.shape}")
u2 = model.up2(torch.cat([u1, d7], 1))
print(f"u2: {u2.shape}")
u3 = model.up3(torch.cat([u2, d6], 1))
print(f"u3: {u3.shape}")
u4 = model.up4(torch.cat([u3, d5], 1))
print(f"u4: {u4.shape}")
u5 = model.up5(torch.cat([u4, d4], 1))
print(f"u5: {u5.shape}")
u6 = model.up6(torch.cat([u5, d3], 1))
print(f"u6: {u6.shape}")
u7 = model.up7(torch.cat([u6, d2], 1))
print(f"u7: {u7.shape}")

print("Step 5: final_up")
out = model.final_up(torch.cat([u7, d1], 1))
print(f"out: {out.shape}")
