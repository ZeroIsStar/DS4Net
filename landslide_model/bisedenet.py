import torch
import torch.nn as nn
import torch.nn.functional as F
#  BisDeNet: A New Lightweight Deep Learning-Based Framework for Efficient Landslide Detection 2024 改原版模型是keras


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, bias=False):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )


class AttentionRefinementModule(nn.Module):
    """ARM: global avg pool → 1×1 conv → sigmoid → channel-wise mul"""
    def __init__(self, in_ch):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_ch, in_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_ch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        att = self.sigmoid(self.bn(self.conv(self.gap(x))))
        return x * att


class FeatureFusionModule(nn.Module):
    """FFM: 拼接 → 1×1 → BN/ReLU → 3×3 → BN/ReLU → 1×1 → dropout → 1×1 → n_classes"""
    def __init__(self, sp_ch, cp_ch, n_classes, dropout=0.1):
        super().__init__()
        self.conv1 = ConvBNReLU(sp_ch + cp_ch, 256, 1, 1, 0)
        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout),
            nn.Conv2d(256, n_classes, 1)
        )

    def forward(self, sp, cp):
        fuse = torch.cat([sp, cp], dim=1)
        fuse = self.conv1(fuse)
        return self.conv2(fuse)

class DenseNet(nn.Module):
    """仅作为 Context Path 的主干，返回 y2_16, y2_32, y2_Global"""
    def __init__(self, growth_rate=12):
        super().__init__()
        self.stem = ConvBNReLU(14, growth_rate*2, 3, 1, 1)

        self.layer1 = self._make_layer(growth_rate*2, 3, growth_rate, down=True)   # 64
        self.layer2 = self._make_layer(self.layer1[-1].out_ch, 3, growth_rate, down=True)  # 32
        self.layer3 = self._make_layer(self.layer2[-1].out_ch, 3, growth_rate, down=True)  # 16
        self.layer4 = self._make_layer(self.layer3[-1].out_ch, 3, growth_rate, down=True)  # 8
        self.y2_16_ch = self.layer4[-1].out_ch

        self.layer5 = self._make_layer(self.y2_16_ch, 6, growth_rate, down=True)   # 4
        self.y2_32_ch = self.layer5[-1].out_ch

    def _make_layer(self, in_ch, num_layers, growth_rate, down=False):
        layers = []
        for i in range(num_layers):
            layers.append(DenseLayer(in_ch + i*growth_rate, growth_rate))
        out_ch = in_ch + num_layers*growth_rate
        if down:
            layers.append(TransitionLayer(out_ch))
            out_ch = int(out_ch * 0.5)
        # 用 nn.Sequential 保存，并挂 out_ch 方便后续拿通道数
        seq = nn.Sequential(*layers)
        seq.out_ch = out_ch
        return seq

    def forward(self, x):
        x = self.stem(x)            # 128
        x = self.layer1(x)          # 64
        x = self.layer2(x)          # 32
        x = self.layer3(x)          # 16
        y2_16 = self.layer4(x)      # 8×8

        y2_32 = self.layer5(y2_16)  # 4×4
        y2_Global = F.adaptive_avg_pool2d(y2_32, 1).flatten(1)

        return y2_16, y2_32, y2_Global


class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 4*growth_rate, 1, bias=False),
            nn.BatchNorm2d(4*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4*growth_rate, growth_rate, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = int(in_ch * 0.5)
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2, 2)
        )
        self.out_ch = out_ch

    def forward(self, x):
        return self.net(x)


class BiSeNet_DenseNet6(nn.Module):
    def __init__(self,in_channels, n_classes):
        super().__init__()
        # === Spatial Path ===
        self.spatial = nn.Sequential(
            ConvBNReLU(in_channels,  64, 3, 2, 1),   # 128 -> 64
            ConvBNReLU(64, 128, 3, 2, 1),   # 64  -> 32
            ConvBNReLU(128, 256, 3, 2, 1)   # 32  -> 16
        )

        # === Context Path ===
        self.backbone = DenseNet()          # 返回 y2_16, y2_32, y2_Global
        self.arm1 = AttentionRefinementModule(self.backbone.y2_16_ch)
        self.arm2 = AttentionRefinementModule(self.backbone.y2_32_ch)

        # === Feature Fusion ===
        self.ffm = FeatureFusionModule(sp_ch=256,
                                       cp_ch=self.backbone.y2_16_ch + self.backbone.y2_32_ch,
                                       n_classes=n_classes)

    def forward(self, x):
        # --- Spatial Path ---
        sp = self.spatial(x)                       # 16×16, 256

        # --- Context Path ---
        y2_16, y2_32, y2_Global = self.backbone(x) # 8×8, 4×4, (B,C)

        # ARM
        arm1 = self.arm1(y2_16)                    # 8×8
        arm2 = self.arm2(y2_32) * y2_Global.view(-1, y2_32.size(1), 1, 1)

        # Upsample 统一到 16×16
        arm1 = F.interpolate(arm1, scale_factor=2,  mode='bilinear', align_corners=False)
        arm2 = F.interpolate(arm2, scale_factor=4,  mode='bilinear', align_corners=False)
        cp = torch.cat([arm1, arm2], dim=1)        # 16×16

        # FFM
        out = self.ffm(sp, cp)                     # 16×16, n_classes

        # Final 8× upsample -> 128×128
        out = F.interpolate(out, scale_factor=8, mode='bilinear', align_corners=False)
        return out

if __name__ == "__main__":
    from thop import profile
    import time
    model = BiSeNet_DenseNet6(14,n_classes=2).cuda()
    x = torch.rand(1, 14, 128, 128).cuda()
    flops, params = profile(model, inputs=(x,))
    model.eval()
    with torch.no_grad():
        num_runs = 1000
        total_time = 0
        # 多次推理，计算平均推理时间
        for _ in range(num_runs):
            start_time = time.time()
            results = model(x)
            end_time = time.time()
            total_time += (end_time - start_time)
            # 计算平均推理时间
        avg_inference_time = total_time / num_runs
            # 计算FPS
        fps = 1 / avg_inference_time
        print(f"FPS: {fps:.2f} frames per second")
        print(f'FLOPs: {flops / 1e9}G')
        print(f'params: {params / 1e6}M')
        print(model(x).shape)
