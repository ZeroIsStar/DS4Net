# MFFENet and ADANet: a robust deep transfer learning method and its application in high precision and fast cross-scene recognition of earthquake-induced landslides
# 2022
import torch
import torch.nn as nn
import torch.nn.functional as F
from landslide_model.modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from landslide_model.modeling.aspp import build_aspp
from landslide_model.modeling.denseGates_decoder import build_decoder
from landslide_model.modeling.backbone import build_backbone


class DeepLab(nn.Module):
    def __init__(self, backbone='resnet', output_stride=16, num_classes=2,
                 sync_bn=True, freeze_bn=False,in_channels = 14):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm)
        self.backbone.conv1 = nn.Conv2d(in_channels,  64,kernel_size=(7, 7), stride=(2, 2),padding=(3, 3),bias=False)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        if freeze_bn:
            self.freeze_bn()

    def forward(self, input):
        x, low_level_feat1, low_level_feat2, low_level_feat3 = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat1, low_level_feat2, low_level_feat3)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)

        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p


if __name__ == "__main__":
    from thop import profile
    import time
    model = DeepLab(backbone='resnet', output_stride=16, in_channels=3).cuda()
    model.eval()
    x = torch.rand(1, 3, 128, 128).cuda()
    output = model(x)
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
