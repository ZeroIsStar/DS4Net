import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref


class MambaLayer(nn.Module):
    def __init__(self, in_chs=512, dim=128, d_state=16):
        super().__init__()
        self.PAPPM = PAPPM(in_chs,dim,outplanes=in_chs)
        self.att = SS2D(d_model=in_chs, d_state=d_state)

    def forward(self, x): # B, C, H, W
        x = self.PAPPM(x).permute(0,2,3,1)
        x = self.att(x).permute(0,3,1,2)
        return x

    def generate_arithmetic_sequence(self, start, stop, step):
        sequence = []
        for i in range(start, stop, step):
            sequence.append(i)
        return sequence

class ConvFFN(nn.Module):
    def __init__(self, in_ch=128, hidden_ch=128, out_ch=64, drop=0.05):
        super(ConvFFN, self).__init__()
        self.fc1 = nn.Conv2d(in_ch, hidden_ch, kernel_size=1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x+shortcut

class SpecSpatialStem(nn.Module):
    def __init__(self, in_ch=14, base_ch=64):
        super().__init__()
        # 光谱卷积（深度可分离）
        self.spat_c = in_ch-2
        self.spec_conv1 = nn.Conv3d(1, base_ch//2, (3,1,1), padding=(1,0,0), bias=False)
        self.spec_conv2 = nn.Conv3d(base_ch//2, base_ch//2, (3,1,1), padding=(1,0,0), bias=False)
        self.bn1 = nn.BatchNorm3d(base_ch//2)
        self.bn2 = nn.BatchNorm3d(base_ch//2)
        self.fuse_channels = nn.Conv3d(base_ch//2, base_ch//2, (self.spat_c, 1, 1))
        self.DEM = nn.Sequential(
            nn.Conv2d(2, base_ch//2, (3, 3), padding=(1, 1)),
            nn.BatchNorm2d(base_ch//2),
            nn.ReLU()
        )
        self.spa_conv = nn.Sequential(
            nn.Conv2d(base_ch,base_ch,kernel_size=3,padding=1,bias=False),
            nn.BatchNorm2d(base_ch),
            nn.ReLU()
        )

        # 2-D 精炼
    def forward(self, x):
        spat = x[:,:12,:,:]
        DEM = x[:,12:,:,:]
        spat = spat.unsqueeze(1)                       # [B,1,C,H,W]
        spat = self.spec_conv1(spat)
        spat = self.bn1(spat)
        spat = F.relu(spat)
        spat = self.spec_conv2(spat)
        spat = self.bn2(spat)
        spat = F.relu(spat)
        spat = self.fuse_channels(spat).squeeze(2)  # [B,64,H,W]
        DEM = self.DEM(DEM)
        x = self.spa_conv(torch.cat([spat, DEM], dim=1))
        return x


class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            # d_state="auto", # 20240109
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))  # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, in_channels=14):
        super().__init__()
        self.stem = SpecSpatialStem(in_channels, growth_rate*2)
        self.layer1 = self._make_layer(growth_rate*2, 3, growth_rate, down=True)   # 64
        self.layer2 = self._make_layer(self.layer1[-1].out_ch, 3, growth_rate, down=True)  # 32
        self.layer3 = self._make_layer(self.layer2[-1].out_ch, 3, growth_rate, down=True)  # 16
        self.layer4 = self._make_layer(self.layer3[-1].out_ch, 3, growth_rate, down=True)  # 8

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
        stem = self.stem(x)            # 128
        x_1 = self.layer1(stem)          # 64
        x_2 = self.layer2(x_1)          # 32
        x_3 = self.layer3(x_2)          # 16
        return stem,x_1,x_2,x_3


class DenseLayer(nn.Module):
    def __init__(self, in_ch, growth_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 2*growth_rate, 1, bias=False),
            nn.BatchNorm2d(2*growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*growth_rate, growth_rate, 3, 1, 1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.net(x)], 1)


class TransitionLayer(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        out_ch = int(64)
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.AvgPool2d(2, 2)
        )
        self.out_ch = out_ch

    def forward(self, x):
        return self.net(x)

class PAPPM(nn.Module):
    def __init__(self, inplanes, branch_planes, outplanes, BatchNorm=nn.BatchNorm2d):
        super(PAPPM, self).__init__()

        self.scale1 = nn.Sequential(nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
                                    BatchNorm(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale2 = nn.Sequential(nn.AvgPool2d(kernel_size=5, stride=2, padding=2),
                                    BatchNorm(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale3 = nn.Sequential(nn.AvgPool2d(kernel_size=9, stride=4, padding=4),
                                    BatchNorm(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )
        self.scale4 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                    BatchNorm(inplanes),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
                                    )

        self.scale0 = nn.Sequential(
            BatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, branch_planes, kernel_size=1, bias=False),
        )

        self.compression = nn.Sequential(
            BatchNorm(branch_planes * 5),
            nn.ReLU(inplace=True),
            nn.Conv2d(branch_planes * 5, outplanes, kernel_size=3,padding=1, bias=False),
        )

        self.shortcut = nn.Sequential(
            BatchNorm(inplanes),
            nn.ReLU(inplace=True),
            nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=False),
        )

    def forward(self, x):
        width = x.shape[-1]
        height = x.shape[-2]
        scale_list = []

        x_ = self.scale0(x)
        scale_list.append(F.interpolate(self.scale1(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale2(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale3(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)
        scale_list.append(F.interpolate(self.scale4(x), size=[height, width],
                                        mode='bilinear', align_corners=False) + x_)

        scale_out = torch.cat(scale_list, 1)

        out = self.compression(torch.cat([x_, scale_out], 1)) + self.shortcut(x)
        return out


class Block(nn.Module):
    def __init__(self, in_chs=64, dim=128, hidden_ch=128, out_ch=64, drop=0.1, d_state=16):
        super(Block, self).__init__()
        self.mamba = MambaLayer(in_chs=in_chs, dim=dim, d_state=d_state)
        self.conv_ffn = ConvFFN(in_ch=in_chs, hidden_ch=hidden_ch, out_ch=out_ch, drop=drop)

    def forward(self, x):
        x = self.mamba(x)
        x = self.conv_ffn(x)
        return x


class PAG(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(PAG, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1,bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.f_y = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )
        self.f_x = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1, bias=False),
        )

    def forward(self, low_stage, high_stage):
        h, w = low_stage.size(2), low_stage.size(3)
        high_stage = F.interpolate(input=high_stage, size=(h, w), mode='bilinear', align_corners=False)
        if low_stage.size(1) != high_stage.size(1):
            low_stage = self.conv(low_stage)
        sim_map = torch.sigmoid(self.f_x(high_stage) * self.f_y(low_stage))
        x = (1 - sim_map) * low_stage + sim_map * high_stage
        return x


class FA(nn.Module):
    def __init__(self, features,concat_nums):
        super(FA, self).__init__()
        # 用 3×3 卷积生成 2 通道偏移量
        self.delta_gen = nn.Sequential(
            nn.Conv2d(features * concat_nums, features, kernel_size=1, bias=False),
            nn.BatchNorm2d(features),
            nn.Conv2d(features, 2 * (concat_nums),   # 每组特征需要 2 个通道偏移
                      kernel_size=3, padding=1, bias=False)
        )
        # 初始化偏移量为 0
        self.delta_gen[2].weight.data.zero_()

        self.num_feats = None   # 由 forward 动态决定

    def bilinear_interpolate_torch_gridsample(self, input, size, delta):
        # delta: (B,2,H,W)
        out_h, out_w = size
        b, c, h, w = input.shape
        # 生成 [-1,1] 网格
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, out_h, device=input.device),
            torch.linspace(-1, 1, out_w, device=input.device),
            indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).repeat(b, 1, 1, 1)
        # 归一化偏移并叠加
        norm = torch.tensor([w, h], device=input.device).view(1, 1, 1, 2)
        grid = grid + delta.permute(0, 2, 3, 1) / norm
        return F.grid_sample(input, grid, mode='bilinear', align_corners=False)

    def forward(self, *feats):
        """
        feats: 任意数量的特征图，按“分辨率由低到高”或任意顺序传入，
               最终全部对齐到 feats[0] 的大小。
        """
        low_stage = feats[0]
        h, w = low_stage.shape[2:]
        self.num_feats = len(feats)

        # 所有特征上采样到同一尺寸，然后 concat
        ups = [F.interpolate(f, size=(h, w), mode='bilinear', align_corners=False)
               for f in feats]
        concat = torch.cat(ups, dim=1)          # (B, C*num_feats, H, W)

        # 生成每幅图的 2 通道偏移
        delta_all = self.delta_gen(concat)      # (B, 2*(num_feats-1), H, W)
        delta_all = delta_all.chunk(self.num_feats, dim=1)  # 每组 2 通道

        aligned = []
        for feat, delta in zip(feats[:], delta_all):
            aligned.append(self.bilinear_interpolate_torch_gridsample(
                F.interpolate(feat, size=(h, w), mode='bilinear', align_corners=False),
                (h, w), delta))
        return torch.cat(aligned, 1)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = PAG(in_channels, out_channels)

    def forward(self, x1, x2):
        x = self.up(x2, x1)
        return x


class Decoder(nn.Module):
    def __init__(self,decoder_channels=64,num_classes=2):
        super().__init__()
        self.up1 = Up(64, decoder_channels)
        self.up2 = Up(64, decoder_channels)
        self.up3 = Up(64, decoder_channels)
        self.up4 = Up(64, decoder_channels)
        self.FA = FA(64, 4)
        self.outc = nn.Sequential(
            nn.Conv2d(decoder_channels*4, decoder_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(decoder_channels),
            nn.ReLU6(inplace=True),
            nn.Conv2d(decoder_channels, num_classes, kernel_size=1)
        )

    def forward(self, inputs):
        stem, x1, x2, x3, P4 = inputs
        out1 = self.up1(P4, x3)
        out2 = self.up2(out1, x2)
        out3 = self.up3(out2, x1)
        out4 = self.up4(out3, stem)
        out = self.FA(out4, out3, out2, out1)
        logits = self.outc(out)
        return logits


class model(nn.Module):
    def __init__(self,
                 in_channels=14,
                 num_classes=2,
                 decoder_channels=64,
                 ):
        super().__init__()
        self.backbone = DenseNet(in_channels=in_channels)
        self.globle = Block(in_chs=64, dim=decoder_channels * 2)
        self.decoder = Decoder(decoder_channels,num_classes)


    def forward(self, x):
        stem, x1, x2, x3 = self.backbone(x)
        P4 = self.globle(x3)
        logits = self.decoder([stem, x1, x2, x3, P4])
        return logits

class base_line(nn.Module):
    def __init__(self,
                 in_channels=14,
                 num_classes=2,
                 decoder_channels=64,
                 ):
        super().__init__()
        self.backbone = DenseNet(in_channels=in_channels)
        self.backbone.stem = nn.Sequential(
            nn.Conv2d(in_channels,64,kernel_size=3,padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(64,out_channels=64,kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=2, kernel_size=1),
        )

    def forward(self, x):
        stem, x1, x2, x3 = self.backbone(x)
        logits = self.decoder(x3)
        return logits


class base_line_SSCE(nn.Module):
    def __init__(self,
                 in_channels=14,
                 num_classes=2,
                 decoder_channels=64,
                 ):
        super().__init__()
        self.backbone = DenseNet(in_channels=in_channels)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(64,out_channels=64,kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=2, kernel_size=1),
        )

    def forward(self, x):
        stem, x1, x2, x3 = self.backbone(x)
        logits = self.decoder(x3)
        return logits


class base_line_FCSS(nn.Module):
    def __init__(self,
                 in_channels=14,
                 ):
        super().__init__()
        self.backbone = DenseNet(in_channels=in_channels)
        self.backbone.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.globle = Block(in_chs=64, dim=64 * 2)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8,mode='bilinear'),
            nn.Conv2d(64,out_channels=64,kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=2, kernel_size=1),
        )

    def forward(self, x):
        stem, x1, x2, x3 = self.backbone(x)
        P4 = self.globle(x3)
        logits = self.decoder(P4)
        return logits


class base_line_LPA(nn.Module):
    def __init__(self,
                 in_channels=14,
                 ):
        super().__init__()
        self.backbone = DenseNet(in_channels=in_channels)
        self.backbone.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.decoder = Decoder(64,2)
    def forward(self, x):
        stem, x1, x2, x3 = self.backbone(x)
        logits = self.decoder([stem, x1, x2, x3,x3])
        return logits


class base_line_SSCE_FCSS(nn.Module):
    def __init__(self,
                 in_channels=14,
                 ):
        super().__init__()
        self.backbone = DenseNet(in_channels=in_channels)
        self.globle = Block(in_chs=64, dim=64 * 2)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=8, mode='bilinear'),
            nn.Conv2d(64, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, out_channels=2, kernel_size=1),
        )


    def forward(self, x):
        stem, x1, x2, x3 = self.backbone(x)
        P4 = self.globle(x3)
        logits = self.decoder(P4)
        return logits


class base_line_SSCE_LPA(nn.Module):
    def __init__(self,
                 in_channels=14,
                 ):
        super().__init__()
        self.backbone = DenseNet(in_channels=in_channels)
        self.decoder = Decoder(64, 2)

    def forward(self, x):
        stem, x1, x2, x3 = self.backbone(x)
        logits = self.decoder([stem, x1, x2, x3,x3])
        return logits

class base_line_FCSS_LPA(nn.Module):
    def __init__(self,
                 in_channels=14,
                 ):
        super().__init__()
        self.backbone = DenseNet(in_channels=in_channels)
        self.backbone.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.globle = Block(in_chs=64, dim=64 * 2)
        self.decoder = Decoder(64, 2)

    def forward(self, x):
        stem, x1, x2, x3 = self.backbone(x)
        P4 = self.globle(x3)
        logits = self.decoder([stem, x1, x2, x3,P4])
        return logits


if __name__ == '__main__':
    from thop import profile
    import time
    x = torch.randn(2, 14, 128, 128).cuda()
    model = base_line_SSCE_FCSS().cuda()
    flops, params = profile(model, inputs=(x,))
    num_runs = 10
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
