import landslide_model
from torch import nn
import torch
from torch.optim.lr_scheduler import LRScheduler
from torch import optim
from loss import TL_loss, focal_hausdorffErloss, Lovasz_ce_loss, DynamicWeightedCrossEntropyLoss,FocalLoss, DynamicFocalLoss,AutoBalanceWeightedLoss, hybridloss, AdaptiveSegmentationLoss,mix_loss
from pytorch_toolbelt.losses import DiceLoss
from loss.lovasz import LovaszSoftmaxLoss
from Data_loader import cfg


class WarmupCosineAnnealingLR(LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=0.0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # 预热阶段：线性提升
            alpha = self.last_epoch / self.warmup_epochs
            return [base_lr * alpha for base_lr in self.base_lrs]
        else:
            # 余弦退火阶段（返回标量列表）
            progress = (self.last_epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + torch.cos(torch.tensor(progress * torch.pi)))
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay.item() for base_lr in self.base_lrs]



class Loader(dict):
    def __init__(self, model_type = None):
        self.cfg = cfg
        self.model_type = model_type
        self.loss_function = self.get_loss_function()
        self.model = self.get_segment_model()
        self.optimizer = self.get_optimizer()
        self.scheduler = self.get_lr_scheduler()

    def get_segment_model(self):
        # 定义模型名称到模型类的映射
        model_mapping = {
            'SCDUNetPP': lambda:  landslide_model.SCDUNetPP(self.cfg.dataset.in_channels, num_class=self.cfg.dataset.Class),
            'pyramidMamba': lambda:  landslide_model.PyramidMamba(),
            'RIPF_Unet': lambda: landslide_model.RiPF_UNet(n_channels=self.cfg.dataset.in_channels,n_classes=self.cfg.dataset.Class, bilinear=True),
            'MFFEnet': lambda: landslide_model.DeepLab(in_channels =self.cfg.dataset.in_channels,num_classes=self.cfg.dataset.Class),
            'TransUnet2': lambda: landslide_model.TransUNet2(num_classes=self.cfg.dataset.Class),
            'MiM_iSTD': lambda: landslide_model.MiM([2] * 3, [16, 32, 64, 128, 256], self.cfg.dataset.in_channels),
            'RFA_ResUnet': lambda: landslide_model.BFA_resunet(num_classes=self.cfg.dataset.Class),
            'bisednet': lambda: landslide_model.BiSeNet_DenseNet6(in_channels = self.cfg.dataset.in_channels,n_classes=self.cfg.dataset.Class),
            'MS2LandsNet': lambda:  landslide_model.MS2LandsNet(input_channel=self.cfg.dataset.in_channels, n_classes=self.cfg.dataset.Class),
            'Base_line': lambda: landslide_model.base_line(num_classes=self.cfg.dataset.Class),
            'ssce': lambda: landslide_model.base_line_SSCE(num_classes=self.cfg.dataset.Class),
            'LPA': lambda: landslide_model.base_line_LPA(),
            'FCSS': lambda: landslide_model.base_line_FCSS(),
            'ssce_fcss': lambda: landslide_model.base_line_SSCE_FCSS(),
            'fcss_lpa': lambda: landslide_model.base_line_FCSS_LPA(),
            'ssce_lpa': lambda: landslide_model.base_line_SSCE_LPA(),
            'DS4Net': lambda: landslide_model.DS4Net(in_channels=self.cfg.dataset.in_channels,num_classes=self.cfg.dataset.Class),
            
            


            # 可以继续添加其他模型
        }
        # 检查模型名称是否存在于映射中
        if self.model_type not in model_mapping:
            available_models = ', '.join(model_mapping.keys())
            raise ValueError(f"Model {self.model_type} not found in model mapping. Available models are: {available_models}")
        return model_mapping[self.model_type]()

    def get_loss_function(self):
        # 初始化为None或默认损失函数
        loss_mapping = {
            'celoss': lambda : nn.CrossEntropyLoss(weight=self.cfg.train.loss_function_weight),
            'Tversky_loss_lovasz': lambda : TL_loss(alpha=0.5, beta=0.5, n_class = self.cfg.dataset.Class),
            'f-h-loss': lambda : focal_hausdorffErloss(),
            'lovasz_ce_loss': lambda : Lovasz_ce_loss(weight=self.cfg.train.loss_function_weight, n_class=self.cfg.dataset.Class),
            'lovasz_softmax': lambda : LovaszSoftmaxLoss,
            'DynamicWeightedCrossEntropyLoss': lambda : DynamicWeightedCrossEntropyLoss(),
            'focalloss': lambda: FocalLoss(),
            'dynamic_focal_loss': lambda:  DynamicFocalLoss(),
            'dice': lambda: DiceLoss(mode='multiclass'),
            'ASL': lambda: AutoBalanceWeightedLoss(classes=2),
            'hybridloss': lambda : hybridloss,
            'ASL_A':lambda : AdaptiveSegmentationLoss(num_classes=self.cfg.dataset.Class),
            'mix_loss':lambda : mix_loss(),
        }
        # 检查损失函数名称是否存在于映射中
        if self.cfg.train.loss_function not in loss_mapping:
            loss = ', '.join(loss_mapping.keys())
            raise ValueError(f"Model {self.cfg.train.loss_function} not found in model mapping. Available models are: {loss}")
        return loss_mapping[self.cfg.train.loss_function]()

    def get_optimizer(self):
        # 优化器
        # 定义优化器映射
        optimizer_mapping = {
            'SGD': lambda: optim.SGD(
                self.model.parameters(),
                lr=self.cfg.optimizer.base_lr,
                momentum=self.cfg.optimizer.momentum,
                weight_decay=self.cfg.optimizer.weight_decay
            ),
            'AdamW': lambda: optim.AdamW(
                self.model.parameters(),
                lr=self.cfg.optimizer.base_lr,
                weight_decay=self.cfg.optimizer.weight_decay
            ),
            'Adam': lambda: optim.Adam(
                self.model.parameters(),
                lr=self.cfg.optimizer.base_lr,
                weight_decay=self.cfg.optimizer.weight_decay
            )
        }

        # 检查优化器名称是否存在于映射中
        if self.cfg.optimizer.type not in optimizer_mapping:
            available_optimizers = ', '.join(optimizer_mapping.keys())
            raise ValueError(
                f"Optimizer {self.cfg.optimizer.type} not found in optimizer mapping. Available optimizers are: {available_optimizers}")

        # 返回对应的优化器实例
        return optimizer_mapping[self.cfg.optimizer.type]()

    def get_lr_scheduler(self):
        scheduler_mapping = {
            'Poly': lambda: optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: max(0.0, float(self.cfg.scheduler.epoch - epoch) / float(max(1,
                                                                                                     self.cfg.scheduler.epoch - self.cfg.scheduler.warmup_epoch))) if epoch >= self.cfg.scheduler.warmup_epoch else float(
                    epoch) / float(max(1, self.cfg.scheduler.warmup_epoch)),

            ),
            'step': lambda: optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.cfg.scheduler.step_size,
                gamma=self.cfg.scheduler.gamma
            ),
            'CosineAnnealingLR': lambda: optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg.scheduler.epoch,
            ),
            'CosineAnnealingWarmRestarts': lambda: optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=8,
                T_mult=2,
                eta_min=1e-6,
            ),
            'normal': lambda: optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=lambda epoch: 1
            ),
            'WarmupCosineAnnealingLR': lambda: WarmupCosineAnnealingLR(
                self.optimizer,
                warmup_epochs=self.cfg.scheduler.warmup_epoch,
                total_epochs=self.cfg.scheduler.epoch,
                eta_min=0,
            )
        }

        # 检查学习率调度器名称是否存在于映射中
        if self.cfg.scheduler.type not in scheduler_mapping:
            available_schedulers = ', '.join(scheduler_mapping.keys())
            raise ValueError(
                f"Scheduler {self.cfg.scheduler.type} not found in scheduler mapping. Available schedulers are: {available_schedulers}")

        return scheduler_mapping[self.cfg.scheduler.type]()










