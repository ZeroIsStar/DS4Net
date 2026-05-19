import os
import torch
import datetime
import logging
import random
import numpy as np
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from configs import Loader
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
from Data_loader import DataLoader, cfg
from metric_tool import SegEvaluator


class Trainer:
    # region 初始化模型，数据集batch_size,学习率还有训练轮数、损失函数、优化器
    def __init__(self, train_loader, val_loader,
                 device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"),
                 model_type=None, dataset_name=None,model_weight=None,
                 early_stop=False, early_patience=20
                 ):
        self.cfg                = cfg
        self.model_type         = model_type
        self.dataset_name       = dataset_name
        self.loder              = Loader(self.model_type)
        self.model              = self.loder.model.to(device)
        self.loss_function      = self.loder.loss_function.to(device)
        self.Train_loader       = train_loader
        self.Val_loader         = val_loader
        self.model_weight       = model_weight
        self.best_val_loss      = float('inf')
        self.patience           = early_patience
        self.counter            = 0
        self.dtype              = np.dtype([('miou', np.float32), ('recall', np.float32), ('precision', np.float32), ('F1', np.float32)])
        self.data               = np.array([], dtype=self.dtype)
        self.best_f1_score      = 0.0
        self.device             = device
        self.early_stop         = early_stop
        self.SegEvaluator       = SegEvaluator(1)
    # endregion

    @staticmethod
    def set_random_seed(seed = 42, deterministic=False):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.enabled = True

    def weights_init(self, model, init_method='kaiming', init_gain=1.414):
        def init_func(m):
            if isinstance(m, nn.Conv2d):
                if init_method == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_method == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_method == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_method == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_method)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        print('\n initialize network with %s method' % init_method)
        model.apply(init_func)

    def make_dir(self):
        result = 'tf-logs'
        os.makedirs(result, exist_ok=True)
        dataset_dir = os.path.join(result, self.dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        model_dir = os.path.join(dataset_dir, self.model_type)
        os.makedirs(model_dir, exist_ok=True)
        loss_function_dir = os.path.join(model_dir, self.cfg.train.loss_function)
        os.makedirs(loss_function_dir, exist_ok=True)

        # 获取当前日期和时间
        current_datetime = datetime.datetime.now()
        # 格式化日期为字符串
        current_date = current_datetime.date()  # 获取日期部分
        current_time = current_datetime.time()  # 获取时间部分

        # 根据需要选择日期格式
        formatted_date = current_date.strftime('%Y_%m_%d')  # 格式化为年-月-日
        formatted_time = current_time.strftime('%H_%M_%S')  # 格式化为时:分:秒
        day_log_dir = os.path.join(loss_function_dir, formatted_date)
        os.makedirs(day_log_dir, exist_ok=True)
        log_dir = os.path.join(day_log_dir, formatted_time)
        os.makedirs(log_dir, exist_ok=True)
        return loss_function_dir, log_dir

    def save_log(self, log_dir):
        # 记录日志
        # 配置日志记录器
        logging.basicConfig(
            level=logging.INFO,  # 设置日志级别为INFO或者更高级别
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(log_dir, 'training.log')),  # 将日志写入指定的文件
            ]
        )
        logging.info(
            f'Starting training with \n'
            f'model={self.model_type},\n'
            f'dataset={self.dataset_name},'
            f'optimizer={self.cfg.optimizer.type}_weight_decay={self.cfg.optimizer.weight_decay}_momentum={self.cfg.optimizer.momentum},\n'
            f'loss_function={self.cfg.train.loss_function},\n'
            f'scheduler={self.cfg.scheduler.type},\n'
            f'learning rate={self.cfg.optimizer.base_lr},\n'
            f' batch_size={self.cfg.dataset.batch_size},\n'
            f'num_epochs={self.cfg.scheduler.epoch}\n')

    def train(self):
        # ....................................创建模型权重和记录地址........................................................
        model_dir, log_dir = self.make_dir()
        self.weights_init(self.model, init_method='kaiming')  # kai ming
        print("-------模型{}开始训练--------".format(self.model_type))
        self.save_log(log_dir)  # 记录日志
        # 加载模型权重
        if self.model_weight is not None:
            self.model.load_state_dict(torch.load(self.model_weight))
        # 模型的训练和验证
        for i in tqdm(range(self.cfg.scheduler.epoch)):
            writer = SummaryWriter(log_dir=log_dir)  # tensorboard --logdir=tensorboard_logs # 调用
            # 模型训练和验证
            # region 训练
            scaler = GradScaler(enabled=True)
            # 模型训练
            train_loss, val_loss = 0, 0
            self.model.train()
            self.SegEvaluator.reset()
            for batch, sample in enumerate(self.Train_loader):
                # with autocast(enabled=True):
                    img, label, _ = sample
                    img, label = img.to(self.device), label.to(self.device)
                    self.loder.optimizer.zero_grad()
                    outputs = self.model(img)
                    if isinstance(outputs, (list, tuple)):
                        outputs_list = outputs
                    else:
                        outputs_list = {outputs}
                    train_total_loss = []
                    for output in outputs_list:
                        output.to(self.device)
                        loss = self.loss_function(output, label, i)
                        train_total_loss.append(loss)
                    loss = sum(train_total_loss) / len(train_total_loss)
                    scaler.scale(loss).backward()
                    # nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.cfg.dataset.clip_grad_value_)
                    scaler.step(self.loder.optimizer)
                    scaler.update()
                    train_loss += loss.item()
            # 更新学习率
            if self.cfg.optimizer.base_lr >= self.cfg.optimizer.min_lr:
                self.loder.scheduler.step()
                # current_lr = self.loder.scheduler.get_lr()
                # print(current_lr)
            # endregion

            # region 验证
            self.model.eval()
            with torch.no_grad():
                for batch, sample in enumerate(self.Val_loader):
                    img, label, _ = sample
                    img, label = img.to(self.device), label.to(self.device)
                    outputs = self.model(img)
                    if isinstance(outputs, (list, tuple)):
                        outputs_list = outputs
                        final_output = outputs[-1]
                    else:
                        final_output = outputs
                        outputs_list = {outputs}
                    val_total_loss = []
                    for output in outputs_list:
                        output.to(self.device)
                        loss = self.loss_function(output, label ,i)
                        val_total_loss.append(loss)
                    loss = sum(val_total_loss) / len(val_total_loss)
                    val_loss += loss.item()
                    output = torch.argmax(torch.softmax(final_output, dim=1), dim=1)
                    output = output.cpu().numpy().astype(np.int8)
                    label = label.cpu().numpy().astype(np.int8)
                    for p, l in zip(output, label):
                        self.SegEvaluator.add_batch(l.squeeze(), p.squeeze())
                metric = self.SegEvaluator.matrix(1)
                Iou = metric['1_IoU']
                recall = metric['Recall']
                precision = metric['Precision']
                F1 = metric['F1']
                Accu = metric['OA']
                kappa = metric['Kappa']
                self.data = np.append(self.data, np.array([(Iou, recall, precision, F1)], dtype=self.dtype))
                # ...............................................如果当前F1得分比最佳F1得分高，则保存模型权重...................
                if F1 > self.best_f1_score:
                    self.best_f1_score = F1
                    best_model_weights = self.model.state_dict()
                    configs = "epoch{:.4g}_iou{:.4g}_Acc{:.4g}_recall{:.4g}_pre{:.4g}_F1_{:.4g}".format(
                        i + 1, Iou, Accu, recall, precision, self.best_f1_score) + ".pth"
                    model_path = os.path.join(log_dir, configs)
                # 如果验证损失不再改善，计数器加1；否则，重置计数器
                # # region 早停策略
                if self.early_stop:
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.counter = 0
                    else:
                        self.counter += 1
                    # 如果计数器达到早停的耐心值，就提前停止训练
                    if self.counter >= self.patience:
                        print("Early stopping after {} epochs.".format(self.cfg.scheduler.epoch))
                        break
                # endregion
                # ............................................region 结果记录部分.............................................
            writer.add_scalars("Epoch_Loss",
                               {"Train_loss": (train_loss / (self.Train_loader.__len__() * self.cfg.dataset.batch_size)),
                                "val_loss": (val_loss / (self.Val_loader.__len__() * self.cfg.dataset.batch_size))}, i + 1)
            writer.add_scalars("evalute", {"IOU": Iou, "recall": recall, "precision": precision, "F1": F1}, i + 1)
            # endregion
            # 关闭
            writer.close()
        torch.save(best_model_weights, model_path)
        return 0
        # endregion


if __name__ == '__main__':
    model_list = ['SCDUNetPP','pyramidMamba','RIPF_Unet','MFFEnet','TransUnet2','MiM_iSTD','RFA_ResUnet','bisednet','CSSLnet','MS2LandsNet']
    dataset_list = ['Luding']
    for model in model_list:
        for dataset in dataset_list:
            Train_loader, Val_loader, _ = DataLoader(dataset_name=dataset).get_dataloader()
            trainer = Trainer(train_loader=Train_loader, val_loader=Val_loader, model_type=model, dataset_name=dataset)
            trainer.set_random_seed()
            trainer.train()

    # .................................region 示例代码单个模型数据集........................................................
    # Train_loader, Val_loader, _ = DataLoader(dataset_name=cfg.dataset.dataset_name).get_dataloader()
    # trainer = Trainer(train_loader=Train_loader, val_loader=Val_loader, model_type=cfg.segment_model.type, dataset_name = cfg.dset.dataset_name)
    # trainer.set_random_seed()
    # trainer.train()

