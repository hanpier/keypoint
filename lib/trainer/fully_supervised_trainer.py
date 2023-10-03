import os
import torch
import time
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from tqdm import tqdm
import cv2
from lib.dataloader.cardata import get_train_loader, get_test_loader
from lib.models.loss import AdaptiveWingLoss, KeypointMSELoss, BoneLoss
from lib.models.detector import Unet_Detector
from lib.models.detector import Bisetnet_Detector
from lib.optimizer.optimizer import build_optimizer
from lib.optimizer.lr_scheduler import LinearWarmupCosineAnnealingLR
from lib.utils.data_cal import AverageMeter
from lib.log import Logger


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class FullySupervisedTrainer:
    def __init__(self, config):
        self.config = config

        self.device = torch.device(self.config.device)

        # dataloader
        self.train_loader = get_train_loader(config=self.config, type='labeled', drop_last=False, trans=True)
        self.test_loader = get_test_loader(config=self.config, type='labeled', drop_last=False)
        self.len_loader = len(self.train_loader)

        # loss functions
        if self.config.supervised_loss == 'awing':
            self.supervised_criterion = AdaptiveWingLoss(alpha=self.config.alpha,
                                                         omega=self.config.omega,
                                                         epsilon=self.config.epsilon,
                                                         theta=self.config.theta,
                                                         use_target_weight=self.config.use_target_weight)
        elif self.config.supervised_loss == 'mse':
            self.supervised_criterion = KeypointMSELoss()

        self.bone_criterion = BoneLoss()

        # model
        self.model = Unet_Detector(self.config)
        # self.model = Bisetnet_Detector(self.config)
        self.model = DataParallel(self.model, device_ids=[0]).to(self.device)

        # optimizer
        self.optimizer = build_optimizer(optimizer_cfg=self.config.optimizer_cfg,
                                         model=self.model)

        self.lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,
                                                          warmup_epochs=self.config.warmup_epoch * self.len_loader,
                                                          max_epochs=self.config.joint_epoch * self.len_loader,
                                                          warmup_start_lr_factor=self.config.start_factor,
                                                          eta_min=0)
        self.current_epoch = 0
        self.min_loss = 1e9
        self.save_epoch = self.config.save_epoch
        self.checkpoint_path = self.config.checkpoint_path

    def train_epoch(self, epoch):
        self.model.train()
        pbar = tqdm(enumerate(range(self.len_loader)), total=self.len_loader,
                    desc=f'Training epoch {epoch + 1}/{self.config.joint_epoch}',
                    ncols=0)
        supervised_dataloader = iter(self.train_loader)
        self.current_epoch = epoch
        supervised_loss_meter = AverageMeter()
        bone_loss_meter = AverageMeter()
        total_loss_meter = AverageMeter()
        self.logger.write('\n')
        self.logger.write('epoch: {} |'.format(epoch))
        self.logger.write('lr: {} |'.format(self.lr_scheduler.get_lr()))

        for i, _ in pbar:
            self.optimizer.zero_grad()

            batch = next(supervised_dataloader)

            batch_image = batch['img'].to(self.device)

            batch_heatmap = batch['heatmap'].to(self.device)

            num_item = batch_image.shape[0]

            # Labeled images
            # forward through model
            batch_heatmap_pred, _ = self.model(batch_image)

            supervised_loss = self.supervised_criterion(batch_heatmap_pred,
                                                        batch_heatmap)
            bone_loss = self.bone_criterion(batch_heatmap_pred,
                                            batch_heatmap, loss_weight=self.config.bone_weight)
            supervised_loss_meter.update(val=supervised_loss.item(),
                                         weight=num_item)
            bone_loss_meter.update(val=bone_loss.item(),
                                   weight=num_item)

            loss_total = supervised_loss + bone_loss

            total_loss_meter.update(val=loss_total.item(),
                                    weight=num_item)

            # back propagation
            loss_total.backward()
            clip_grad_norm_(self.model.parameters(), self.config.clip)

            self.optimizer.step()

            # update the learning rate
            self.lr_scheduler.step()

            pbar.set_postfix({
                'supervised_loss': round(supervised_loss_meter.average(), 8),
                'bone_loss': round(bone_loss_meter.average(), 8),
                'total_loss': round(total_loss_meter.average(), 8),
                'top lr': self.lr_scheduler.get_last_lr()[-1],
                'bottom lr': self.lr_scheduler.get_last_lr()[0]
            })

        result = {'train/supervised_loss': round(supervised_loss_meter.average(), 8),
                  'train/bone_loss': round(bone_loss_meter.average(), 8),
                  'train/total_loss': round(total_loss_meter.average(), 8)}
        for k, v in result.items():
            self.logger.write(f'{k}: {v}|')
            self.logger.scalar_summary(k, v, epoch)
        return result

    def test_epoch(self):
        self.model.eval()

        for index, batch in enumerate(self.test_loader):
            if index >= len(self.test_loader):
                return
            batch_image = batch['img'].to(self.device)
            # mask = torch.zeros(1, 3, 704, 1280, dtype=torch.bool)
            # mask[:, :, :200, :] = True
            # batch_image[mask] = 0  # 将掩码部分的像素置为0，或者根据需求进行其他处理
            img = cv2.imread(
                os.path.join('./data', self.config.test_date, self.config.state, 'imgs', batch['img_name'][0]))
            with torch.no_grad():
                heatmap_pred, _ = self.model(batch_image)
                pre = self.model.module.predict(heatmap_pred)
                self.model.module.show_point_on_picture(img, pre)
                save_dir = os.path.join('./results', self.config.test_date)
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                result_dir = os.path.join(save_dir, batch['img_name'][0].split('.')[0] + '_keypoint.jpg')
                print('keypoint result path：', result_dir)
                cv2.imwrite(result_dir, img)
        return

    def save_checkpoint(self, epoch, dir='checkpoint_last.pt', type='latest'):
        checkpoint_path = os.path.join(self.checkpoint_path, dir)
        checkpoint = {'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict(),
                      'epoch': self.current_epoch,
                      'min_loss': self.min_loss
                      }
        torch.save(checkpoint, checkpoint_path)
        if type != 'latest':
            print(f"-----> save {type} checkpoint at epoch {epoch + 1}")

    def load_checkpoint_from_pt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.current_epoch = checkpoint['epoch']
        self.min_loss = checkpoint['min_loss']
        print("----> load checkpoint")

    def train(self, resume=False, pre=False):
        if resume:
            self.load_checkpoint_from_pt(os.path.join(self.checkpoint_path, self.config.checkpoint_path))
            time_str = self.checkpoint_path.split('/')
            time_str = time_str[0]
            self.checkpoint_path = os.path.join(self.checkpoint_path, time_str)
            self.logger = Logger(config=self.config, time_str=time_str)
            print('Load complete')
        else:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            os.makedirs(os.path.join(self.checkpoint_path, str(time_str)))
            self.checkpoint_path = os.path.join(self.checkpoint_path, str(time_str))
            self.logger = Logger(config=self.config, time_str=time_str)

        for epoch in range(self.current_epoch, self.config.labeled_epoch + self.config.joint_epoch):
            result = self.train_epoch(epoch)

            current_loss = result['train/total_loss']
            if current_loss <= self.min_loss:
                self.min_loss = current_loss
                self.save_checkpoint(epoch=epoch, dir='checkpoint_best.pt', type='best')
            if (epoch + 1) % self.save_epoch == 0:
                save_weight_step = "model_" + str(epoch + 1).zfill(4) + ".pt"
                self.save_checkpoint(epoch=epoch, dir=save_weight_step, type='epoch')
            self.save_checkpoint(epoch=epoch, dir='checkpoint_last.pt', type='latest')

    def test(self, weight_path):
        self.load_checkpoint_from_pt(os.path.join(self.checkpoint_path, weight_path))
        self.test_epoch()
