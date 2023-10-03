import os
import torch
import time
from torch.nn.utils import clip_grad_norm_
from torch.nn import DataParallel
from tqdm import tqdm
from lib.dataloader.cardata import get_train_loader
from lib.models.loss import AdaptiveWingLoss, KeypointMSELoss, BoneLoss
from lib.models.detector import Unet_Detector
from lib.optimizer.optimizer import build_optimizer
from lib.optimizer.lr_scheduler import LinearWarmupCosineAnnealingLR
from lib.utils.data_cal import AverageMeter, ema_decay_scheduler
from lib.log import Logger
class EMATrainer:
    def __init__(self, config):
        self.config = config
        
        self.device = torch.device(self.config.device)
        
        # dataloader
        self.labeled_train_loader = get_train_loader(config=self.config, type='labeled', drop_last=False)
        self.unlabeled_train_loader_w = get_train_loader(config=self.config, choose='weak', type='unlabeled', drop_last=False)
        self.unlabeled_train_loader_s = get_train_loader(config=self.config, choose='strong', type='unlabeled', drop_last=False)
        self.len_loader = max(len(self.labeled_train_loader), len(self.unlabeled_train_loader_w),
                              len(self.unlabeled_train_loader_s))
        # self.test_loader = get_test_loader(config=self.config)
        
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
        self.u_criterion = torch.nn.MSELoss()

            
        # model network
        self.model = Unet_Detector(self.config)
        if self.config.use_ema:
            from lib.models.ema import ModelEMA
            self.model = ModelEMA(self.config, self.model, self.config.ema_decay)
        self.model = DataParallel(self.model, device_ids=[0]).to(self.device)
        # optimizer
        self.optimizer = build_optimizer(optimizer_cfg=self.config.optimizer_cfg,
                                         model=self.model)
        
        self.lr_scheduler = LinearWarmupCosineAnnealingLR(optimizer=self.optimizer,
                                                          warmup_epochs=self.config.warmup_epoch * self.len_loader,
                                                          max_epochs=self.config.joint_epoch * self.len_loader,
                                                          warmup_start_lr_factor=self.config.start_factor,
                                                          eta_min=1e-6)
        self.current_epoch = 0
        self.min_loss = 1e9
        self.save_epoch = self.config.save_epoch

    def train_epoch(self, epoch):
        print()
        self.model.train()
        
        pbar = tqdm(enumerate(range(self.len_loader)), total=self.len_loader,
                    desc=f'Training epoch {epoch + 1}/{self.config.joint_epoch}',
                    ncols=0)
        
        supervised_dataloader = iter(self.labeled_train_loader)
        
        unsupervised_dataloader_s = iter(self.unlabeled_train_loader_s)
        unsupervised_dataloader_w = iter(self.unlabeled_train_loader_w)
        
        self.current_epoch = epoch

        loss_meter = AverageMeter()
        loss_x_meter = AverageMeter()
        loss_u_meter = AverageMeter()
        bone_loss_meter = AverageMeter()

        self.logger.write('\n')
        self.logger.write('epoch: {} |'.format(epoch))
        self.logger.write('lr: {} |'.format(self.lr_scheduler.get_lr()))
        
        for i, _ in pbar:
            self.optimizer.zero_grad()
            
            labeled_batch = next(supervised_dataloader)
            unlabeled_batch_w = next(unsupervised_dataloader_w)
            unlabeled_batch_s = next(unsupervised_dataloader_s)
            
            num_labeled_item = self.config.labeled_batch_size
            num_unlabeled_item = self.config.unlabeled_batch_size
            
            labeled_batch_image = labeled_batch['img'].to(self.device)
            labeled_batch_heatmap = labeled_batch['heatmap'].to(self.device)

            unlabeled_batch_image_w = unlabeled_batch_w['img'].to(self.device)
            unlabeled_batch_image_s = unlabeled_batch_s['img'].to(self.device)

            # Unlabeled images
            # forward through model
            unlabeled_batch_heatmap_pred_w = self.model(unlabeled_batch_image_w)
            unlabeled_batch_heatmap_pred_s = self.model(unlabeled_batch_image_s)

            heatmap_pred_w_np = unlabeled_batch_heatmap_pred_w.detach().cpu().numpy()

            # Calculate maximum scores for each channel in unlabeled_batch_heatmap_pred_w
            max_scores_w = heatmap_pred_w_np.max(axis=(2, 3))
            # 创建一个布尔掩码，标识大于阈值的位置
            mask = max_scores_w >= self.config.score_threshold
            if not mask.any():
                continue
            # 使用掩码将符合条件的部分复制到新矩阵中
            unlabeled_selected_heatmap_pred_s = unlabeled_batch_heatmap_pred_s[mask, :, :]
            unlabeled_selected_heatmap_pred_w = unlabeled_batch_heatmap_pred_w[mask, :, :]

            loss_u = self.u_criterion(unlabeled_selected_heatmap_pred_s,  unlabeled_selected_heatmap_pred_w)
            loss_u_meter.update(val=loss_u.item(), weight=num_unlabeled_item)

            # Labeled images
            labeled_batch_heatmap_pred = self.model(labeled_batch_image)
            loss_x = self.supervised_criterion(labeled_batch_heatmap_pred, labeled_batch_heatmap)
            loss_x_meter.update(val=loss_x.item(), weight=num_labeled_item)
            
            consistency_loss_weight = 0.1
            bone_loss = self.bone_criterion(labeled_batch_heatmap_pred,
                                            labeled_batch_heatmap, loss_weight=self.config.bone_weight)
            bone_loss_meter.update(val=bone_loss.item(),
                                  weight=num_unlabeled_item)

            loss = loss_x + bone_loss + loss_u * consistency_loss_weight
            loss_meter.update(val=loss.item(), weight=num_unlabeled_item)
            
            # back propagation
            if self.config.use_ema:
                self.model.update(self.model)
            loss.backward()
            clip_grad_norm_(self.model.parameters(), self.config.clip)

            self.optimizer.step()
            
            ema_decay = ema_decay_scheduler(self.config.start_ema_decay,
                                            self.config.end_ema_decay,
                                            max_step=self.config.ema_ramp_up_epoch * self.len_loader,
                                            step=epoch * self.len_loader + i)
            
            # update weights for ema
            if self.config.use_ema:
                self.model.update(self.model, ema_decay)
            
            # update the learning rate
            self.lr_scheduler.step()

            pbar.set_postfix({
                'loss_x': round(loss_x_meter.average(), 8),
                'loss_u': round(loss_u_meter.average(), 8),
                'bone_loss': round(bone_loss_meter.average(), 8),
                'loss': round(loss_meter.average(), 8),
                'top lr': self.lr_scheduler.get_last_lr()[-1],
                'bottom lr': self.lr_scheduler.get_last_lr()[0]
            })
            
        result = {'loss_x': round(loss_x_meter.average(), 8),
                  'loss_u': round(loss_u_meter.average(), 8),
                  'bone_loss': round(bone_loss_meter.average(), 8),
                  'loss': round(loss_meter.average(), 8),
                  }
        for k, v in result.items():
            self.logger.write(f'{k}: {v}|')
            self.logger.scalar_summary(k, v, epoch)
        return result

    def save_checkpoint(self, epoch, dir='checkpoint_last.pt', type='latest'):
        checkpoint_path = os.path.join(self.config.checkpoint_path, dir)
        checkpoint = {'state_dict': self.model.state_dict(),
                      'optimizer': self.optimizer.state_dict(),
                      'lr_scheduler': self.lr_scheduler.state_dict(),
                      'epoch': self.current_epoch,
                      'min_loss': self.min_loss
                      }
        torch.save(checkpoint, checkpoint_path)
        print(f"-----> save {type} checkpoint at epoch {epoch+1}")

    def load_checkpoint_from_resume(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.current_epoch = checkpoint['epoch']
        self.min_loss = checkpoint['min_loss']
        print("----> load resume checkpoint")
        
    def load_checkpoint_from_pt(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.current_epoch = 0
        self.min_loss = self.min_loss
        print("----> load pt checkpoint")

    def train(self, resume=False, pre=False):
        if resume or pre:
            if resume:
                self.load_checkpoint_from_resume(os.path.join(self.config.checkpoint_path, self.config.resume_path))
            elif pre:
                self.load_checkpoint_from_pt(os.path.join(self.config.checkpoint_path, self.config.resume_path))
            time_str = self.config.resume_path.split('/')
            time_str = time_str[0]
            self.config.checkpoint_path = os.path.join(self.config.checkpoint_path, time_str)
            self.checkpoint_path = os.path.join(self.config.checkpoint_path, self.config.resume_path)
            print('Load resume chekpoint complete')

        else:
            time_str = time.strftime('%Y-%m-%d-%H-%M')
            os.makedirs(os.path.join(self.config.checkpoint_path, str(time_str)))
            self.checkpoint_path = os.path.join(self.config.checkpoint_path, str(time_str))

        self.logger = Logger(config=self.config, time_str=time_str)

        for epoch in range(self.current_epoch, self.config.labeled_epoch + self.config.joint_epoch):
            result = self.train_epoch(epoch)
            current_loss = result['loss']
            if current_loss <= self.min_loss:
                self.min_loss = current_loss
                self.save_checkpoint(epoch=epoch, dir='model_best.pt', type='best')
            if (epoch + 1) % self.save_epoch == 0:
                save_weight_step = "model_" + str(epoch + 1).zfill(3) + ".pt"
                self.save_checkpoint(epoch=epoch, dir=save_weight_step, type='epoch')
            self.save_checkpoint(epoch=epoch, dir='model_last.pt', type='latest')
