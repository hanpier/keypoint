from easydict import EasyDict as edict
import os
import json
def get_config():
    C = edict()
    config = C

    """STATE"""
    C.trainer = 'fully_supervised'  # fully_supervised ema transfer
    C.state = 'test'  # train/test
    C.weight_path = '2023-09-18-10-48/checkpoint_best.pt'  # for test checkpoint_best model_best

    C.test_date = '08_25'
    C.test_way = 'test'

    """LOG CONFIG"""
    C.save_debug = 'exp'
    C.checkpoint_path = 'weights'  # the dir to save
    C.resume = False
    C.pre = False

    C.resume_path = './checkpoint_best.pt'

    """ PATH CONFIG """
    C.train_date = '09_15'  # 07_24
    C.train_way = 'train'
    C.unlabel_train_date = '09_11'
    C.unlabel_train_way = 'train'
    C.use_ema = False
    C.score_threshold = 0.9  # for semi
    C.score = 0.5  # for test

    # for dann
    C.w_d = 0.01
        
    """ DATA CONFIG """
    
    C.labeled_batch_size = 8
    C.unlabeled_batch_size = 8
    C.test_batch_size = 1
    C.num_workers = 0
    
    C.flip_prob = 0.5
    C.rotate_prob = 0.7
    C.rotate_factor = 45
    
    C.flip_indices = [1, 0, 2, 3, 4]
    
    """ MODEL CONFIG """
    C.device = "cuda:0"
    C.backbone = 'Unet'  # Unet Bisetnetv1

    C.input_size = [1280, 704]  # [512, 256]  [1280, 704]
    C.heatmap_size = (320, 176)  # (128, 64) (320, 176)
    C.scale_factor = 4          # 8, 4
    
    C.codec_type = "MSRAHeatmap"
    C.codec_cfg = {"input_size":C.input_size,
                   "heatmap_size":C.heatmap_size}
    
    C.test_cfg = {'flip_test': True,
                  'shift_heatmap': True}

    C.clip = 2.

    """OPTIMILIZER CONFIG"""
    C.optimizer_cfg = {
        'type': 'AdamW',
        'lr': 5e-3,
        'weight_decay': 0.05,
    }

    C.transfer_optimizer_cfg = {
        'type': 'AdamW',
        'lr': 5e-3,
        'weight_decay': 0.05,
    }
            
    """ LOSS CONFIG """
    C.supervised_loss = 'mse'  # awing, mse
    C.bone_weight = 0.1  # 10 1 0.1
    C.alpha = 2.1
    C.omega = 14
    C.epsilon = 1
    C.theta = 0.5
    
    C.use_target_weight = False
    C.dataset_keypoint_weights = [1.2, 1., 1., 1.2, 1., 1.2, 1.2, 1., 1., 1.2, 1.2, 1., 1.2, 1., 1., 1.2],
    
    C.consistency_loss = 'mse'
    C.final_consistency_loss_weight = 3
    C.consistency_loss_weight_ramp_up_epoch = 5

    
    """ TRAIN CONFIG """
    C.joint_epoch = 1000
    C.labeled_epoch = 0
    
    C.warmup_epoch = 5
    C.start_factor = 0.01
    C.seed = 1234
    
    C.start_ema_decay = 0.98
    C.end_ema_decay = 0.999
    C.ema_ramp_up_epoch = 5

    C.save_epoch = 200
    
    return config
