import shutil
import os
import sys

sys.dont_write_bytecode = True
from lib.trainer.fully_supervised_trainer import FullySupervisedTrainer
from lib.trainer.semi_supervised_trainer import EMATrainer
from lib.trainer.transfer_trainer import Transfer_trainer
from lib.Config.config import get_config
from argparse import ArgumentParser
import traceback
import torch
import numpy as np
import imgaug as ia

parser = ArgumentParser(description='Mean teacher network for AFLW')
parser.add_argument("--trainer", default='fully_supervised', choices=['ema', 'fully_supervised'], help='type of trainer')
parser.add_argument("--notpretrain", action='store_false', default=True, help='whether to use pretrained for backbone')
parser.add_argument("--resume", action='store_true', default=False, help='resume training from last checkpoint')
parser.add_argument("--pre", action='store_true', default=True, help='pre training from last checkpoint')
parser.add_argument("--joint_epoch", type=int, default=50, help='total number of epochs')
parser.add_argument("--rampup", type=int, default=10,
                    help='number of ramp up epoch for learning rate, [consistency loss weight, ema decay rate]')
parser.add_argument("--batchsize", type=int, default=1, help='batch size')
args = parser.parse_args()

if __name__ == "__main__":
    cfg = get_config()

    # if not args.notpretrain:
    #     cfg.backbone_pretrained = None
    #     cfg.backbone_cfg.pretrained = None

    # cfg.joint_epoch = args.joint_epoch

    cfg.consistency_loss_weight_ramp_up_epoch = args.rampup
    cfg.ema_ramp_up_epoch = args.rampup
    cfg.warmup_epoch = args.rampup

    cfg.name = cfg.trainer + "_" + cfg.backbone + "_" + cfg.supervised_loss

    for k, v in cfg.items():
        print(f"{k}: {v}")
    print("----------------------------------------------------------------------------------------------------")

    os.environ['PYTHONHASHSEED'] = str(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(cfg.seed)
    ia.seed(cfg.seed)


    # if args.trainer == 'ema':
    #     trainer = EMATrainer(cfg)
    if cfg.trainer == 'fully_supervised':
        trainer = FullySupervisedTrainer(cfg)
    elif cfg.trainer == 'ema':
        trainer = EMATrainer(cfg)
    elif cfg.trainer == 'transfer':
        trainer = Transfer_trainer(cfg)
    if cfg.state == 'train':
        try:
            trainer.train(resume=cfg.resume, pre=cfg.pre)
        except Exception:
            print(traceback.print_exc())
    elif cfg.state == 'test':
        trainer.test(cfg.weight_path)

        # shutil.rmtree(cfg.snapshot_dir)
    # except KeyboardInterrupt:
    #     shutil.rmtree(cfg.snapshot_dir)
