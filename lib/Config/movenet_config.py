import os
import torch
import imgaug.augmenters as iaa
# only bone loss and heatmap loss
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
# Unet clip:5
config_loc = {
    # 网络训练部分
    # 'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'gpu': (0,1),
    'model':'Unet', # Unet,Move
    'clip_gradient': 5,
    'feature_scale': 2,
    'step': 600,
    'seed': 26,
    'device': torch.device("cuda:0"),
    'batch_size': 8,
    'epochs': 800,
    'save_epoch': 10,
    'learning_rate': 5e-5,
    'sigma': 1,
    'lr_scheduler': 'step1',  # 可以选择'step1','step2'梯度下降，'exponential'指数下降

    # 原图尺寸
    'img_h': 720,
    'img_w': 1280,

    'keypoint_weights': [1.2, 1., 1., 1.2, 1., 1.2, 1.2, 1., 1., 1.2, 1.2, 1., 1.2, 1., 1., 1.2],

    # 裁剪后的尺寸
    'cut_h': 704, # 640
    'cut_w': 1280,

    # 网络输入的图像尺寸
    'input_h': 704, # 176
    'input_w': 1280, # 320

    #1280/4=320
    'scale': 4,

    # heatmap size
    'h_h':176,
    'h_w':320,

    # 高斯核大小
    'gauss_h': 11,
    'gauss_w': 11,

    # 关键点个数
    'kpt_n': 16,

    # 网络评估部分
    'test_batch_size': 1,
    'test_threshold': 0.5,

    # 设置路径部分
    'train_date': '07_24',
    'train_way': 'train',
    'test_date': '08_25',
    'test_way': 'test',
    # 调用的模型
    # 'pkl_file': '2023-08-16-08-11/model_070.pth',
    # 'pkl_file': '2023-08-16-05-59/model_280.pth',
    # 'pkl_file': '2023-08-22-09-20/model_last.pth',
    # 2023-08-22-07-15
    # 2023-08-08-01-14
    'pkl_file': '2023-08-29-01-07/model_best.pth',
    'debug': False,

    # 是否加载预训练模型
    'use_old_pkl': True,
    'old_pkl': '2023-08-29-01-07/model_last.pth', # weight后面的完整路径

    # # pytorch < 1.6
    # 'pytorch_version': False,

    # remember location
    'start_x': 200,
    'start_y': 200,
    'start_angle': 0,

    # max x,y
    'max_x': 300,
    'max_y': 250,
    'max_angle': 90,

    # min x,y
    'min_x': 100,
    'min_y': 100,

    # key points relative location
    'distance_12': 360,
    'distance_13': 200,
    'distance_23': 410,

    'delta': 50,

    'seq': iaa.Sequential([
        iaa.Fliplr(0.5),  # 镜像50%
        iaa.Multiply((0.1, 1.9)),
        iaa.OneOf([
            sometimes(iaa.Affine(rotate=(-5, 5), scale=(0.75, 1.25))),  # 旋转，缩放
            sometimes(iaa.PerspectiveTransform(scale=(0.05, 0.1), keep_size=True)),

        ]),
        iaa.OneOf([
            sometimes(iaa.Dropout((0.1, 0.2))),  # randomly remove up to 10% of the pixels
            sometimes(iaa.CoarseDropout((0.05, 0.1), size_percent=(0.01, 0.05))),
            sometimes(iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
        ]),
        sometimes(
            iaa.SomeOf((0, 3),
                       [
                           iaa.OneOf([
                               iaa.Rain(speed=(0.01, 0.1)),
                               iaa.Clouds(),
                               iaa.Fog(),
                               iaa.Snowflakes(flake_size=(0.1, 0.3), speed=(0.01, 0.05)),
                           ]),
                           iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(3, 5)),
                               iaa.MedianBlur(k=(3, 5)),
                               iaa.MotionBlur(k=10),
                           ]),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
                           iaa.LinearContrast((0.75, 1.5)),
                           iaa.PadToFixedSize(width=1280, height=720),
                           iaa.CropToFixedSize(width=1280, height=720),
                           # iaa.PadToFixedSize(width=512, height=512, position="center"),
                       ],
                       random_order=True)),
            # iaa.SaveDebugImageEveryNBatches('../tmp', 10000)
    ],
        random_order=True
    )
#     # 'photo_to_world':
  }
