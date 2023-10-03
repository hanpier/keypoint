import torchvision
import torch.utils.data
from PIL import Image
import imgaug.augmenters as iaa
sometimes = lambda aug: iaa.Sometimes(0.5, aug)
from lib.codec.msra_heatmap import MSRAHeatmap
from lib.pre_processing.data_pre_loc import *
from lib.utils.data_cal import *
from lib.dataloader.transforms import ImageAugmentation, TransformFixMatch
CODECS = {"MSRAHeatmap": MSRAHeatmap}
# RUN lib.processing.utils.mean.py for mean and std
label_mean = [0.5156, 0.5411, 0.5201]
label_std = [0.2088, 0.1970, 0.2194]
unlabel_mean = [0.5471, 0.5305, 0.5156]
unlabel_std = [0.1815, 0.1941, 0.1980]
seq = iaa.Sequential([
                     iaa.Fliplr(0.5),  # 镜像50%
                     iaa.Multiply((0.1, 1.9)),
                     iaa.OneOf([  # 坐标变化
                     sometimes(iaa.Affine(rotate=(-5, 5), scale=(0.75, 1.25))),  # 旋转，缩放
                     sometimes(iaa.PerspectiveTransform(scale=(0.05, 0.1), keep_size=True)),

                     ]),
                     iaa.OneOf([  # 遮挡
                     sometimes(iaa.Dropout((0.1, 0.2))),  # randomly remove up to 10% of the pixels
                     sometimes(iaa.CoarseDropout((0.05, 0.1), size_percent=(0.01, 0.05))),
                     sometimes(iaa.Cutout(nb_iterations=(1, 5), size=0.2, squared=False)),
                     ]),
                     sometimes(
                     iaa.SomeOf((1, 5),
                     [  # 模拟自然场景
                        iaa.OneOf([
                            iaa.Rain(speed=(0.01, 0.1)),
                            iaa.Clouds(),
                            iaa.Fog(),
                            iaa.Snowflakes(flake_size=(0.1, 0.3), speed=(0.01, 0.05)),
                        ]),
                       iaa.OneOf([ # 噪声
                           iaa.GaussianBlur((0, 3.0)),
                           iaa.AverageBlur(k=(3, 5)),
                           iaa.MedianBlur(k=(3, 5)),
                           iaa.MotionBlur(k=10),
                       ]),
                        iaa.OneOf([
                            iaa.RemoveSaturation(),  # 去除饱和度
                            iaa.AddToBrightness((-30, 30)),
                            iaa.Grayscale(alpha=(0.0, 1.0)),
                            iaa.ChangeColorTemperature((1100, 10000))
                        ]),
                        iaa.OneOf([
                            iaa.GammaContrast((0.5, 2.0)),
                            iaa.LogContrast(gain=(0.6, 1.4)),
                            iaa.LinearContrast((0.4, 1.6)),  # 对比度
                            iaa.AllChannelsHistogramEqualization()

                        ]),
                       iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255)),
                       iaa.PadToFixedSize(width=1280, height=720), # 缩放
                       iaa.CropToFixedSize(width=1280, height=720),
                       # iaa.PadToFixedSize(width=512, height=512, position="center"),
                    ],
                    random_order=True)),
            # iaa.SaveDebugImageEveryNBatches('../tmp', 10000)
    ],
        random_order=True
    )
class Label_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, config, transform=None):
        super(Label_Dataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_name_list = os.listdir(os.path.join(dataset_path, 'imgs'))
        self.transform = transform
        self.config = config

    def __getitem__(self, index):
        codec = CODECS[self.config.codec_type](**self.config.codec_cfg)
        # 先处理img
        img_name = self.img_name_list[index]
        # convert('RGB')保证读取的顺序是RGB
        img = PIL.Image.open(os.path.join(self.dataset_path, 'imgs', img_name)).convert('RGB')
        #  img = letterbox_image(img, codec.input_size)  # resize to (W,H)
        item = dict()
        if (self.config.state == 'test'):
            img = torchvision.transforms.ToTensor()(img)
            img = img[:, :codec.input_size[1], :codec.input_size[0]]  # 3*704*1280
            item['img'] = img
            item['img_name'] = img_name
            return item
        # 读入标签
        label_name = img_name.split('.')[0] + '.json'
        keypoints = json_to_numpy(os.path.join(self.dataset_path, 'labels', label_name))
        if self.transform is not None:
            img, keypoints = self.transform(img, keypoints)

        img = torchvision.transforms.ToTensor()(img)
        #  img = transforms.Normalize(mean=label_mean, std=label_std)(img)

        img = img[:, :codec.input_size[1], :codec.input_size[0]]  # 3*704*1280
        # Keypoint
        scaled_keypoints = np.round(keypoints / codec.scale_factor)
        scaled_keypoints_heatmap = codec.encode(keypoints=scaled_keypoints)
        item['img'] = img
        item['keypoints'] = scaled_keypoints
        item['heatmap'] = scaled_keypoints_heatmap
        item['img_name'] = img_name

        return item

    def __len__(self):

        return len(self.img_name_list)


class UnLabel_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, config,choose, transform=None):
        super(UnLabel_Dataset, self).__init__()
        self.dataset_path = dataset_path
        self.img_name_list = os.listdir(os.path.join(dataset_path, 'imgs'))
        self.transform = transform
        self.config = config
        self.choose = choose

    def __getitem__(self, index):
        codec = CODECS[self.config.codec_type](**self.config.codec_cfg)
        # 先处理img
        img_name = self.img_name_list[index]
        # convert('RGB')保证读取的顺序是RGB
        img = PIL.Image.open(os.path.join(self.dataset_path, 'imgs', img_name)).convert('RGB')
        img = letterbox_image(img, self.config.input_size)
        item = dict()

        if self.transform is not None:
            img = self.transform(img, self.choose)
        img = torchvision.transforms.ToTensor()(img)
        img = transforms.Normalize(mean=unlabel_mean, std=unlabel_std)(img)
        img = img[:, :codec.input_size[1], :codec.input_size[0]]  # 3*704*1280
        # Keypoint
        item['img'] = img
        item['img_name'] = img_name

        return item

    def __len__(self):

        return len(self.img_name_list)

def get_train_loader(config, trans=True, choose='none', type: str = 'labeled', drop_last=True):
    if type == 'labeled':
        if trans:
            transformations = ImageAugmentation(
                imgaug_augmenter=seq,
                num_joints=16
            )
        else:
            transformations = None

        dataset = Label_Dataset(dataset_path=os.path.join('./data', config.train_date, config.state),
                          config=config,
                          transform=transformations)
        batch_size = config.labeled_batch_size

    elif type == 'unlabeled':
        transformations = TransformFixMatch()
        dataset = UnLabel_Dataset(dataset_path=os.path.join('./data', config.unlabel_train_date, config.state),
                          config=config,
                          transform=transformations,
                          choose=choose)
        batch_size = config.unlabeled_batch_size
    else:
        raise Exception("type is wrong")

    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=batch_size,
                                    num_workers=config.num_workers,
                                    drop_last=drop_last,
                                    shuffle=True,
                                    pin_memory=True)
    return dataloader


def get_test_loader(config, type: str = 'labeled', drop_last=True):
    if type == 'labeled':
        dataset = Label_Dataset(dataset_path=os.path.join('./data', config.test_date, config.state),
                          config=config,
                          transform=None)
    elif type == 'unlabeled':
        transformations = ImageAugmentation(
            imgaug_augmenter=seq,
            num_joints=16
        )
        dataset = Label_Dataset(dataset_path=os.path.join('./data', config.test_date, config.state),
                          config=config,
                          transform=transformations,
                          )
    dataloader = InfiniteDataLoader(dataset,
                                    batch_size=config.test_batch_size,
                                    num_workers=config.num_workers,
                                    drop_last=drop_last,
                                    shuffle=True,
                                    pin_memory=True)
    return dataloader

