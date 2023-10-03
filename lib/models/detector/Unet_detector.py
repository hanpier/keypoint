from ..backbone import U_net
from ..head import Unet_Head
from ..stem import CompressingNetwork
import numpy as np
from torch import nn
import torch
import cv2
from lib.codec.msra_heatmap import MSRAHeatmap
CODECS = {"MSRAHeatmap": MSRAHeatmap}

class Unet_Detector(nn.Module):
    def __init__(self, config, num_classes=16, mode='train'):
        super(Unet_Detector, self).__init__()

        self.stem = CompressingNetwork()

        self.backbone = U_net()

        self.header = Unet_Head(num_classes, mode)


        self.codec = CODECS[config.codec_type](**config.codec_cfg)
        self.config = config
        self._initialize_weights()

    def forward(self, x):

        x = self.stem(x)
        x, feat_x = self.backbone(x)  # n,24,48,48
        x = self.header(x)
        # print([x0.shape for x0 in x])
        return x, feat_x

    def _extract_features(self, x):

        x = self.stem(x)
        x = self.backbone(x)  # n,24,48,48
        x = self.header(x)

        return x

    def _initialize_weights(self):
        self.stem.init_weight()
        self.backbone.init_weight()
        self.header.init_weight()

    def predict(self, items):
        """
        Predict the keypoints on original image by combining the 2 versions of heatmap: the curernt version
        and the flipped version to improve the accuracy
        """
        self.eval()

        batch_keypoints = []
        batch_scores = []
        for i, heatmap in enumerate(items):
            heatmap_np = heatmap.detach().cpu().numpy()
            keypoints, scores = self.codec.decode(heatmap_np)
            batch_keypoints.append(keypoints)
            batch_scores.append(scores)
        return [(np.stack(batch_keypoints, axis=0), np.stack(batch_scores, axis=0))]

    def show_point_on_picture(self, img, _pre):
        # print(landmarks.shape)
        point = _pre[0][0].reshape(-1, 2)
        score = _pre[0][1].reshape(-1, 1)
        for idx, pre_point in enumerate(point):
            point = tuple([int(pre_point[0]), int(pre_point[1])])
            # img = cv2.circle(img, center=point, radius=2, color=(0, 0, 255), thickness=-1)
            if score[idx][0] > self.config.score:
                img = cv2.circle(img, center=point, radius=2, color=(0, 0, 255), thickness=-1)
                cv2.putText(img, str(idx), (point[0] + 5, point[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                            cv2.LINE_AA)
            # cv2.putText(img, str(score[idx][0]), (point[0] - 10, point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            #             cv2.LINE_AA)
        return img
