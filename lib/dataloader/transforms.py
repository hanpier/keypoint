import copy
import cv2
import numpy as np
import random
from torchvision import transforms
from imgaug.augmentables import Keypoint, KeypointsOnImage
from imgaug.augmenters import Augmenter, Identity
from lib.dataloader.randaugment import RandAugment


def distort(input_image, keypoints, distortion_probability=-1):
    # 相机内参矩阵
    if random.random() <= distortion_probability:
        k1 = random.uniform(0, 2)
        k2 = random.uniform(0, 2)
        fx = 1000  # 焦距
        fy = 1000
        cx = input_image.shape[1] / 2  # 主点
        cy = input_image.shape[0] / 2

        dist_coeffs = np.array([k1, k2, 0, 0, 0])
        distorted_keypoints = []

        # 针对每个关键点，将坐标映射到校正后的图像坐标空间
        for (x, y) in keypoints:
            undistorted_point = np.array([[x, y]], dtype=np.float32)
            distorted_point = cv2.undistortPoints(undistorted_point,
                                                  cameraMatrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                                                  distCoeffs=dist_coeffs)
            distorted_x = distorted_point[0][0][0] * fx + cx
            distorted_y = distorted_point[0][0][1] * fy + cy
            distorted_keypoints.append((distorted_x, distorted_y))
        undistorted_image = cv2.undistort(input_image, cameraMatrix=np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]]),
                                          distCoeffs=dist_coeffs)
        return undistorted_image, distorted_keypoints
    else:
        # 如果不应用畸变，直接返回原始图像和关键点
        return input_image, keypoints

class ImageAugmentation:
    def __init__(self, imgaug_augmenter: Augmenter = Identity(), num_joints=16):
        self.ia_sequence = imgaug_augmenter
        self.num_joints = num_joints

    def __call__(self, img, label):
        # PIL to array numpy
        img = np.array(img)
        # print(img.shape)
        label = copy.deepcopy(label)

        # Prepare augmentables for imgaug
        keypoints = []

        points = np.array(label, np.float32).reshape(self.num_joints, 2)
        undistorted_image, distorted_keypoints = distort(input_image=img, keypoints=points)
        for i in range(self.num_joints):
            keypoints.append(Keypoint(x=distorted_keypoints[i][0], y=distorted_keypoints[i][1]))
            # print(keypoints)
        # Augmentation
        image_aug, kps_aug = self.ia_sequence(
            image=undistorted_image,
            keypoints=KeypointsOnImage(keypoints, shape=img.shape),
        )

        # Write augmentation back to annotations
        aug_keypoints = []
        for i in range(self.num_joints):
            aug_kp = kps_aug.items.pop(0)
            aug_keypoints.extend([aug_kp.x, aug_kp.y])
        aug_keypoints = np.array(aug_keypoints, np.float32).reshape(self.num_joints, 2)
        return image_aug, aug_keypoints


class TransformFixMatch(object):
    def __init__(self):
        self.weak = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=(704, 1280),
            #                       padding=int(32*0.125),
            #                       padding_mode='reflect')
        ])
        self.strong = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomCrop(size=(704, 1280),
            #                       padding=int(32*0.125),
            #                       padding_mode='reflect'),
            RandAugment(n=2, m=10)])

    def __call__(self, x, choose):
        if(choose == 'weak'):
            weak = self.weak(x)
            return weak
        elif (choose == 'strong'):
            strong = self.strong(x)
            return strong
        elif (choose == 'none'):
            return x

