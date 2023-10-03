import os
import json
import numpy as np
import math
import matplotlib.pyplot as plt
from ..Config.movenet_config import config_loc as cfg
from scipy.ndimage import gaussian_filter
import cv2
import PIL
from torchvision import transforms
import torch
from PIL import Image, ImageFont, ImageDraw
from itertools import product
# os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
_range_weight_x = np.array([[x for x in range(48)] for _ in range(48)])
_range_weight_y = _range_weight_x.T
scale = cfg['scale']
img_size = [cfg['input_h'],cfg['input_w']]
heatmap_size = [cfg['input_h'] // scale,cfg['input_w'] // scale] #(h, w)
# json变成加入高斯的np

from PIL import Image, ImageOps


def letterbox_image(image, keypoints, target_size):
    # 获取原始图像的尺寸
    img_width, img_height = image.size

    # 计算填充尺寸
    width_ratio = target_size[0] / img_width
    height_ratio = target_size[1] / img_height
    ratio = min(width_ratio, height_ratio)
    if width_ratio < height_ratio:
        new_width = target_size[0]
        new_height = int(img_height * width_ratio)
    else:
        new_width = int(img_width * height_ratio)
        new_height = target_size[1]

    # 调整图像大小并进行填充
    image = image.resize((new_width, new_height), Image.BICUBIC)
    new_keypoints = keypoints / ratio
    new_img = Image.new('RGB', target_size, (128, 128, 128))  # 填充为灰色
    new_img.paste(image, (0, 0))

    return new_img, new_keypoints

def json_to_numpy(dataset_path):
    with open(dataset_path) as fp:
        json_data = json.load(fp)
        points = json_data['shapes']

    # print(points)
    landmarks = []
    for point in points:
        for p in point['points']:
            landmarks.append(p)

    # print(landmarks)
    landmarks = np.array(landmarks, dtype=np.int)
    landmarks = landmarks.reshape(-1, 2)

    # 保存为np
    # np.save(os.path.join(save_path, name.split('.')[0] + '.npy'), landmarks)

    return landmarks

def generate_heatmaps(landmarks, height, width, sigma=cfg['sigma']):
    _gaussians = {}
    heatmaps = []
    w = width
    h = height
    for points in landmarks:
        heatmap = np.zeros((height, width))
        points = points.reshape(-1, 2)
        for point in points:
            mu_x, mu_y = int(point[0]),int(point[1])
            # print("mu_x, mu_y: ",mu_x, mu_y)
            tmp_size = 3 * sigma
            # Top-left
            # print(mu_x - tmp_size, mu_y - tmp_size)
            x1, y1 = mu_x - tmp_size, mu_y - tmp_size
            # print(x1,y1)

            # Bottom right
            x2, y2 = mu_x + tmp_size + 1, mu_y + tmp_size + 1
            # print("x2, y2: ",x2, y2)
            if x1 >= w or y1 >= h or x2 < 0 or y2 < 0:
                return heatmaps

            size = 2 * tmp_size + 1
            tx = np.arange(0, size, 1, np.float32)
            ty = tx[:, np.newaxis]
            x0 = y0 = size // 2

            # The gaussian is not normalized, we want the center value to equal 1
            g = _gaussians[sigma] if sigma in _gaussians \
                else np.exp(- ((tx - x0) ** 2 + (ty - y0) ** 2) / (2 * sigma ** 2))
            _gaussians[sigma] = g

            # Determine the bounds of the source gaussian
            g_x_min, g_x_max = max(0, -x1), min(x2, w) - x1
            g_y_min, g_y_max = max(0, -y1), min(y2, h) - y1

            # Image range
            img_x_min, img_x_max = max(0, x1), min(x2, w)
            img_y_min, img_y_max = max(0, y1), min(y2, h)
            # print([g_y_min,g_y_max, g_x_min,g_x_max])

            heatmap[img_y_min:img_y_max, img_x_min:img_x_max] = \
                g[g_y_min:g_y_max, g_x_min:g_x_max]
            am = np.amax(heatmap)
            if cfg['model'] == 'Unet':
                heatmap /= am / 255
            elif cfg['model'] == 'Move':
                heatmap /= am / 255
            else:
                ValueError("无效的模型类型")
        heatmaps.append(heatmap)

    heatmaps = np.array(heatmaps)
    return heatmaps

def generate_offset_heatmap(heatmap_size, keypoints, keypoints_visible, radius_factor):
    """Generate offset heatmaps of keypoints, where each keypoint is
    represented by 3 maps: one pixel-level class label map (1 for keypoint and
    0 for non-keypoint) and 2 pixel-level offset maps for x and y directions
    respectively.

    Args:
        heatmap_size (Tuple[int, int]): Heatmap size in [W, H]
        keypoints (np.ndarray): Keypoint coordinates in shape (N, K, D)
        keypoints_visible (np.ndarray): Keypoint visibilities in shape
            (N, K)
        radius_factor (float): The radius factor of the binary label
            map. The positive region is defined as the neighbor of the
            keypoint with the radius :math:`r=radius_factor*max(W, H)`

    Returns:
        tuple:
        - heatmap (np.ndarray): The generated heatmap in shape
            (K*3, H, W) where [W, H] is the `heatmap_size`
        - keypoint_weights (np.ndarray): The target weights in shape
            (K,)
    """

    N, K, _ = keypoints.shape
    W, H = heatmap_size

    heatmaps = np.zeros((K, 3, H, W), dtype=np.float32)
    keypoint_weights = keypoints_visible.copy()

    # xy grid
    x = np.arange(0, W, 1)
    y = np.arange(0, H, 1)[:, None]

    # positive area radius in the classification map
    radius = radius_factor * max(W, H)

    for n, k in product(range(N), range(K)):
        if keypoints_visible[n, k] < 0.5:
            continue

        mu = keypoints[n, k]

        x_offset = (mu[0] - x) / radius
        y_offset = (mu[1] - y) / radius

        heatmaps[k, 0] = np.where(x_offset**2 + y_offset**2 <= 1, 1., 0.)
        heatmaps[k, 1] = x_offset
        heatmaps[k, 2] = y_offset

    heatmaps = heatmaps.reshape(K * 3, H, W)

    return heatmaps, keypoint_weights


def show_heatmap(heatmaps):
    for i, heatmap in enumerate(heatmaps):
        plt.imshow(heatmap, cmap='hot', interpolation='nearest')
        print(f'./figure/heatmap_{i}.png')
        plt.savefig(f'./figure/heatmap_{i}.png')  # 保存为PNG文件，文件名带有索引i
        plt.close()  # 关闭当前图形，准备下一个循环


def label2reg(keypoints, cx, cy):
    regs = np.zeros((len(keypoints) * 2, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    # print(keypoints)
    for i in range(len(keypoints)):
        # if keypoints[i + 2] == 0:
        #     continue

        x = keypoints[i][0]
        y = keypoints[i][1]
        if x == heatmap_size[0]: x = (heatmap_size[0] - 1)
        if y == heatmap_size[1]: y = (heatmap_size[1] - 1)
        if x > heatmap_size[0] or x < 0 or y > heatmap_size[1] or y < 0:
            continue

        reg_x = x - cx
        reg_y = y - cy

        for j in range(cy - 2, cy + 3):
            if j < 0 or j > heatmap_size[1] - 1:
                continue
            for k in range(cx - 2, cx + 3):
                if k < 0 or k > heatmap_size[0] - 1:
                    continue
                if cx < heatmap_size[0] / 2 - 1:
                    regs[i * 2][j][k] = reg_x - (cx - k)
                    m = reg_x - (cx - k)
                else:
                    regs[i * 2][j][k] = reg_x + (cx - k)
                    n = reg_x + (cx - k)
                if cy < heatmap_size[1] / 2 - 1:
                    regs[i * 2 + 1][j][k] = reg_y - (cy - j)
                else:
                    regs[i * 2 + 1][j][k] = reg_y + (cy - j)
    return np.array(regs)


def label2offset(keypoints, cx, cy, regs):
    offset = np.zeros((len(keypoints) * 2, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    # print(keypoints)
    # print(regs.shape)#(14, 48, 48)
    for i in range(len(keypoints)):
        # if keypoints[i * 3 + 2] == 0:
        #     continue

        large_x = int(keypoints[i][0])
        large_y = int(keypoints[i][1])

        small_x = int(regs[i * 2, cy, cx] + cx)
        small_y = int(regs[i * 2 + 1, cy, cx] + cy)

        offset_x = large_x / scale - small_x
        offset_y = large_y / scale - small_y

        if small_x == heatmap_size[0] : small_x = (heatmap_size[0] - 1)
        if small_y == heatmap_size[1] : small_y = (heatmap_size[1] - 1)
        if small_x > heatmap_size[0] or small_x < 0 or small_y > heatmap_size[1] or small_y < 0:
            continue
        # print(offset_x, offset_y)

        # print()
        offset[i * 2][small_y][small_x] = offset_x
        offset[i * 2 + 1][small_y][small_x] = offset_y

    # print(heatmaps.shape)

    return np.array(offset)

def label2offset2(keypoints, scaled_keypoints):
    offset = np.zeros((len(keypoints) * 2, heatmap_size[0], heatmap_size[1]), dtype=np.float32)
    # for i in range(len(keypoints)):
    #     # if keypoints[i * 3 + 2] == 0:
    #     #     continue
    #
    #     x = keypoints[i][0]
    #     y = keypoints[i][1]
    #
    #     small_x = scaled_keypoints[i][0]
    #     small_y = scaled_keypoints[i][1]
    #
    #     offset_x = x - small_x * 4
    #     offset_y = y - small_y * 4
    #
    #     offset[i * 2][y][x] = offset_x
    #     offset[i * 2 + 1][y][x] = offset_y

    return np.array(offset)


def heatmap_to_point(heatmaps):
    scale = cfg['scale']

    points = []

    for heatmap in heatmaps:
        pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        point0 = pos[0]
        point1 = pos[1]
        points.append([point1, point0])
    return np.array(points, dtype=np.float32)


def show_inputImg_and_keypointLabel(imgPath, heatmaps):
    points = []
    for heatmap in heatmaps:
        pos = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        points.append([pos[1], pos[0]])

    img = PIL.Image.open(imgPath).convert('RGB')
    img = transforms.ToTensor()(img)  #
    img = img[:, :cfg['cut_h'], :cfg['cut_w']]

    img = img.unsqueeze(0)  # 增加一维
    resize = torch.nn.Upsample(scale_factor=(1, 1), mode='bilinear', align_corners=True)
    img = resize(img)

    img = img.squeeze(0)  # 减少一维

    print(img.shape)

    img = transforms.ToPILImage()(img)
    draw = ImageDraw.Draw(img)
    for point in points:
        print(point)
        draw.point((point[0], point[1]), fill='yellow')

    # 保存
    img.save(os.path.join('..','show', 'out.jpg'))



if __name__ == '__main__':
    landmarks = json_to_numpy('/home/cidi/桌面/Car/data/08_21/test/labels/20200414144431527_003218_HB.json')
    print('关键点坐标', landmarks, '-------------', sep='\n')
    landmarks.reshape(1,-1)
    m = np.mean(landmarks,axis = 0)
    print(m)
    # heatmaps = generate_heatmaps(landmarks, cfg['input_h'], cfg['input_w'],sigma=10)
    # heatmaps = generate_heatmaps(landmarks, cfg['input_h'], cfg['input_w'], (cfg['gauss_h'], cfg['gauss_w']))
    # print(heatmaps)
    # print(heatmaps.shape)

    # show heatmap picture
    # show_heatmap(heatmaps)
    #
    # a = heatmap_to_point(heatmaps)
    #
    # # show cut image and the keypoints
    # show_inputImg_and_keypointLabel('../data/08_11_in/train/imgs/20200414144428035_003197_HB.jpg', heatmaps)
