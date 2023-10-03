import torch
import onnxruntime as rt
import cv2
import time
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


def preprocess(img):
    input_shape = img.shape
    assert len(input_shape) == 4, 'expect shape like (1, H, W, C)'
    # img = (np.transpose(img, (0, 3, 1, 2)) / 255. - self.mean) / self.std
    img = (np.transpose(img, (0, 3, 1, 2)))
    return img.astype(np.float32)

# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: ",device)
#device = 'cpu'

onnx_weights = './tools/tmp-sim.onnx'

providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if device != 'cpu' else ['CPUExecutionProvider']
session = rt.InferenceSession(onnx_weights, providers=providers)
print(session.get_providers())

img = cv2.imread('./data/07_24/train/imgs/20200414143900254_001377_HB.jpg')
img = cv2.resize(img, (512, 320))
img = img[None,:]
img = preprocess(img)

latency = []
for i in range(50):
    inputs = {session.get_inputs()[0].name: img}
    start = time.time()
    outs = session.run(None, inputs)[0]
    latency.append(time.time() - start)
    # outs = softmax(outs)
    # print(outs)
print("OnnxRuntime {} Inference time = {} ms".format('cuda', format(sum(latency) * 1000 / len(latency), '.2f')))