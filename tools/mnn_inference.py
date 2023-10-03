import numpy as np
import MNN
import cv2
import os
import time
class Pose():
    def __init__(self, model_path, joint_num=16, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.model_path = model_path
        self.joint_num = joint_num
        self.mean = np.array(mean).reshape(1, -1, 1, 1)
        self.std = np.array(std).reshape(1, -1, 1, 1)
        a = os.path.exists(model_path)
        self.interpreter = MNN.Interpreter(model_path)
        self.model_sess = self.interpreter.createSession({
            'numThread': 1
        })

    def preprocess(self, img):
        input_shape = img.shape
        assert len(input_shape) == 4, 'expect shape like (1, H, W, C)'
        # img = (np.transpose(img, (0, 3, 1, 2)) / 255. - self.mean) / self.std
        img = (np.transpose(img, (0, 3, 1, 2)))
        return img.astype(np.float32)

    def inference(self, img):
        input_shape = img.shape
        assert len(input_shape) == 4, 'expect shape like (1, C, H, W)'

        input_tensor = self.interpreter.getSessionInput(self.model_sess)

        tmp_input = MNN.Tensor(input_shape,
                               MNN.Halide_Type_Float,
                               img.astype(np.float32),
                               MNN.Tensor_DimensionType_Caffe)
        input_tensor.copyFrom(tmp_input)

        self.interpreter.runSession(self.model_sess)
        output_tensor = self.interpreter.getSessionOutputAll(self.model_sess)

        joint_coord = np.array(output_tensor['297'].getData())

        return joint_coord

    def post_process(self, coords, bbox):
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        target_coords = coords * np.array([w, h])
        target_coords += np.array([bbox[0], bbox[1]])
        return target_coords

    def predict(self, img):
        img = self.preprocess(img)
        # print(img.shape)
        joint_coord = self.inference(img)
        # joint_coord = self.post_process(joint_coord)
        return joint_coord

img = cv2.imread('./data/07_24/train/imgs/20200414143900254_001377_HB.jpg')
# img = img[:704, :, :]
img = cv2.resize(img, (704, 1280))
inputs = img[None,:]
mnn_model = Pose('./tools/tmp.mnn')
for x in range(5):
    s = time.time()
    for i in range(100):
        mnn_model.predict(inputs)
    print(f'elapse {(time.time() - s)*10:.4f} ms')

