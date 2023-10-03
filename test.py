from lib.models.backbone.Bisetnetv1 import BiSeNetV1
from lib.models.backbone.Unet import U_net
from thop import profile
from lib.models.stem.comprenet import CompressingNetwork
from lib.models.detector import Unet_Detector
import torch
if __name__ == "__main__":
    from torchsummary import summary

    # model = CompressingNetwork().cuda()
    # print(summary(model, (3, 640, 1280)))

    # model = MoveNet_Backbone().cuda()
    # print(summary(model, (3, 704, 1280)))

    import torch  # 命令行是逐行立即执行的

    # content = torch.load('/home/xiewei/home/xiewei/250t_process/ouyangshiyu/semi/weights/2023-09-18-10-48/checkpoint_best.pt')
    # print(content['state_dict'])

    model = U_net().cuda()
    print(summary(model, (16, 176, 320)))
    #
    input = torch.randn(1, 16, 320, 512).cuda()
    macs, params = profile(model, inputs=(input,))
    print('the flops is {}G,the params is {}M'.format(round(macs / (10 ** 9), 2), round(params / (10 ** 6), 2)))

    import pickle


    # 现在，'data' 变量包含了从 .pkl 文件中加载的数据，你可以根据数据的类型和结构来进一步处理它

    # model = Unet_Rle_Detector(in_channel=3).cuda()
    # print(summary(model, (3, 704, 1280)))
    # dummy_input1 = torch.randn(1, 3, 704, 1280).cuda()
    # input_names = ["input1"]  # 自己命名
    # output_names = ["output1"]
    #
    # torch.onnx.export(model, dummy_input1, "pose.onnx",
    #                   verbose=True, input_names=input_names, output_names=output_names,
    #                   do_constant_folding=True, opset_version=11)