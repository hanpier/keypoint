

##### 切换环境
conda activate point
cd /home/xiewei/home/xiewei/250t_process/ouyangshiyu/semi
export PYTHONPATH=`pwd`:$PYTHONPATH

##### 训练和推理

python train.py # 进行训练和推理

##### tesorboard可视化

tensorboard --logdir='../info/exp' --load_fast true

##### 转换为onnx并推理

python ./tools/pytorch2onnx.py --checkpoint_path None

python -m onnxsim ./tools/tmp.onnx ./tools/tmp-sim.onnx

python ./tools/onnx_inference.py

##### 转换为mnn并推理

python -m MNN.tools.mnnconvert -f ONNX --modelFile ./tools/tmp-sim.onnx --MNNModel ./tools/tmp.mnn --fp16 --bizCode MNN

python ./tools/mnn_inference.py

