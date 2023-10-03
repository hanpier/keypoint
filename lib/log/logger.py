from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch
USE_TENSORBOARD = True
try:
    import tensorboardX

    print('Using tensorboardX')
except:
    USE_TENSORBOARD = False

class Logger(object):
    def __init__(self, config, time_str=None):
        """Create a summary writer logging to log_dir."""
        if not os.path.exists(config.save_debug):
            os.makedirs(config.save_debug)

        self.time_str = time_str

        file_name = os.path.join(config.save_debug, 'opt.txt')

        config_file_name = os.path.join('./lib/Config', 'config.py')
        with open(config_file_name, 'rt') as config_file:
            config_content = config_file.read()

        with open(file_name, 'wt') as opt_file:
            opt_file.write('==> torch version: {}\n'.format(torch.__version__))
            opt_file.write('==> cudnn version: {}\n'.format(
                torch.backends.cudnn.version()))
            opt_file.write('\n==> Config:\n')
            opt_file.write(config_content)

        log_dir = config.save_debug + '/logs_{}'.format(time_str)
        if USE_TENSORBOARD:
            self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
        # 确保不会覆盖之前的内容
        if config.resume:
            self.log = open(log_dir + '/log.txt', 'a')
        else:
            self.log = open(log_dir + '/log.txt', 'w')
        try:
            os.system('cp {}/opt.txt {}/'.format(config.save_debug, log_dir))
        except:
            pass
        self.start_line = True

    def write(self, txt):
        if self.start_line:
            self.log.write('{}: {}'.format(self.time_str, txt))
        else:
            self.log.write(txt)
        self.start_line = False
        if '\n' in txt:
            self.start_line = True
            self.log.flush()

    def close(self):
        self.log.close()

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        if USE_TENSORBOARD:
            self.writer.add_scalar(tag, value, step)


# if __name__ == "__main__":
#     # 创建Logger对象
#     logger = Logger()
#
#     # 测试写入日志
#     logger.write("Hello, this is a test log message.")
#     logger.write("This is another log message.")
#     logger.write("A log message with a new line.\nNew line content.")
#
#     # 关闭Logger
#     logger.close()
#
#     # 测试TensorBoard写入
#     if USE_TENSORBOARD:
#         for step in range(10):
#             # 假设有一个损失函数的值和步骤数
#             loss_value = 0.1 * step
#             logger.scalar_summary("Loss", loss_value, step)

