import torch
import configargparse
import numpy as np

#configargparse使用
# parse = configargparse.ArgumentParser()
# parse.add_argument('--config', is_config_file_arg=True, help='config path')
# parse.add_argument('--test1', type=str, default='hello world', help='write something')
# parse.add_argument('--test2', action='store_true', help='yes or no')
# args = parse.parse_args()
# print(args.test1)
# print(args.test2)

#查看poses_bounds.npy文件中的内容
# data = np.load('data/nerf_llff_data/apple/poses_bounds.npy')
# print(data.shape)