import torch
import configargparse
import numpy as np
from subprocess import check_output

#configargparse使用
# parse = configargparse.ArgumentParser()
# parse.add_argument('--config', is_config_file_arg=True, help='config path')
# parse.add_argument('--test1', type=str, default='hello world', help='write something')
# parse.add_argument('--test2', action='store_true', help='yes or no')
# args = parse.parse_args()
# print(args.test1)
# print(args.test2)

#查看poses_bounds.npy文件中的内容
# file_path = r'data\nerf_llff_data\fern\poses_bounds.npy'.replace('\\','/')
# print(file_path)
# data = np.load(file_path)
# print(data.shape)

check_output('cp {}/* {}'.format('data/nerf_llff_data/fern/images', 'test/output'), shell=True)