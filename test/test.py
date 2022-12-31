import torch
import configargparse
import numpy as np
from subprocess import check_output
import imageio.v2 as imageio
import os
import torch.nn as nn

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
# poses = data[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
# bds = data[:, -2:].transpose([1,0])
# print(poses.shape)
# print(bds.shape)
# print(poses[:,4,0])

# check_output('cp {}/* {}'.format('data/nerf_llff_data/fern/images', 'test/output'), shell=True)
# print(np.array((3,4)).reshape([2, 1]))
# def imread(f):
#         if f.endswith('png'):
#             return imageio.imread(f, ignoregamma=True)
#         else:
#             return imageio.imread(f)
# imgdir = 'data/nerf_llff_data/fern/images_8'
# imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]

# imgs = [imread(f)[...,:3]/255. for f in imgfiles]

# imgs = np.stack(imgs, -1) 
# print(imgs.shape)

# a = np.arange(15).reshape((3,5))
# b = -np.arange(12).reshape((3,4))
# print(a)
# print(b)
# a = b
# print(a)
# embed_fns = []
# out_dim = 3
# d = 3
# embed_fns.append(lambda x : x)
# for freq in [1., 2., 4., 8.]:
#             for p_fn in [torch.sin, torch.cos]:
#                 embed_fns.append(lambda x, p_fn=p_fn, freq=freq : p_fn(x * freq))
#                 out_dim += d
# print(out_dim)
# print(len(embed_fns))
# input_ch = 63
# W = 256
# skips = [4]
# D = 8
# input_ch_views = 27
# a = nn.ModuleList(
#             [nn.Linear(input_ch, W)] + [nn.Linear(W, W) if i not in skips else nn.Linear(W + input_ch, W) for i in range(D-1)])
# # print(a)
# # b = nn.ModuleList([nn.Linear(input_ch_views + W, W//2)])
# # print(b)
# # c = nn.Linear(W, W)
# # print(c)
# for i, l in enumerate(a):
#     print(i)
# W = 504
# H = 378
# i, j = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32), indexing='xy')
# print(i)
