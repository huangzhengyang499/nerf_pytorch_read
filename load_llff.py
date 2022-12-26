import numpy as np
# import os, imageio
import os
import imageio.v2 as imageio


########## Slightly modified version of LLFF data loading code 
##########  see https://github.com/Fyusion/LLFF for original

### 重新规范化图片的分辨率
### 将images文件夹里的图片缩小到原来的1/8并且保存到images_8文件夹中
# factors=[8]
def _minify(basedir, factors=[], resolutions=[]):
    ### 这一段判断文件夹data/nerf_llff_data/fern/images_8是否存在，若已存在则直接返回，不执行之后的代码了
    needtoload = False
    for r in factors:
        # imgdir = data/nerf_llff_data/fern/images_8
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return
    
    from shutil import copy
    # check_output可以用于执行shell命令
    from subprocess import check_output
    
    ### 这一段主要用于得到data/nerf_llff_data/fern/images文件夹里所有图片的地址，存在imgs里
    # imgdir = data/nerf_llff_data/fern/images
    imgdir = os.path.join(basedir, 'images')
    # 得到images文件夹下所有图片的地址
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    # imgdir_orig = data/nerf_llff_data/fern/images
    imgdir_orig = imgdir
    
    # 得到当前进程的工作目录，wd=C:\Users\huang\Desktop\nerf-pytorch
    wd = os.getcwd()

    ### 这一段将images文件夹里的图片缩小到原来的1/8并且保存到images_8文件夹中
    for r in factors + resolutions:
        if isinstance(r, int):
            # name = images_8
            name = 'images_{}'.format(r)
            # resizearg = (100./8)% = 12.5%
            resizearg = '{}%'.format(100./r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        
        # imgdir = data/nerf_llff_data/fern/images_8
        imgdir = os.path.join(basedir, name)
        # 如果images_8已经存在，则不执行之后的代码
        if os.path.exists(imgdir):
            continue
            
        print('Minifying', r, basedir)
        
        # 创建文件夹data/nerf_llff_data/fern/images_8
        os.makedirs(imgdir)
        # check_output用于执行shell命令：cp data/nerf_llff_data/fern/images/* data/nerf_llff_data/fern/images_8
        # 该指令将images文件夹里的所有图片复制到images_8文件夹中
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)
        
        # ext = 'JPG'
        ext = imgs[0].split('.')[-1]
        # args = 'mogrify -resize 12.5% -format png *.JPG'
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        # 改变当前目录到data/nerf_llff_data/fern/images_8
        os.chdir(imgdir)
        # 执行命令mogrify -resize 12.5% -format png *.JPG
        # 该命令将jpg图片缩小到原来的12.5%并转化为png格式
        check_output(args, shell=True)
        # 改变当前目录到C:\Users\huang\Desktop\nerf-pytorch
        os.chdir(wd)
        
        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
            
        
        
### 加载poses_bounds.npy，得到姿态矩阵poses，深度边界信息bds，所有读取的图片imgs
### 其中poses中高、宽和相机焦距都缩了8倍，图片的RGB值进行了归一化处理
def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):

    # 读取poses_bounds.npy得到poses_arr:(N, 17), N为图片数量
    # 其中，前15个为3*5姿态矩阵，后两个为远/近深度边界信息(视角到场景的最近和最远距离)
    # 对于姿态矩阵，前3*3为旋转矩阵，中3*1为平移向量，（前中3*4矩阵为c2w矩阵）后3*1为图像的高height、宽度width和相机的焦距Focal
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))
    # poses:(N, 15)->(N, 3, 5)->(3, 5, N)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1,2,0])
    # bds:(N, 2)->(2, N)
    bds = poses_arr[:, -2:].transpose([1,0])
    
    # 从images文件夹里取出第一张图片的地址，存在img0中
    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) \
            if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    # 得到img0的shape
    sh = imageio.imread(img0).shape

    sfx = ''

    # factor是下采样倍数，默认是8
    if factor is not None:
        # sfx = _8
        sfx = '_{}'.format(factor)
        # 将images文件夹里的图片缩小到原来的1/8并且保存到images_8文件夹中
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1
    
    # imgdir = 'data/nerf_llff_data/fern/images_8'
    imgdir = os.path.join(basedir, 'images' + sfx)
    if not os.path.exists(imgdir):
        print( imgdir, 'does not exist, returning' )
        return
    
    # 得到images_8文件夹里每一张图片的地址
    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')]
    if poses.shape[-1] != len(imgfiles):
        print( 'Mismatch between imgs {} and poses {} !!!!'.format(len(imgfiles), poses.shape[-1]) )
        return
    
    # 得到images_8文件夹里第一张图片的shape，用来代表所有图片的shape
    sh = imageio.imread(imgfiles[0]).shape
    # 将3*5姿态矩阵中代表图像高和宽的数据换成下采样8倍后的高和宽
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])
    # 将3*5姿态矩阵中代表相机焦距的数据缩小8倍
    # 因为图像下采样了8倍，所以相机参数也需要改变，其中c2w矩阵不变，相机焦距等比例缩放
    poses[2, 4, :] = poses[2, 4, :] * 1./factor
    
    if not load_imgs:
        return poses, bds
    
    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)
    
    # 把images_8文件夹里每个图片的RGB值都/255进行归一化
    imgs = [imread(f)[...,:3]/255. for f in imgfiles]
    # imgs:(h,w,3,N)
    imgs = np.stack(imgs, -1)  
    
    print('Loaded image data', imgs.shape, poses[:,-1,0])
    # poses:(3,5,N),已下采样8倍
    # bds:(2,N)
    # imgs:(h,w,3,N)
    return poses, bds, imgs

    
            
            
    

def normalize(x):
    return x / np.linalg.norm(x)

### view_matrix是一个构造相机矩阵的的函数
### 输入是相机的Z轴朝向、up轴的朝向(即相机平面朝上的方向Y)、以及相机中心，输出3*4的c2w矩阵
def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    # 用Y轴与Z轴叉乘得到X轴
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    # 构造出3*4的c2w矩阵
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3,:3].T, (pts-c2w[:3,3])[...,np.newaxis])[...,0]
    return tt

### 输入多个相机的位姿，得到多个相机的平均位姿
def poses_avg(poses):
    # 取出图像的h、w和相机的焦距信息，每个相机的焦距都是一样的
    hwf = poses[0, :3, -1:]

    # poses的第三列就是相机在世界坐标系中的坐标
    # 计算所有相机的坐标的平均值，得到所有相机的中心坐标center
    center = poses[:, :3, 3].mean(0)
    # 对所有相机的Z轴求求和然后归一化得到vec2向量（方向向量相加其实等效于平均方向向量）
    vec2 = normalize(poses[:, :3, 2].sum(0))
    # 对所有的相机的Y轴求平均得到up向量
    up = poses[:, :3, 1].sum(0)
    # 利用viewmatrix得到所有相机的平均c2w矩阵，然后和hwf合并，得到3*5的相机姿态矩阵
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    
    return c2w


### 这个函数和模型训练没有关系，主要是用来生成一个相机轨迹用于新视角的合成
### 生成一个螺旋式的相机轨迹
def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])
    hwf = c2w[:,4:5]
    
    for theta in np.linspace(0., 2. * np.pi * rots, N+1)[:-1]:
        c = np.dot(c2w[:3,:4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta*zrate), 1.]) * rads) 
        z = normalize(c - np.dot(c2w[:3,:4], np.array([0,0,-focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses
    

### 中心化相机位姿（包括位置和朝向），使每个相机坐标系的XYZ轴朝向和世界坐标系保持一致
def recenter_poses(poses):

    poses_ = poses+0
    bottom = np.reshape([0,0,0,1.], [1,4])
    # 输入多个相机的位姿，得到多个相机的平均位姿
    c2w = poses_avg(poses)
    # 给c2w底下添加上[0,0,0,1.]，得到完整的4*4的c2w矩阵，此时的c2w矩阵是4*4
    c2w = np.concatenate([c2w[:3,:4], bottom], -2)
    bottom = np.tile(np.reshape(bottom, [1,1,4]), [poses.shape[0],1,1])
    # 为poses底下也添加上[0,0,0,1.]，此时的poses变为(N,4,4)
    poses = np.concatenate([poses[:,:3,:4], bottom], -2)

    # 用多个相机的平均位姿c2w的逆左乘poses，目的是使每个相机坐标系的朝向和世界坐标系保持一致
    poses = np.linalg.inv(c2w) @ poses
    poses_[:,:3,:4] = poses[:,:3,:4]
    poses = poses_
    return poses


#####################


def spherify_poses(poses, bds):
    
    p34_to_44 = lambda p : np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1,:], [1,1,4]), [p.shape[0], 1,1])], 1)
    
    rays_d = poses[:,:3,2:3]
    rays_o = poses[:,:3,3:4]

    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0,2,1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0,2,1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)
    
    center = pt_mindist
    up = (poses[:,:3,3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1,.2,.3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:,:3,:4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:,:3,3]), -1)))
    
    sc = 1./rad
    poses_reset[:,:3,3] *= sc
    bds *= sc
    rad *= sc
    
    centroid = np.mean(poses_reset[:,:3,3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad**2-zh**2)
    new_poses = []
    
    for th in np.linspace(0.,2.*np.pi, 120):

        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0,0,-1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)
    
    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0,:3,-1:], new_poses[:,:3,-1:].shape)], -1)
    poses_reset = np.concatenate([poses_reset[:,:3,:4], np.broadcast_to(poses[0,:3,-1:], poses_reset[:,:3,-1:].shape)], -1)
    
    return poses_reset, new_poses, bds
    

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False):
    

    poses, bds, imgs = _load_data(basedir, factor=factor) # factor=8 downsamples original imgs by 8x
    print('Loaded', basedir, bds.min(), bds.max())
    
    # Correct rotation matrix ordering and move variable dim to axis 0
    # 将poses的第0列取负数并与第一列交换顺序，目的是把llff下的相机坐标系变换到nerf中的相机坐标系
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    # poses:(N,3,5)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)
    # imgs:(N,h,w,3)
    imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
    images = imgs
    # bds:(N,2)
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)
    
    # Rescale if bd_factor is provided
    # sc是进行边界放缩的比例
    # 深度边界信息和平移向量一起缩放sc倍，平移向量的含义是相机中心在世界坐标系下的坐标
    sc = 1. if bd_factor is None else 1./(bds.min() * bd_factor)
    poses[:,:3,3] *= sc
    bds *= sc
    
    if recenter:
        # 使每个相机坐标系的XYZ轴朝向和世界坐标系保持一致
        poses = recenter_poses(poses)
        
    if spherify:
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:
        # 求recenter后的poses的一个平均poses
        c2w = poses_avg(poses)
        print('recentered', c2w.shape)
        print(c2w[:3,:4])

        ## Get spiral
        # Get average pose
        up = normalize(poses[:, :3, 1].sum(0))

        # Find a reasonable "focus depth" for this dataset
        close_depth, inf_depth = bds.min()*.9, bds.max()*5.
        dt = .75
        mean_dz = 1./(((1.-dt)/close_depth + dt/inf_depth))
        focal = mean_dz

        # Get radii for spiral path
        shrink_factor = .8
        zdelta = close_depth * .2
        tt = poses[:,:3,3] # ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 90, 0)
        c2w_path = c2w
        N_views = 120
        N_rots = 2
        if path_zflat:
#             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3,3] = c2w_path[:3,3] + zloc * c2w_path[:3,2]
            rads[2] = 0.
            N_rots = 1
            N_views/=2

        # Generate poses for spiral path
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)
        
    # render_poses:(120,3,5)，共120个视角，每个视角的shape是(3,5)
    render_poses = np.array(render_poses).astype(np.float32)

    c2w = poses_avg(poses)
    print('Data:')
    print(poses.shape, images.shape, bds.shape)
    
    # 计算每个相机中心与平均相机中心的距离
    dists = np.sum(np.square(c2w[:3,3] - poses[:,:3,3]), -1)
    # 获得测试集的id，dists最小的那个相机
    i_test = np.argmin(dists)
    print('HOLDOUT view is', i_test)
    
    images = images.astype(np.float32)
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test


# if __name__ == "__main__":
#     poses, bds, imgs = _load_data("data/nerf_llff_data/fern", factor=8)
#     images, poses, bds, render_poses, i_test = load_llff_data("data/nerf_llff_data/fern", factor=8,
#                                                                   recenter=True, bd_factor=.75,
#                                                                   spherify=False)

    
