"""Utility functions for DiffPooling and GCNPooling evaluation, Can be use for matching postprocess"""

import numpy as np
import argparse
from knn_cuda import KNN
import torch
import random

def _calc_distances(p0, points):
    return ((p0 - points) ** 2).sum(axis=1)


def _fps_from_given_pc(pts, k, given_pc):
    farthest_pts = np.zeros((k, 3))
    t = np.size(given_pc) // 3 # 整除，得到目前已采样的点数
    farthest_pts[0:t] = given_pc    # 将已采样的点放进集合中，接下来FPS采剩余的点
    complete_num = k - t # by cxl 统计需要补全多少点

    distances = _calc_distances(farthest_pts[0], pts) # 计算第一个采样点和原始点的距离，distance为 采样子集 距离 其余点 的最小距离，shape=[npoint,]
    for i in range(1, t):   # 再计算剩余的采样点和原始点的距离，并更新距离矩阵
        distances = np.minimum(distances, _calc_distances(farthest_pts[i], pts)) 

    for i in range(t, k):
        farthest_pts[i] = pts[np.argmax(distances)]
        distances = np.minimum(distances, _calc_distances(farthest_pts[i], pts))

    return farthest_pts, complete_num

def _rs_from_given_pc(other_pc, k, given_pc):
    random_pts = np.zeros((k, 3))
    t = np.size(given_pc) // 3 # 整除，得到目前已采样的点数
    random_pts[0:t] = given_pc    # 将已采样的点放进集合中，接下来FPS采剩余的点
    complete_num = k - t # by cxl 统计需要补全多少点
    if complete_num > 0:
        random_pts[t:k] = random.sample(other_pc.tolist(),complete_num)

    return random_pts, complete_num

def _unique(arr):
    _, idx = np.unique(arr, return_index=True)
    return arr[np.sort(idx)]


def nn_matching(full_pc, idx, k, complete="fps"):
    complete_num = 0 # by cxl 默认没有点需要补全, 需要batchsize=1时才能正确计算
    batch_size = np.size(full_pc, 0)
    out_pc = np.zeros((full_pc.shape[0], k, 3))
    for ii in range(0, batch_size):
        best_idx = idx[ii]
        if complete == "fps":
            best_idx = _unique(best_idx)   # 若有多个点match到同一个点上则用FPS来取剩下所需的点
            out_pc[ii], complete_num = _fps_from_given_pc(full_pc[ii], k, full_pc[ii][best_idx])
        elif complete == "random":
            best_idx = _unique(best_idx)
            other_pc = np.delete(full_pc[ii],best_idx,axis=0) 
            out_pc[ii], complete_num = _rs_from_given_pc(other_pc , k, full_pc[ii][best_idx])
        else:
            out_pc[ii] = full_pc[ii][best_idx]
    return out_pc[:, 0:k, :], complete_num


# fmt: off
def get_parser():
    parser = argparse.ArgumentParser("SampleNet: Differentiable Point Cloud Sampling")

    parser.add_argument("--skip-projection", action="store_true", help="Do not project points in training")

    parser.add_argument("-in", "--num-in-points", type=int, default=1024, help="Number of input Points [default: 1024]")
    parser.add_argument("-out", "--num-out-points", type=int, default=64, help="Number of output points [2, 1024] [default: 64]")
    parser.add_argument("--bottleneck-size", type=int, default=128, help="bottleneck size [default: 128]")
    parser.add_argument("--alpha", type=float, default=0.01, help="Simplification regularization loss weight [default: 0.01]")
    parser.add_argument("--gamma", type=float, default=1, help="Lb constant regularization loss weight [default: 1]")
    parser.add_argument("--delta", type=float, default=0, help="Lb linear regularization loss weight [default: 0]")

    # projection arguments
    parser.add_argument("-gs", "--projection-group-size", type=int, default=8, help='Neighborhood size in Soft Projection [default: 8]')
    parser.add_argument("--lmbda", type=float, default=0.01, help="Projection regularization loss weight [default: 0.01]")

    return parser
# fmt: on

# Create by chenxiaolei
def nn_match(x,y,num_out_points,complete="fps"):
    """x is raw input point, y is simple point and will be match to x"""
    # input need B 3 N（注意以列为点）
    _, idx = KNN(1, transpose_mode=False)(x.contiguous(), y.contiguous())  # 用KNN找最近的1个点，idx=[1,1,32]
    # Convert to numpy arrays in B x N x 3 format. we assume 'bcn' format.
    x = x.permute(0, 2, 1).cpu().detach().numpy()
    y = y.permute(0, 2, 1).cpu().detach().numpy()

    idx = idx.cpu().detach().numpy()
    idx = np.squeeze(idx, axis=1)

    #debug
    # print('idx=',idx)
    z, complete_num = nn_matching(
        x, idx, num_out_points, complete=complete
    )
    # Matched points are in B x N x 3 format.
    match = torch.tensor(z, dtype=torch.float32).cuda()

    return match, complete_num


# debug
if __name__ == '__main__':
    x = torch.tensor([[[0.5,0.1,0.1],[0.3,0.2,0.1],[0.4,0.4,0.4],[0.2,0.1,0.1]],
                        [[0.1,0.1,0.1],[0.2,0.2,0.2],[0.3,0.3,0.3],[0.2,0.2,0.1]]])
    print('x=',x)
    y = torch.tensor([[[0.35,0.51,0.21],[0.34,0.50,0.18]],
                        [[0.16,0.13,0.11],[0.42,0.27,0.25]]])
    print('y=',y)
    x,y = x.cuda(),y.cuda()
    sampled, _= nn_match(x.transpose(2,1),y.transpose(2,1),num_out_points=2,complete="random")
    print('sampled point=',sampled)
