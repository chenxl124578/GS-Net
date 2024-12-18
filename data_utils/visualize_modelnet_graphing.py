'''
Visualize modelnet graphing process 
Using open3d, so need to run in local instead remote servers
'''

import os
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import pathlib
import shutil
from tqdm import tqdm
import matplotlib.pyplot as plt
import open3d

from ModelNetDataLoader import ModelNetDataSet
from graph_construction import *

torch.manual_seed(0)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROOT_DIR = BASE_DIR

def show_point(points):
    """
    :param src_points: [N, 3] src_points.
    :param des_points: [N, 3] des_points.
    :param edges: [M, 2],M pairs of connections src_points[edges[0]] -> des_points[edges[1]]
    :return: None
    """
    # line_set = open3d.geometry.LineSet()
    colors = [[1, 0, 0] for i in range(len(points))]

    pcd=open3d.geometry.PointCloud()
    pcd.points= open3d.utility.Vector3dVector(points)

    pcd.colors = open3d.utility.Vector3dVector(colors)

    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 10  # 点云大小
        for geometry in geometry_list:
            vis.add_geometry(geometry)

        # vis.update_geometry()
        # vis.poll_events()
        # vis.update_renderer()
        vis.run()
        vis.destroy_window()
    custom_draw_geometry_load_option([pcd])

    # open3d.visualization.draw_geometries([pcd])

def show_graph(points, edges):
    """
    :param src_points: [N, 3] src_points.
    :param des_points: [N, 3] des_points.
    :param edges: [M, 2],M pairs of connections src_points[edges[0]] -> des_points[edges[1]]
    :return: None
    """
    # line_set = open3d.geometry.LineSet()
    edges = edges.T
    edge_colors = [[1, 0, 0] for i in range(len(edges))]
    point_colors = [[1, 0, 0] for i in range(len(points))]

    # pcd=open3d.geometry.PointCloud()
    # pcd.points= open3d.utility.Vector3dVector(points)
    pcd=open3d.geometry.PointCloud()
    pcd.points= open3d.utility.Vector3dVector(points)

    pcd.colors = open3d.utility.Vector3dVector(point_colors)

    line_set = open3d.geometry.LineSet(
        points=open3d.utility.Vector3dVector(points),
        lines=open3d.utility.Vector2iVector(edges),
    )
    line_set.colors = open3d.utility.Vector3dVector(edge_colors)

    # open3d.visualization.draw_geometries([line_set,pcd])

    # line_set.points = open3d.utility.Vector3dVector(points)
    # line_set.lines = open3d.utility.Vector2iVector(edges)

    # add color if wanted
    # line_set.colors = np.zeros((edges.shape[0], 3), dtype=np.float32)
    
    def custom_draw_geometry_load_option(geometry_list):
        vis = open3d.visualization.Visualizer()
        vis.create_window()
        vis.get_render_option().point_size = 10  # 点云大小
        for geometry in geometry_list:
            vis.add_geometry(geometry)

        # vis.update_geometry()
        # vis.poll_events()
        # vis.update_renderer()
        vis.run()
        vis.destroy_window()
    custom_draw_geometry_load_option([pcd,line_set])

    # custom_draw_geometry_load_option([line_set])

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    # parser.add_argument('--gpu', type=str, default='1', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40],  help='training on ModelNet10/40')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--save_dir', type=str, default='data/ModelNet40_visualization_debug/', help='experiment root')
    parser.add_argument('--load_data', type=str, default='test',help='test or train dataset')
    parser.add_argument('--unset_axis', action='store_true', default=False, help='if debug set False and just use 128 sample, if ready to train set True')

    # debug true or false
    parser.add_argument('--not_debug', action='store_true', default=False, help='if debug set False and just use 128 sample, if ready to train set True')
    return parser.parse_args()

def draw(x, y, z, name, file_dir, color=None, unset_axis=False):
    """
    绘制单个样本的三维点图
    """
    classname = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table',
                'tent','toilet','tv_stand','vase','wardrobe','xbox']
    if color is None:
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name+'-{}.png'.format(i)
            save_name = file_dir.joinpath(save_name)
            ax.scatter(x[i], y[i], z[i], c='r')
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            # plt.show()
    else:
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'tan', 'orangered', 'lightgreen', 'coral', 'aqua', 'gold', 'plum', 'khaki', 'cyan', 'crimson', 'lawngreen', 'thistle', 'skyblue', 'lightblue', 'moccasin', 'pink', 'lightpink', 'fuchsia', 'chocolate', 'tomato', 'orchid', 'grey', 'plum', 'peru', 'purple', 'teal', 'sienna', 'turquoise', 'violet', 'wheat', 'yellowgreen', 'deeppink', 'azure', 'ivory', 'brown']
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  # 创建一个三维的绘图工程
            save_name = name + '-{}-{}({}).png'.format(i, color[i],classname[color[i]])
            save_name = file_dir.joinpath(save_name)
            ax.scatter(x[i], y[i], z[i], c=colors[color[i]])

            # 优化坐标系
            if not unset_axis:
                ax.set_xlim(-1.2,1.2)
                ax.set_ylim(-1.2,1.2)
                ax.set_zlim(-1.2,1.2)
                ax.set_xticks(np.arange(-1.2, 1.2, 0.4))
                ax.set_yticks(np.arange(-1.2, 1.2, 0.4))
                ax.set_zticks(np.arange(-1.2, 1.2, 0.4))
                ax.set_zlabel('Z')  # 坐标轴
                ax.set_ylabel('Y')
                ax.set_xlabel('X')
            else:
                plt.axis('off')
            plt.draw()
            plt.savefig(save_name)
            # plt.show()


def main(args):

    save_dir = Path(args.save_dir)
    save_dir.mkdir(exist_ok=True)
    save_dir = save_dir.joinpath(args.load_data)
    save_dir.mkdir(exist_ok=True)



    DATA_DIR = os.path.join(BASE_DIR, "data")
    data_path = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")
    dataset = ModelNetDataSet(root=data_path, num_point=args.num_point, split=args.load_data,not_debug=args.not_debug)
    dataLoader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    for batch_id, (points, target) in tqdm(enumerate(dataLoader, 0), total=len(dataLoader), smoothing=0.9):
        if args.batch_size == 1:
            target = target.view(1)
        else:
            target = np.squeeze(target)
        
        adj = get_radius_sparse_graph(x=points,r=0.1)
        points = np.squeeze(points)
        # points = points[:256]   
        # points = points.permute(0,2,1)  # need to be -> b c n
        
        save_name_prefix = 'input-{}'.format(batch_id)
        show_point(points)
        show_graph(points,adj)

        # draw(points[:, 0, :], points[:, 1, :], points[:, 2, :], save_name_prefix, save_dir, color=target, unset_axis=args.unset_axis)

if __name__ == '__main__':
    print("begin running visulize_modelnet_graphing.py")
    args = parse_args()
    main(args)




