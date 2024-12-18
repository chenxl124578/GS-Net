'''
By Xiaolei Chen
Create in 2022/5/1
use to test model with sampled point of raw input
'''

import os
from random import sample
from time import time

gpu_id = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id    # GPU setting
from models.src.sputils import nn_match
import sys
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import importlib
import shutil
import provider
import matplotlib.pyplot as plt

from data_utils import ply


import numpy as np
import torch
import torch.nn as nn
# import tensorboardX
# from tensorboardX import SummaryWriter

from models.gsnet_pyg import GSNet
from data_utils.ModelNetDataLoader import ModelNetDataSet
from data_utils.ScanobjectNN_dataloader import ScanobjectNNDataSet
from data_utils.semanticKITTI_cls_loader import SemanticKITTI_cls_DataSet

# from data_utils.ModelNetDataLoader_gnn import ModelNetDataSet

from data_utils.graph_construction import *
from data_utils import data_prep_util

# torch.manual_seed(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models import pointnet_cls

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('evaluating')
    # parser.add_argument('--gpu', type=str, default='2', help='specify gpu device')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='modelnet40 , scanobjectnn , semantickitti_cls')
    # scanobjectnn
    parser.add_argument('--dataclass', type=str, default='OBJ_BG', help='OBJ_BG, PB_T25, PB_T25_R, PB_T50_R, PB_T50_RS')
    # semantickitti_cls
    parser.add_argument('--keep_reflect', action='store_true', default=False, help='True or False to control the feature dimention of the pointclouds. True:4, False:3')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--classifier_model', default='pointnet_cls', help='model name [default: pointnet_cls],or pointnet_cls_vsTF, or dgcnn_cls')
    parser.add_argument('--classifier_model_path', default='weights/acc88_45_PointNet_classifier_model_modelnet40.pth', help='Path to model.pth file of a pre-trained classifier')
    parser.add_argument('--sampler_model', default='gsnet_pyg', help='Sampler model name: diffpool_gnn')
    parser.add_argument('--sampler_model_path', default='log/gsnet_pyg/logdir/checkpoints/best_model.pth', help='Path to model.pth file of a pre-trained sampler')

    parser.add_argument('--num_category', default=40, type=int, choices=[6, 40, 15],  help='training on ModelNet40 or scanobjectsNN or semantickitti_cls')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')

    parser.add_argument('--test_data', type=str, default='test',help='test or train dataset')
    parser.add_argument('--save_sampled_point', action='store_true', default=False, help='if True, save sampled point to log_eval')
    parser.add_argument('--vis_sampled_point', action='store_true', default=False, help='if True, save sampled point to log_eval')

    # debug true or false
    parser.add_argument('--not_debug', action='store_true', default=False, help='if debug set False and just use 128 sample, if ready to train set True')

    # diffpool sampler arguments
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim [default: 30]')
    parser.add_argument('--output_dim', type=int, default=128, help=' [default: 30]')
    parser.add_argument('--num_gc_layers', type=int, default=3, help='[default: 3]')
    parser.add_argument('--assign_ratio', type=float, default=0.03125, help=' 0.03125: input 1024 output 32 point; 0.5: 1024->512->256...->32')
    parser.add_argument('--num_pool', type=int, default=1, help='Number of pooling [default: 1],1or5')
    parser.add_argument('--dropout', type=float, default=0.0, help='[default: 0.0]')
    parser.add_argument('--linkpred', action='store_true',default=False, help='[default: False]')
    parser.add_argument('--simplification_loss', action='store_true', default=False, help='[default: True]')
    
    parser.add_argument('--assign_input_dim', type=int, default=3, help='[default: 3]')
    parser.add_argument('--alpha', type=float, default=20, help='Simplification regularization loss weight [default: 30]')

    parser.add_argument('--diffout_normalize_dim', type=int,default=1, help='diffpool output normalize, 0 not normalize, 1 or 2 normalize dim 1 or 2')
    parser.add_argument('--assign_norm', type=str,default='none', help='assign normalize, None "norm" "bn" or "layernorm"')
    parser.add_argument('--bn_format', type=str,default='features', help='BN input channel name , "none" "num_nodes"  or "features"')

    # inference 
    parser.add_argument('--match',action='store_true',default=False,help='Match simpled point to raw point')
    parser.add_argument('--complete_method',type=str,default="fps",help='set fps or random for completeness')
    
    parser.add_argument('--record_overlap_point',action='store_true',default=False,help='record the num of overlap point in generate points')
    parser.add_argument('--save_retrieval_vectors',action='store_true',default=False,help='record the num of overlap point in generate points')
    parser.add_argument('--points_noise',type=float,default=0,help='noise standard deviation(sigma in gauss noise)')

    # dgcnn_k
    parser.add_argument('--k', type=int ,default=1, help='when --model == dgcnn_cls, num of nearest neighbors to use')

    
    # projection arguments
    # parser.add_argument("--projection_group_size", type=int, default=7, help='Neighborhood size in Soft Projection [default: 7]')
    # parser.add_argument('--lmbda', type=float, default=1, help='Projection regularization loss weight [default: 1]')
    # parser.add_argument("--skip_projection", action="store_true", help="Do not project points in training")


    return parser.parse_args()

def create_model(args):
    cls = importlib.import_module(args.classifier_model)
    if args.classifier_model == "pointnet_cls":
        classifier = cls.get_model(args.num_category, normal_channel=args.use_normals)
    elif args.classifier_model == "dgcnn_cls":
        classifier = cls.DGCNN(args.k, emb_dims=1024, output_channels=args.num_category)


    classifier.requires_grad_(False)
    classifier.eval()

    # Create sampling network
    sampler = GSNet(max_num_nodes=args.num_point,input_dim=3,output_dim=3,hidden_dim=args.hidden_dim,
                                embedding_dim=args.output_dim,assign_ratio=args.assign_ratio)
    sampler.requires_grad_(False)
    sampler.eval()   

    if args.classifier_model_path is not None:
        classifier.load_state_dict(torch.load(args.classifier_model_path)['model_state_dict'])
        print('Use classifier model from %s' % args.classifier_model_path)
    else:
        raise ValueError

    classifier.sampler = sampler

    if args.sampler_model_path is not None:
        classifier.sampler.load_state_dict(torch.load(args.sampler_model_path)['model_state_dict'])
        print('Use sampler model from %s' % args.sampler_model_path)
    else:
        raise ValueError

    return classifier

def draw(x, y, z, name, file_dir,points_np, classname, color=None):
    """
    绘制单个样本的三维点图
    """
    xp,yp,zp=points_np[:,:, 0],points_np[:, :, 1],points_np[:, :, 2]

    # classname = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
    #             'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table',
    #             'tent','toilet','tv_stand','vase','wardrobe','xbox']
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
            # ax.scatter(x[i], y[i], z[i], c=colors[color[i]])
            ax.scatter(x[i], y[i], z[i], s=40,alpha=1,c="orangered")
            ax.scatter(xp[i],yp[i],zp[i], alpha=0.1, c="silver")

        
            # 优化坐标系
            # ax.set_xlim(-1.2,1.2)
            # ax.set_ylim(-1.2,1.2)
            # ax.set_zlim(-1.2,1.2)
            # ax.set_xticks(np.arange(-1.2, 1.2, 0.4))
            # ax.set_yticks(np.arange(-1.2, 1.2, 0.4))
            # ax.set_zticks(np.arange(-1.2, 1.2, 0.4))
            
            ax.set_zlabel('Z')  # 坐标轴
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.axis('off')   # 去掉坐标轴显示

            plt.draw()
            plt.savefig(save_name,dpi=600)
            # plt.show()

def save_point(points,save_name_prefix,save_dir,target):
    """
    将batch中的点保存为ply,输入的points为BxNx3
    """
    points = points.reshape(points.shape[0],-1,3)
    classname = ['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair','cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand','person','piano','plant','radio','range_hood','sink','sofa','stairs','stool','table',
                'tent','toilet','tv_stand','vase','wardrobe','xbox']
    for in_batch_id in range(points.shape[0]):

        filename = save_name_prefix + '-{}-{}({}).ply'.format(in_batch_id, target[in_batch_id], classname[target[in_batch_id]])
        filename = save_dir.joinpath(filename)
        ply.write_ply(filename, points[in_batch_id], ['x','y','z'])

def noise_Gaussian(sigma, points): # 点云加高斯噪声
    noise = np.random.normal(loc=0, scale=sigma, size=points.shape)   # loc均值 scale标准差
    out = points + noise
    return out


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    
    num_out_point = int(args.assign_ratio * args.num_point) 

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log_eval/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('diffpool')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)

    print("logs saving in ", exp_dir)
    # checkpoints_dir = exp_dir.joinpath('checkpoints/')
    # checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    vis_dir = exp_dir.joinpath(args.test_data)
    save_dir = exp_dir.joinpath('sampled_point_ply/')
    log_dir.mkdir(exist_ok=True)
    vis_dir.mkdir(exist_ok=True)
    save_dir.mkdir(exist_ok=True)

    '''LOG'''
    # args = parse_args()
    logger = logging.getLogger("Model_Eval")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.sampler_model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    log_string("GPU ID: %s" % gpu_id)

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # data_path = 'data/modelnet40_normal_resampled/'
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if args.dataset == "modelnet40":
        data_path = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")
        test_dataset = ModelNetDataSet(root=data_path, num_point=args.num_point, split=args.test_data,not_debug=args.not_debug)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
        
    elif args.dataset == "scanobjectnn":
        data_path = os.path.join(DATA_DIR, "scanobjectnn","h5_files","main_split")
        test_dataset = ScanobjectNNDataSet(root=data_path, num_point=args.num_point, split=args.test_data,dataclass=args.dataclass)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.dataset == "semantickitti_cls":
        data_path = os.path.join(DATA_DIR, "instance_data","SemanticKITTI_cls_v2")
        test_dataset = SemanticKITTI_cls_DataSet(root=data_path, split='test', keep_reflect=args.keep_reflect, not_debug=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)


    SHAPE_NAMES = [
        line.rstrip()
        for line in open(
            os.path.join(data_path, "shape_names.txt")
        )
    ]
    """MODEL LOADING"""
    num_class = args.num_category
    # cls = importlib.import_module(args.classifier_model)

    model = create_model(args)
    # classifier_loss = cls.get_loss()

    shutil.copy('./models/%s.py' % args.classifier_model, str(exp_dir))
    shutil.copy('models/pointnet_utils.py', str(exp_dir))
    shutil.copy('./models/%s.py' % args.sampler_model, str(exp_dir))
    shutil.copy('./eval_diffpool_gnnsample.py', str(exp_dir))    

    model = model.cuda()

    

    """testing"""
    with torch.no_grad():
        logger.info('Starting testing......')
        # mean_correct = []
        # train_proj_loss_sum = 0
        total_seen = 0
        total_correct = 0
        t_getgraph_sum = 0
        t_sample_sum = 0
        t_clas_sum = 0
        t_epoch_start = time()
        t_match_sum = 0

        class_acc = np.zeros((num_class, 3))
        retrieval_vectors = []
        out_data_label = []
        # loss_sum = 0
        # simp_loss_sum = 0
        # task_loss_sum = 0
        count = 0
        complete_nums = []
        
        for batch_id, (points, target) in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
                
            '''graph construction'''
            # adj = get_knn_graph_feature(x=points,k=20)
            t_before_getgraph = time()
            adj = get_radius_graph(x=points,r=0.2)
            t_after_getgraph = time()

            adj = adj.cuda()
            if args.batch_size == 1:
                target = target.view(1)   
            else:
                target = np.squeeze(target) # batchsize=1时用squeeze有bug

            # points=[bnc] 训sampler时不需要增强
            points, target = points.cuda(), target.cuda()   # input bnc

            # debug, tensor-numpy
            points_np = points.cpu().detach().numpy()   # input -1~1
            # print(points)
            # print(points.dtype)
            # print(points_np)
            # print(points_np.dtype)

            if args.points_noise != 0.0:
                points = torch.from_numpy(noise_Gaussian(args.points_noise,points_np).astype('float32')).cuda()
            # print(points)
            # print(points.dtype)

            t_before_sample = time()
            simp_pc = model.sampler(points,adj)  # input bnc,output b n c 
            t_after_sample = time()

            if args.record_overlap_point:    # 统计generate point 中有无重叠点，或统计重叠点数量(需要batchsize=1)
                temp = np.squeeze(simp_pc) # n c
                x = ["%fa%fb%fc" % (temp[i,0],temp[i,1],temp[i,2]) for i in range(len(temp))]
                overlap_num = int(num_out_point - len(np.unique(x)))
                if overlap_num != 0 :
                    log_string("overlap num: %d, in batch: %d, class: %d" % (overlap_num,batch_id,target))

            if args.match :  # add postprocess
                # input B 3 N (Bx3x32), output B N 3
                t_match_begin = time()
                sample_pc, complete_num = nn_match(x=points.transpose(1,2),y=simp_pc.transpose(1,2),num_out_points=num_out_point,complete=args.complete_method)  
                t_match_end = time()
                complete_nums.append(complete_num)
                save_name_prefix = 'match-{}'.format(batch_id)
            else:
                sample_pc = simp_pc   # bnc
                t_match_begin,t_match_end = 0,0
                save_name_prefix = 'nomatch-{}'.format(batch_id)
                

            # simp_pc, proj_pc = model.sampler(points)  # input bxcxn
            sample_pc = sample_pc.transpose(2,1).contiguous()    # need to be bcn

            # debug, tensor->numpy
            sample_pc_np = sample_pc.cpu().detach().numpy()   # 
            simple_pc_np = simp_pc.cpu().detach().numpy()   # input -0.484~0.47 不符合pointnet输入分布

            t_before_clas = time()

            if args.classifier_model == "pointnet_cls":
                pred, _ , retrieval_vector = model(sample_pc)    # input bxcxn
                pred_choice = pred.data.max(1)[1]

            elif args.classifier_model == "dgcnn_cls":
                logit = model(sample_pc)                   # dgcnn 最后直接输出 logit
                pred_choice = logit.max(1)[1]

            t_after_clas = time()


            if args.save_sampled_point:
                save_point(sample_pc_np,save_name_prefix,save_dir,pred_choice)

            if args.vis_sampled_point:
                draw(sample_pc_np[:, 0, :], sample_pc_np[:, 1, :], sample_pc_np[:, 2, :], save_name_prefix, vis_dir, points_np=points_np, classname=SHAPE_NAMES,color=pred_choice)

            if args.save_retrieval_vectors:
                retrieval_vectors.append(retrieval_vector.cpu().detach())
                out_data_label.append(target.cpu())

            # if not args.simplification_loss:
            #     simplification_loss = model.sampler.loss(adj=adj)
            # else:
            #     simplification_loss = model.sampler.get_simplification_loss(
            #                     points,simp_pc,num_out_point)                                             

            # 第一个batch中采样loss=0.1719
            # samplenet_loss = args.alpha * simplification_loss   # default alpha=1

            # 但分类loss=7598.9038
            # task_loss = classifier_loss(pred,target.long(),trans_feat)   
            # loss = task_loss + samplenet_loss
            # loss_sum += loss.item()
            # simp_loss_sum += simplification_loss.item()
            # task_loss_sum += task_loss.item()
            
            # compute acc
            # correct = pred_choice.eq(target.long().data).cpu().sum()    # 第一个batch中正确14个,第2个正确17个，第3个正确14个
            # mean_correct.append(correct.item() / float(points.size()[0]))
            total_seen += points.size()[0]
            total_correct += pred_choice.eq(target.long().data).cpu().sum()

            # compute time cost
            t_getgraph_sum += t_after_getgraph - t_before_getgraph
            t_sample_sum += t_after_sample - t_before_sample
            t_clas_sum += t_after_clas - t_before_clas
            t_match_sum += t_match_end - t_match_begin

            # compute macc
            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1

            count += 1

            
        t_epoch_end = time()

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        log_string(class_acc)
        for i, name in enumerate(SHAPE_NAMES):
            log_string("%10s:\t%0.3f,\ttotal num: %f" % (name, class_acc[i][2],class_acc[i][1]))  # batchsize=1时才是准确值
            
        class_acc = np.mean(class_acc[:, 2])

        t_iteration_avg = float(t_epoch_end - t_epoch_start)/count
        frame = count/float(t_epoch_end - t_epoch_start)
        t_getgraph_avg = float(t_getgraph_sum)/count
        t_sample_avg = float(t_sample_sum)/count
        t_clas_avg = float(t_clas_sum)/count
        t_match_avg = float(t_match_sum)/count

        if args.save_retrieval_vectors:
            data_dtype = 'float32'
            label_dtype = 'int64'   # dataloader里load_h5py将label从uint8转成了int64
            out_data_retrieval_one_file = np.vstack(retrieval_vectors)
            out_data_label_one_file = np.vstack(out_data_label)
            retrieval_path = os.path.join(exp_dir, 'retrieval/')
            if not os.path.exists(retrieval_path):
                os.makedirs(retrieval_path)
            data_prep_util.save_h5(os.path.join(retrieval_path,'retrieval_vectors_%d.h5'%(num_out_point)),
                                out_data_retrieval_one_file, out_data_label_one_file, data_dtype, label_dtype)


    # loss_avg = float(loss_sum)/count
    # simp_loss_avg = float(simp_loss_sum)/count
    # train_proj_loss_avg = float(train_proj_loss_sum)/train_count
    # task_loss_avg = float(task_loss_sum)/count


    # instance_acc = np.mean(mean_correct)
    # log_string('Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
    log_string('total_correct:%f, total_seen:%f' % (total_correct,total_seen))
    log_string('Acc :%f, Class Accuracy: %f' % ((total_correct / total_seen),class_acc))
    # log_string('loss avg: %f' % loss_avg)
    # log_string('simplification loss avg: %f' % simp_loss_avg)
    # log_string('Train projection loss avg: %f' % train_proj_loss_avg)
    # log_string('classification loss avg: %f' % task_loss_avg)
    
    log_string('frame =  %f' % frame)   # 每秒钟多少帧
    log_string('iteration time avg: %f' % t_iteration_avg)  # 一个iteration耗时，=1/frame
    log_string('getgraph time avg: %f' % t_getgraph_avg)    # 建图耗时
    log_string('sample time avg: %f' % t_sample_avg)    # 采样耗时
    log_string('matching time avg: %f' % t_match_avg)  # match耗时
    log_string('classification time avg: %f' % t_clas_avg)  # 下游分类耗时

    log_string('each sample need to complete = %s' % complete_nums) # 需要补全的数量
    log_string('End of Testing...')



if __name__ == '__main__':
    print("begin running eval_gsnet.py")
    args = parse_args()
    main(args)
