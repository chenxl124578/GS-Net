import os
import random
random.seed(0)
from time import time
import sys


gpu_id = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id    # GPU setting
from tqdm import tqdm
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from data_utils import ply
import logging
import importlib

import datetime
import numpy as np
import torch

from data_utils.ModelNetDataLoader import ModelNetDataSet
from data_utils.ScanobjectNN_dataloader import ScanobjectNNDataSet
from data_utils.semanticKITTI_cls_loader import SemanticKITTI_cls_DataSet

from data_utils import data_prep_util
torch.manual_seed(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('evaluating')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='modelnet40 , scanobjectnn, semantickitti_cls')
    # scanobjectNN
    parser.add_argument('--dataclass', type=str, default='OBJ_BG', help='OBJ_BG, PB_T25, PB_T25_R, PB_T50_R, PB_T50_RS')
    # semantickitti_cls
    parser.add_argument('--keep_reflect', action='store_true', default=False, help='True or False to control the feature dimention of the pointclouds. True:4, False:3')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size in training')
    parser.add_argument('--classifier_model', default='pointnet_cls', help='model name [default: pointnet_cls], or dgcnn_cls')
    parser.add_argument('--classifier_model_path', default='weights/acc90_42_PointNet_classifier_model_modelnet40.pth', help='or weights/acc88_45_PointNet_classifier_model_modelnet40.pth')
    parser.add_argument('--sampler_model', default='none', help='Sampler model name: none')
    parser.add_argument('--num_category', default=40, type=int, choices=[10, 40, 15, 6],  help='training on ModelNet10/40, , scanobjectNN 15 , semantickitti_cls 6')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')

    parser.add_argument('--test_data', type=str, default='test',help='test or train dataset')
    parser.add_argument('--save_sampled_point', action='store_true', default=False, help='if True, save sampled point to log_eval')
    parser.add_argument('--vis_sampled_point', action='store_true', default=False, help='if True, save sampled point to log_eval')

    # debug true or false
    parser.add_argument('--not_debug', action='store_true', default=False, help='if debug set False and just use 128 sample, if ready to train set True')
    parser.add_argument('--save_retrieval_vectors',action='store_true',default=False,help='record the num of overlap point in generate points')

    parser.add_argument('--k', type=int ,default=1, help='when --model == dgcnn_cls, num of nearest neighbors to use')

    return parser.parse_args()


def draw(x, y, z, name, file_dir,points_np, classname, color=None):
    """
    绘制单个样本的三维点图
    """
    xp,yp,zp=points_np[:,:, 0],points_np[:, :, 1],points_np[:, :, 2]

    if color is None:
        for i in range(len(x)):
            ax = plt.subplot(projection='3d') 
            save_name = name+'-{}.png'.format(i)
            save_name = file_dir.joinpath(save_name)
            ax.scatter(x[i], y[i], z[i], c='r')
            ax.set_zlabel('Z')  
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.draw()
            plt.savefig(save_name)
            # plt.show()
    else:
        colors = ['red', 'blue', 'green', 'yellow', 'orange', 'tan', 'orangered', 'lightgreen', 'coral', 'aqua', 'gold', 'plum', 'khaki', 'cyan', 'crimson', 'lawngreen', 'thistle', 'skyblue', 'lightblue', 'moccasin', 'pink', 'lightpink', 'fuchsia', 'chocolate', 'tomato', 'orchid', 'grey', 'plum', 'peru', 'purple', 'teal', 'sienna', 'turquoise', 'violet', 'wheat', 'yellowgreen', 'deeppink', 'azure', 'ivory', 'brown']
        for i in range(len(x)):
            ax = plt.subplot(projection='3d')  
            save_name = name + '-{}-{}({}).png'.format(i, color[i],classname[color[i]])
            save_name = file_dir.joinpath(save_name)
            ax.scatter(x[i], y[i], z[i], s=40,alpha=1,c="black")   
            
            ax.scatter(xp[i],yp[i],zp[i], alpha=0.1, c="silver")
            
            ax.set_zlabel('Z')  
            ax.set_ylabel('Y')
            ax.set_xlabel('X')
            plt.axis('off')
            plt.grid(b=None)
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


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    num_out_point = int(args.num_point) 

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log_eval/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.classifier_model)
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('eval_cls')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
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
        data_path = os.path.join(DATA_DIR, "SemanticKITTI_cls_v2")
        test_dataset = SemanticKITTI_cls_DataSet(root=data_path, split=args.test_data, keep_reflect=args.keep_reflect, not_debug=args.not_debug)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    SHAPE_NAMES = [
        line.rstrip()
        for line in open(
            os.path.join(data_path, "shape_names.txt")
        )
    ]
    """MODEL LOADING"""
    num_class = args.num_category
    model = importlib.import_module(args.classifier_model)
    if args.classifier_model == "pointnet_cls":
        classifier = model.get_model(num_class, normal_channel=args.use_normals, change_init=False)
        classifier = classifier.cuda()
    elif args.classifier_model == "dgcnn_cls":
        classifier = model.DGCNN(args.k, emb_dims=1024, output_channels=num_class)
        classifier = classifier.cuda()
    else: 
        raise AssertionError

    classifier.requires_grad_(False)
    classifier.eval()
    if args.classifier_model_path is not None:
        classifier.load_state_dict(torch.load(args.classifier_model_path)['model_state_dict'])
        print('Use classifier model from %s' % args.classifier_model_path)
    else:
        raise ValueError
    
    """testing"""
    with torch.no_grad():

        logger.info('Starting testing......')
        total_seen = 0
        total_correct = 0
        class_acc = np.zeros((num_class, 3))
        retrieval_vectors = []
        out_data_label = []
        count = 0
        t_sample_sum = 0
        t_test_epoch_start = time()
        for batch_id, (points, target) in tqdm(enumerate(testDataLoader, 0), total=len(testDataLoader), smoothing=0.9):
            if args.batch_size == 1:
                target = target.view(1)
            else:
                target = np.squeeze(target)

            points, target = points.cuda(), target.cuda()
            points_np = points.cpu().detach().numpy()   # input -1~1

            # points:[b,n,3] 
            t_before_sample = time()
            
            samp_pc = points
            
            t_after_sample = time()
            
            samp_pc = samp_pc.transpose(1,2).to(torch.float32) # transpose to b3n
            samp_pc = samp_pc.cuda()
            sample_pc_np = samp_pc.cpu().detach().numpy()   # 

            if args.classifier_model == "pointnet_cls":
                pred, _ , retrieval_vector = classifier(samp_pc)    # input bxcxn
                pred_choice = pred.data.max(1)[1]

            elif args.classifier_model == "dgcnn_cls":
                logit = classifier(samp_pc)                   # dgcnn 最后直接输出 logit
                pred_choice = logit.max(1)[1]


            save_name_prefix = '{}-{}'.format(args.sampler_model,batch_id)

            if args.save_sampled_point:
                save_point(sample_pc_np,save_name_prefix,save_dir,pred_choice)

            if args.vis_sampled_point:
                draw(sample_pc_np[:, 0, :], sample_pc_np[:, 1, :], sample_pc_np[:, 2, :], save_name_prefix, vis_dir, points_np,classname=SHAPE_NAMES,color=pred_choice)

            if args.save_retrieval_vectors:
                retrieval_vectors.append(retrieval_vector.cpu().detach())
                out_data_label.append(target.cpu())

            count += 1

            t_sample_sum += t_after_sample - t_before_sample


            for cat in np.unique(target.cpu()):
                classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
                class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
                class_acc[cat, 1] += 1
            
            # correct = pred_choice.eq(target.long().data).cpu().sum()
            # mean_correct.append(correct.item() / float(points.size()[0]))

            total_seen += points.size()[0]
            total_correct += pred_choice.eq(target.long().data).cpu().sum()


        # log_string('Test iteration time avg: %f' % test_t_test_iteration_avg)
        # log_string('Test graphing time avg: %f' % test_t_graph_avg)
        # log_string('Test sample time avg: %f' % test_t_sample_avg)
        # log_string('Test classification time avg: %f' % test_t_clas_avg)
        t_test_epoch_end = time()
        time_avg = float(t_test_epoch_end - t_test_epoch_start)/count
        frame = count/float(t_test_epoch_end - t_test_epoch_start)
        t_sample_avg = float(t_sample_sum)/count

        class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
        log_string(class_acc)
        for i, name in enumerate(SHAPE_NAMES):
            log_string("%10s:\t%0.3f,\ttotal num: %f" % (name, class_acc[i][2],class_acc[i][1]))  # batchsize=1时才是准确值
        
        class_acc = np.mean(class_acc[:, 2])

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



        # instance_acc = np.mean(mean_correct)
        log_string('total_correct:%f, total_seen:%f' % (total_correct,total_seen))
        log_string('Acc :%f, Class Accuracy: %f' % ((total_correct / total_seen),class_acc))
        log_string('frame =  %f' % frame)
        log_string('iteration time avg: %f' % time_avg)
        log_string('sample time avg: %f' % t_sample_avg)
        
        log_string('End of Testing...')



if __name__ == '__main__':
    print("begin running eval_classifier.py")
    args = parse_args()
    main(args)