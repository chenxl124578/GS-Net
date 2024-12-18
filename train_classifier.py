"""
By Xiaolei Chen
Create in 2022/3/28

to train PointNet(or others model in future) classifier
"""

import os
import sys
import argparse
import datetime
from pathlib import Path
from tqdm import tqdm
import logging
import importlib
import shutil
import provider


import numpy as np
import torch
import torch.nn as nn
# import tensorboardX
# from tensorboardX import SummaryWriter

from data_utils.ModelNetDataLoader import ModelNetDataSet
from data_utils.ScanobjectNN_dataloader import ScanobjectNNDataSet
from data_utils.semanticKITTI_cls_loader import SemanticKITTI_cls_DataSet
from data_utils.dgcnn_util import cal_loss

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--dataset', type=str, default='semantickitti_cls', help='modelnet40 , scanobjectnn , semantickitti_cls')

    # scanobjectNN
    parser.add_argument('--dataclass', type=str, default='OBJ_BG', help='if scanobject: OBJ_BG, PB_T25, PB_T25_R, PB_T50_R, PB_T50_RS')

    # semantickitti_cls
    parser.add_argument('--keep_reflect', action='store_true', default=False, help='True or False to control the feature dimention of the pointclouds. True:4, False:3')

    parser.add_argument('--gpu', type=str, default='3', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--model', default='pointnet_cls', help='model name [default: pointnet_cls], dgcnn_cls or pointnet_cls')
    parser.add_argument('--num_category', default=6, type=int, choices=[40, 15, 6],  help='training on ModelNet40 , scanobjectNN 15 , semantickitti_cls 6')
    parser.add_argument('--epoch', default=250, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-3, help='adam weights decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--augmentation',type=str,default='old',help='new, old or none')

    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--decay_step', type=int, default=20, help='Decay step for lr decay [default: 200000]')
    parser.add_argument('--lr_decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
    
    parser.add_argument('--k', type=int ,default=1, help='when --model == dgcnn_cls, means num of nearest neighbors to use. Origin code is 20, but here default is 1')

    parser.add_argument('--change_init', action='store_true', default=False, help='change layer init')
    parser.add_argument('--save_retrieval_vectors',action='store_true',default=False,help='record the num of overlap point in generate points')

    return parser.parse_args()

def calc_macro_mean_average_precision(retrieval_vectors, labels):
    B, _ = retrieval_vectors.shape
    sum_avg_per = 0
    for b in range(B):   # 计算每个样本的precision？
        sum_avg_per += calc_average_precision(retrieval_vectors, labels, b)
    return sum_avg_per/float(B)


def calc_average_precision(vecs, labels, idx):
    dists = (vecs - vecs[idx]) ** 2
    dists = np.sum(dists, axis=1)
    rets = np.argsort(dists)    # 按照距离排序，返回的是index
    matches = (labels[rets] == labels[idx])  # 越靠前越接近越容易True 例如(True,True,True,False,True,False....)
    matches_cum = np.cumsum(matches, dtype='float32')       # 累计和
    precision = matches_cum / range(1, labels.shape[0] + 1) # 计算precision 例如[1,2,3,3,4,4,4,4,4] / [1,2,3,4,5,6,7,8,9]

    relevant_idx = np.where(matches)[0]   # 对比同label的 例如[0,1,2,4]
    precision_relevant = precision[relevant_idx]  # 例如[1,1,1,4/5]
    return np.sum(precision_relevant)/float(np.size(precision_relevant)) # (3+4/5)/4


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def test(model, loader, args, num_class=40, ):
    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    classifier = model.eval()

    retrieval_vectors = []
    out_data_label = []
    out_data_retrieval_one_file = None
    out_data_label_one_file = None
    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        origin_target = target
        target = np.squeeze(target)
        points = points.transpose(2, 1)

        points, target = points.cuda(), target.cuda()
        if args.model == "pointnet_cls":
            pred, _ , retrieval_vector = classifier(points)    # pointnet 最后有用 log_softmax
            pred_choice = pred.data.max(1)[1]

        elif args.model == "dgcnn_cls":
            logit = classifier(points)                   # dgcnn 最后直接输出 logit
            pred_choice = logit.max(1)[1]

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        if args.save_retrieval_vectors:
            retrieval_vectors.append(retrieval_vector.cpu().detach())  # [B,256]
            out_data_label.append(origin_target)   # [B]

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]  # bs=1时才有参考价值，计算的是每一类的准确率
    class_acc = np.mean(class_acc[:, 2])
    instance_acc = np.mean(mean_correct)
    if args.save_retrieval_vectors:
        out_data_retrieval_one_file = np.vstack(retrieval_vectors)
        out_data_label_one_file = np.vstack(out_data_label)

    return instance_acc, class_acc, out_data_retrieval_one_file, out_data_label_one_file


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification_'+args.model)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    # args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    log_string("begin running Train_Diffpool_gnnsample.py, PID: %d" % (os.getpid()))


    """download modelNet40----cxl"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(BASE_DIR)

    # Download dataset for point cloud classification
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")) and args.dataset == "modelnet40":
        www = "https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip"
        zipfile = os.path.basename(www)
        os.system("wget %s --no-check-certificate; unzip %s" % (www, zipfile))
        os.system("mv %s %s" % (zipfile[:-4], DATA_DIR))
        os.system("rm %s" % (zipfile))

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # data_path = 'data/modelnet40_normal_resampled/'

    if args.dataset == "modelnet40":
        data_path = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")
        train_dataset = ModelNetDataSet(root=data_path, num_point=args.num_point, split='train',not_debug=True)
        test_dataset = ModelNetDataSet(root=data_path, num_point=args.num_point, split='test',not_debug=True)
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    elif args.dataset == "scanobjectnn":
        data_path = os.path.join(DATA_DIR, "scanobjectnn","h5_files","main_split")
        train_dataset = ScanobjectNNDataSet(root=data_path, num_point=args.num_point, split='train',dataclass=args.dataclass)
        test_dataset = ScanobjectNNDataSet(root=data_path, num_point=args.num_point, split='test',dataclass=args.dataclass)
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)
    elif args.dataset == "semantickitti_cls":
        data_path = os.path.join(DATA_DIR, "SemanticKITTI_cls_v2")
        train_dataset = SemanticKITTI_cls_DataSet(root=data_path, split='train', keep_reflect=args.keep_reflect, not_debug=True)
        test_dataset = SemanticKITTI_cls_DataSet(root=data_path, split='test', keep_reflect=args.keep_reflect, not_debug=True)
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

    '''MODEL LOADING'''
    num_class = args.num_category
    model = importlib.import_module(args.model)
    if args.model == "pointnet_cls":
        shutil.copy('./models/%s.py' % args.model, str(exp_dir))
        shutil.copy('models/pointnet_utils.py', str(exp_dir))
        shutil.copy('./train_classifier.py', str(exp_dir))
        classifier = model.get_model(num_class, normal_channel=args.use_normals, change_init=args.change_init)
        criterion = model.get_loss()
        classifier.apply(inplace_relu)
        classifier = classifier.cuda()
        criterion = criterion.cuda()
    elif args.model == "dgcnn_cls":
        shutil.copy('data_utils/dgcnn_util.py', str(exp_dir))
        shutil.copy('models/dgcnn_cls.py', str(exp_dir))
        shutil.copy('train_classifier.py', str(exp_dir))
        classifier = model.DGCNN(args.k, emb_dims=1024, output_channels=num_class)
        criterion = cal_loss
        classifier = classifier.cuda()
    else: 
        raise AssertionError
    
    try:
        checkpoint = torch.load(str(exp_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    if args.optimizer == 'Adam':
        print("use Adam")
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer == 'sgd':
        print("use SGD")
        optimizer = torch.optim.SGD(classifier.parameters(), lr=0.001, momentum=0.9)
    else:
        raise ValueError

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.lr_decay_rate)
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0

    '''TRANING'''
    logger.info('Start training...')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        classifier = classifier.train()

        scheduler.step()
        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            """data augmentation 和 pointnet(pytorch)一致
                放弃了原版pointnet中的rotation 和 jitter 和 shuffle_data
                SampleNet中使用的rotation, jitter和shuffle_data
            """
            if args.augmentation == 'new':
                points = provider.random_point_dropout(points)          
                points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
                points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            elif args.augmentation == 'old':
                # points, target, _ = provider.shuffle_data(points,target)
                points = provider.rotate_point_cloud(points)
                points = provider.jitter_point_cloud(points)
            elif args.augmentation == 'none':
                pass
            else:
                raise ValueError

            points = torch.Tensor(points)
            points = points.transpose(2, 1)
            target = np.squeeze(target)
            points, target = points.cuda(), target.cuda()
            
            if args.model == "pointnet_cls":
                pred, trans_feat, _ = classifier(points)    # pointnet 最后有用 log_softmax
                loss = criterion(pred, target.long(), trans_feat) 
                pred_choice = pred.data.max(1)[1]

            elif args.model == "dgcnn_cls":
                logit = classifier(points)                   # dgcnn 最后直接输出 logit
                loss = criterion(logit, target.long())
                pred_choice = logit.max(1)[1]

            correct = pred_choice.eq(target.long().data).cpu().sum()
            mean_correct.append(correct.item() / float(points.size()[0]))
            loss.backward()
            optimizer.step()
            global_step += 1

        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)

        """evaling"""
        with torch.no_grad():
            instance_acc, class_acc, out_data_retrieval_one_file, out_data_label_one_file = test(classifier.eval(), testDataLoader, args, num_class=num_class)
            if (args.save_retrieval_vectors and (epoch+1)% 1 == 0):  # 每个epoch保存一次 retrieval并计算mAP值
                data_dtype = 'float32'
                label_dtype = 'int64'   # dataloader里load_h5py将label从uint8转成了int64
                retrieval_path = os.path.join(exp_dir, 'retrieval/')
                if not os.path.exists(retrieval_path):
                    os.makedirs(retrieval_path)
                res_macro = calc_macro_mean_average_precision(out_data_retrieval_one_file, out_data_label_one_file)
                retrieval_save_path = str(retrieval_path) + 'retrieval_%d_%f.pth' % (epoch+1,res_macro)
                if res_macro <= 0.688876:
                    logger.info('Save retrieval model...')
                    log_string('Saving at %s' % retrieval_save_path)
                    state = {
                        'epoch': epoch+1,
                        'instance_acc': instance_acc,
                        'class_acc': class_acc,
                        'model_state_dict': classifier.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, retrieval_save_path)


            if (instance_acc >= best_instance_acc):
                best_instance_acc = instance_acc
                best_epoch = epoch + 1

            if (class_acc >= best_class_acc):
                best_class_acc = class_acc
            log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
            log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))

            if (instance_acc >= best_instance_acc):
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': best_epoch,
                    'instance_acc': instance_acc,
                    'class_acc': class_acc,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

            # if (instance_acc >= 0.8105 and instance_acc <= 0.8114):  # for scanobjectNN OA=81.1
            #     logger.info('Save special model...')
            #     savepath = str(checkpoints_dir) + '/acc81_1_model.pth'
            #     log_string('Saving at %s' % savepath)
            #     state = {
            #         'epoch': best_epoch,
            #         'instance_acc': instance_acc,
            #         'class_acc': class_acc,
            #         'model_state_dict': classifier.state_dict(),
            #         'optimizer_state_dict': optimizer.state_dict(),
            #     }
            #     torch.save(state, savepath)
            global_epoch += 1
        


    logger.info('End of training...')
if __name__ == '__main__':  

    args = parse_args()
    main(args)