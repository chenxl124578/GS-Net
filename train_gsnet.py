'''
By Xiaolei Chen
Train GS-Net
'''
import os
from time import time
gpu_id="2"
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
import numpy as np
import torch

from models.gsnet_pyg import GSNet

from data_utils.ModelNetDataLoader import ModelNetDataSet
from data_utils.graph_construction import *
from data_utils.ScanobjectNN_dataloader import ScanobjectNNDataSet
from data_utils.semanticKITTI_cls_loader import SemanticKITTI_cls_DataSet

from data_utils.dgcnn_util import cal_loss

torch.manual_seed(0)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
from models import pointnet_cls


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    # parser.add_argument('--gpu', type=str, default='2', help='specify gpu device')
    parser.add_argument('--dataset', type=str, default='modelnet40', help='modelnet40 , scanobjectnn, semantickitti_cls')
    
    # scanobjectnn
    parser.add_argument('--dataclass', type=str, default='OBJ_BG', help='OBJ_BG, PB_T25, PB_T25_R, PB_T50_R, PB_T50_RS')
    
    # semantickitti_cls
    parser.add_argument('--keep_reflect', action='store_true', default=False, help='True or False to control the feature dimention of the pointclouds. True:4, False:3')

    parser.add_argument('--batch_size', type=int, default=32, help='batch size in training')
    parser.add_argument('--classifier_model', default='pointnet_cls', help='model name [default: pointnet_cls] or dgcnn_cls')
    parser.add_argument('--classifier_model_path', default='weights/acc88_45_PointNet_classifier_model_modelnet40.pth', help='Path to model.ckpt file of a pre-trained classifier')
    parser.add_argument('--sampler_model', default='gsnet_pyg', help='Sampler python files name')

    parser.add_argument('--num_category', default=40, type=int, choices=[40, 15, 6],  help='training on ModelNet40, scanobjectNN 15 , semantickitti_cls 6')
    parser.add_argument('--epoch', default=500, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='adam', help='optimizer for training[adam or sgd]')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='adam weights decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
    parser.add_argument('--decay_step', type=int, default=60, help='Decay step for lr decay [default: 60]')
    parser.add_argument('--lr_decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--train_tasknet', action='store_true', default=False, help='train pointnet or dgcnn')
    parser.add_argument('--train_gsnet', type=bool, default=True)
    
    # debug true or false
    parser.add_argument('--not_debug', action='store_true', default=False, help='if debug set False and just use 128 sample, if ready to train set True')
    # eval save best model which evaluate on matched point 
    parser.add_argument('--val_match',action='store_true', default=False, help='if true, feed matched point to PointNet in eval and save best match model')
    # diffpool sampler arguments
    parser.add_argument('--hidden_dim', type=int, default=128, help='hidden_dim [default: 128]')
    parser.add_argument('--output_dim', type=int, default=128, help=' [default: 128]')
    parser.add_argument('--num_gc_layers', type=int, default=3, help='[default: 3]')
    # assign_ratio: 0.5=1/2, 0.25=1/4, 0.125=1/8, 0.0625=1/16, 0.03125=1/32, 0.015625=1/64, 0.0078125=1/128
    parser.add_argument('--assign_ratio', type=float, default=0.03125, help=' 0.03125: input 1024 output 32 point; 0.5: 1024->512->256...->32')
    parser.add_argument('--linkpred',  action='store_true', default=False, help='[default: False]')
    parser.add_argument('--simplification_loss',  action='store_true', default=True, help='[default: True]')
    
    parser.add_argument('--alpha', type=float, default=20, help='Simplification regularization loss weight [default: 20]')

    # graphing
    parser.add_argument('--graphing', type=str,default="ball", help='ball or knn')
    parser.add_argument('--radius', type=float,default=0.2, help='radius if graphing=ball')
    parser.add_argument('--k', type=int,default=5, help='k if graphing=knn')
    parser.add_argument('--dgcnn_k', type=int ,default=1, help='when --model == dgcnn_cls, num of nearest neighbors to use in DGCNN')

    return parser.parse_args()

def test(model, loader, classifier_loss, logger, num_class=40, val_match=False):
    def log_string(str):
        logger.info(str)
        print(str)

    mean_correct = []
    class_acc = np.zeros((num_class, 3))
    loss_sum = 0
    count = 0
    total_seen = 0
    total_correct = 0
    match_total_correct = 0
    simp_loss_sum = 0
    proj_loss_sum = 0
    task_loss_sum = 0
    link_loss_sum = 0

    t_graph_sum = 0
    t_sample_sum = 0
    t_clas_sum = 0
    t_test_epoch_start = time()

    for j, (points, target) in tqdm(enumerate(loader), total=len(loader)):
        target = np.squeeze(target)
        # points = points.transpose(2, 1)

        t_before_graph = time()
        if args.graphing == "ball":
            adj = get_radius_graph(x=points,r=args.radius)
        if args.graphing == "knn":
            adj = get_knn_graph(x=points, k=args.k)

        t_after_graph = time()

        adj = adj.cuda()

        points, target = points.cuda(), target.cuda()
 
        t_before_sample = time()
        simp_pc = model.sampler(points,adj)  # input bxnxc,output bnc
        t_after_sample = time()

        simp_pc = simp_pc.transpose(2,1)

        t_before_clas = time()
        if args.classifier_model == "pointnet_cls":
            pred, trans_feat, _ = model(simp_pc)    # input bxcxn
            task_loss = classifier_loss(pred,target.long(),trans_feat) 
            pred_choice = pred.data.max(1)[1]

        elif args.classifier_model == "dgcnn_cls":
            logit = model(simp_pc)                   # dgcnn 最后直接输出 logit
            task_loss = classifier_loss(logit, target.long())
            pred_choice = logit.max(1)[1]
        t_after_clas = time()

        simp_pc = simp_pc.transpose(2,1)  # transpose to bnc for simplification loss
        linkprediction_loss = model.sampler.loss(adj=adj)
            
        
        simplification_loss = model.sampler.get_simplification_loss(
                        points,simp_pc,args.assign_ratio * args.num_point)      
        
        samplenet_loss = args.alpha * simplification_loss   

        loss = task_loss + samplenet_loss + linkprediction_loss

        loss_sum += loss.item()
        simp_loss_sum += simplification_loss.item()
        task_loss_sum += task_loss.item()
        link_loss_sum += linkprediction_loss.item()

        count += 1

        for cat in np.unique(target.cpu()):
            classacc = pred_choice[target == cat].eq(target[target == cat].long().data).cpu().sum()
            class_acc[cat, 0] += classacc.item() / float(points[target == cat].size()[0])
            class_acc[cat, 1] += 1

        correct = pred_choice.eq(target.long().data).cpu().sum()
        mean_correct.append(correct.item() / float(points.size()[0]))

        total_seen += points.size()[0]
        total_correct += pred_choice.eq(target.long().data).cpu().sum()

        #compute time cost
        t_graph_sum += t_after_graph - t_before_graph
        t_sample_sum += t_after_sample - t_before_sample
        t_clas_sum += t_after_clas - t_before_clas

        # matched point acc
        if val_match:
            match_pc, _ = nn_match(x=points.transpose(1,2),y=simp_pc.transpose(1,2),num_out_points=simp_pc.shape[1]) # output BNC
            match_pc = match_pc.transpose(2,1).contiguous() # output B 3 N
            if args.classifier_model == "pointnet_cls":
                match_pred, match_trans_feat, _ = model(match_pc)    # input bxcxn
                match_pred_choice = match_pred.data.max(1)[1]

            elif args.classifier_model == "dgcnn_cls":
                match_logit = model(match_pc)                   # dgcnn 最后直接输出 logit
                match_pred_choice = match_logit.max(1)[1]
            match_total_correct += match_pred_choice.eq(target.long().data).cpu().sum()

    t_test_epoch_end = time()
    test_t_test_iteration_avg = float(t_test_epoch_end - t_test_epoch_start)/count
    test_t_graph_avg = float(t_graph_sum)/count
    test_t_sample_avg = float(t_sample_sum)/count
    test_t_clas_avg = float(t_clas_sum)/count

    loss_avg = float(loss_sum)/count
    simp_loss_avg = float(simp_loss_sum)/count
    # proj_loss_avg = float(proj_loss_sum)/count
    task_loss_avg = float(task_loss_sum)/count
    link_loss_avg = float(link_loss_sum)/count


    log_string('Test iteration time avg: %f' % test_t_test_iteration_avg)
    log_string('Test graphing time avg: %f' % test_t_graph_avg)
    log_string('Test sample time avg: %f' % test_t_sample_avg)
    log_string('Test classification time avg: %f' % test_t_clas_avg)
    log_string('total_correct:%f, match_total_correct:%f,total_seen:%f' % (total_correct,match_total_correct,total_seen))

    class_acc[:, 2] = class_acc[:, 0] / class_acc[:, 1]
    class_acc = np.mean(class_acc[:, 2])

    return (total_correct / total_seen), class_acc, loss_avg, simp_loss_avg, task_loss_avg, link_loss_avg, (match_total_correct/total_seen)

def create_model(args):
    cls = importlib.import_module(args.classifier_model)
    if args.classifier_model == "pointnet_cls":
        classifier = cls.get_model(args.num_category, normal_channel=args.use_normals)
    elif args.classifier_model == "dgcnn_cls":
        classifier = cls.DGCNN(args.dgcnn_k, emb_dims=1024, output_channels=args.num_category)
    else: 
        raise AssertionError
    
    if args.train_tasknet:
        classifier.requires_grad_(True)
        classifier.train()
    else:
        classifier.requires_grad_(False)
        classifier.eval()

    # Create sampling network
    sampler = GSNet(max_num_nodes=args.num_point,input_dim=3,output_dim=3,hidden_dim=args.hidden_dim,
                                embedding_dim=args.output_dim,assign_ratio=args.assign_ratio)

    if args.train_gsnet:
        sampler.requires_grad_(True)
        sampler.train()
    else:
        sampler.requires_grad_(False)
        sampler.eval()
    if args.classifier_model_path is not None:
        classifier.load_state_dict(torch.load(args.classifier_model_path)['model_state_dict'])
        print('Use pretrained model from %s' % args.classifier_model_path)

    classifier.sampler = sampler
    return classifier


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)
    '''HYPER PARAMETER'''
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath(args.sampler_model)
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    for i in range(5):
        if os.path.exists(exp_dir):
            exp_dir = exp_dir.joinpath("v%d"%i)
        else:
            break
    print("log in path:", exp_dir)

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
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.sampler_model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    '''HYPER PARAMETER'''
    log_string("using GPU ID: %s" % gpu_id)
    log_string("begin running Train_Diffpool_gnnsample.py, PID: %d" % (os.getpid()))

    '''DATA LOADING'''
    log_string('Load dataset ...')
    # data_path = 'data/modelnet40_normal_resampled/'
    DATA_DIR = os.path.join(BASE_DIR, "data")
    if args.dataset == "modelnet40":
        data_path = os.path.join(DATA_DIR, "modelnet40_ply_hdf5_2048")
        train_dataset = ModelNetDataSet(root=data_path, num_point=args.num_point, split='train',not_debug=args.not_debug)
        test_dataset = ModelNetDataSet(root=data_path, num_point=args.num_point, split='test',not_debug=args.not_debug)
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.dataset == "scanobjectnn":
        data_path = os.path.join(DATA_DIR, "scanobjectnn","h5_files","main_split")
        train_dataset = ScanobjectNNDataSet(root=data_path, num_point=args.num_point, split='train',dataclass=args.dataclass)
        test_dataset = ScanobjectNNDataSet(root=data_path, num_point=args.num_point, split='test',dataclass=args.dataclass)
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    elif args.dataset == "semantickitti_cls":
        data_path = os.path.join(DATA_DIR, "SemanticKITTI_cls_v2")
        train_dataset = SemanticKITTI_cls_DataSet(root=data_path, split='train', keep_reflect=args.keep_reflect, not_debug=True)
        test_dataset = SemanticKITTI_cls_DataSet(root=data_path, split='test', keep_reflect=args.keep_reflect, not_debug=True)
        trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=10, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)



    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    # train_dataset = ModelNet(data_path, '40', True, transform, pre_transform)
    # test_dataset = ModelNet(data_path, '40', False, transform, pre_transform)
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
    #                           num_workers=6)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
    #                          num_workers=6)

    """MODEL LOADING"""
    num_class = args.num_category
    cls = importlib.import_module(args.classifier_model)
    model = create_model(args)
    if args.classifier_model == "pointnet_cls":
        classifier_loss = cls.get_loss()
    if args.classifier_model == "dgcnn_cls":
        classifier_loss = cal_loss

    shutil.copy('./models/%s.py' % args.classifier_model, str(exp_dir))
    shutil.copy('models/pointnet_utils.py', str(exp_dir))
    shutil.copy('data_utils/dgcnn_util.py', str(exp_dir))
    shutil.copy('./models/%s.py' % args.sampler_model, str(exp_dir))
    shutil.copy('./train_diffpool_gnnsample.py', str(exp_dir))    

    start_epoch = 0

    model = model.cuda()
    log_string('classifier training:%s' % model.training)
    log_string('sampler training:%s' % model.sampler.training)

    """optimizer"""
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())

    log_string("Now set optimizer")
    if args.optimizer=="adam":
        optimizer = torch.optim.Adam(
            learnable_params,      # just train samplenet
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    elif args.optimizer=="sgd":
        optimizer = torch.optim.SGD(learnable_params,lr=args.learning_rate,momentum=args.momentum)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.lr_decay_rate)



    """training"""
    global_epoch = 0
    global_step = 0
    best_instance_acc = 0.0
    best_class_acc = 0.0
    match_best_instance_acc = 0.0

    logger.info('Starting training......')
    for epoch in range(start_epoch, args.epoch):
        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        mean_correct = []
        train_loss_sum = 0
        train_simp_loss_sum = 0
        # train_proj_loss_sum = 0
        train_task_loss_sum = 0
        train_link_loss_sum = 0
        train_count = 0
        t_getgraph_sum = 0
        t_sample_sum = 0
        t_clas_sum = 0
        t_train_epoch_start = time()

        for batch_id, (points, target) in tqdm(enumerate(trainDataLoader, 0), total=len(trainDataLoader), smoothing=0.9):
            # if batch_id == 2:    # debug
            #     break
            # points = points.transpose(2, 1)  # input sampler [b,c,n]
            '''graph construction'''
            # adj = get_knn_graph_feature(x=points,k=20)
            t_before_getgraph = time()
            
            if args.graphing == "ball":
                adj = get_radius_graph(x=points,r=args.radius)
            if args.graphing == "knn":
                adj = get_knn_graph(x=points, k=args.k)
            adj = adj.cuda()

            t_after_getgraph = time()


            target = np.squeeze(target)
            
            # points=[bnc] 训sampler时不需要增强
            points, target = points.cuda(), target.cuda()   # input bnc

            # debug, tensor-numpy
            points_np = points.cpu().detach().numpy()   # input -1~1
            
            t_before_sample = time()
            simp_pc = model.sampler(points,adj)  # input bnc,output b n c , input -1~1
            t_after_sample = time()

            # simp_pc, proj_pc = model.sampler(points)  # input bxcxn
            simp_pc = simp_pc.transpose(2,1)  # transpose to bcn

            # debug, tensor->numpy
            simp_pc_np = simp_pc.cpu().detach().numpy()   # input -0.484~0.47 不符合pointnet输入分布

            t_before_clas = time()
            if args.classifier_model == "pointnet_cls":
                pred, trans_feat, _ = model(simp_pc)    # input bxcxn
                task_loss = classifier_loss(pred,target.long(),trans_feat) 
                pred_choice = pred.data.max(1)[1]

            elif args.classifier_model == "dgcnn_cls":
                logit = model(simp_pc)                   # dgcnn 最后直接输出 logit
                task_loss = classifier_loss(logit, target.long())
                pred_choice = logit.max(1)[1]
            t_after_clas = time()

            simp_pc = simp_pc.transpose(2,1)  # transpose to bnc for simplification loss
            linkprediction_loss = model.sampler.loss(adj=adj)
           
            simplification_loss = model.sampler.get_simplification_loss(
                            points,simp_pc,args.assign_ratio * args.num_point)     
            # 第一个batch中采样loss=0.1719
            samplenet_loss = args.alpha * simplification_loss   # default alpha=1

            # 但分类loss=7598.9038
            loss = task_loss + samplenet_loss + linkprediction_loss
            train_loss_sum += loss.item()
            train_simp_loss_sum += simplification_loss.item()
            train_task_loss_sum += task_loss.item()
            train_link_loss_sum += linkprediction_loss.item()

            # compute acc
            correct = pred_choice.eq(target.long().data).cpu().sum()    # 第一个batch中正确14个,第2个正确17个，第3个正确14个
            mean_correct.append(correct.item() / float(points.size()[0]))

            #compute time cost
            t_getgraph_sum += t_after_getgraph - t_before_getgraph
            t_sample_sum += t_after_sample - t_before_sample
            t_clas_sum += t_after_clas - t_before_clas


            # Backward + Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1
            train_count += 1
        
        t_train_epoch_end = time()
        train_t_train_iteration_avg = float(t_train_epoch_end - t_train_epoch_start)/train_count
        train_t_getgraph_avg = float(t_getgraph_sum)/train_count
        train_t_sample_avg = float(t_sample_sum)/train_count
        train_t_clas_avg = float(t_clas_sum)/train_count

        train_loss_avg = float(train_loss_sum)/train_count
        train_simp_loss_avg = float(train_simp_loss_sum)/train_count
        # train_proj_loss_avg = float(train_proj_loss_sum)/train_count
        train_task_loss_avg = float(train_task_loss_sum)/train_count
        train_link_loss_avg = float(train_link_loss_sum)/train_count

        scheduler.step()


        train_instance_acc = np.mean(mean_correct)
        log_string('Train Instance Accuracy: %f' % train_instance_acc)
        log_string('Train loss avg: %f' % train_loss_avg)
        log_string('Train simplification loss avg: %f' % train_simp_loss_avg)
        # log_string('Train projection loss avg: %f' % train_proj_loss_avg)
        log_string('Train classification loss avg: %f' % train_task_loss_avg)
        log_string('Train linkprediction loss avg: %f' % train_link_loss_avg)

        log_string('Train iteration time avg: %f' % train_t_train_iteration_avg)
        log_string('Train getgraph time avg: %f' % train_t_getgraph_avg)
        log_string('Train sample time avg: %f' % train_t_sample_avg)
        log_string('Train classification time avg: %f' % train_t_clas_avg)


        """evaling"""
        task_state = model.training
        if model.sampler is not None:
            sampler_state = model.sampler.training

        model.eval()
        with torch.no_grad():
            instance_acc, class_acc, loss_avg, simp_loss_avg, task_loss_avg,link_loss_avg,match_instance_acc = test(model, testDataLoader, classifier_loss, logger, num_class=num_class,val_match=args.val_match)
        if args.val_match and (match_instance_acc >= match_best_instance_acc):
            match_best_instance_acc = match_instance_acc
            match_best_epoch = epoch + 1

        if (instance_acc >= best_instance_acc):
            best_instance_acc = instance_acc
            best_epoch = epoch + 1

        if (class_acc >= best_class_acc):
            best_class_acc = class_acc
        log_string('Test Instance Accuracy: %f, Class Accuracy: %f' % (instance_acc, class_acc))
        log_string('Best Instance Accuracy: %f, Class Accuracy: %f' % (best_instance_acc, best_class_acc))
        log_string('Test loss avg: %f ' % (loss_avg))
        log_string('Test simplification loss avg: %f' % simp_loss_avg)
        # log_string('Test projection loss avg: %f' % proj_loss_avg)
        log_string('Test classification loss avg: %f' % task_loss_avg)
        log_string('Test linkprediction loss avg: %f' % link_loss_avg)

        if (instance_acc >= best_instance_acc):   # 保存 generate point 最高准确率的 model
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': best_epoch,
                'instance_acc': instance_acc,
                'class_acc': class_acc,
                'model_state_dict': model.sampler.state_dict(),  # 我只保存sampler的weights
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
        
        if args.val_match and (match_instance_acc >= match_best_instance_acc):
            logger.info('Save match model...')
            savepath = str(checkpoints_dir) + '/match_best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': match_best_epoch,
                'instance_acc': match_instance_acc,
                'model_state_dict': model.sampler.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        global_epoch += 1

        model.train(task_state)
        if model.sampler is not None:
            model.sampler.train(sampler_state)
        
    logger.info('End of training...')

if __name__ == '__main__':
    args = parse_args()
    main(args)

    # debug用
    # path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
    #                 'data/ModelNet40')
    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(1024)
    # train_dataset = ModelNet(path, '40', True, transform, pre_transform)
    # test_dataset = ModelNet(path, '40', False, transform, pre_transform)
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True,
    #                           num_workers=6)
    # test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False,
    #                          num_workers=6)