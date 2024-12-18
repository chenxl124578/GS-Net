import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import sys
import h5py
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, 'utils'))
# import provider


parser = argparse.ArgumentParser()
parser.add_argument('--dump_dir', default='log_eval/gsnet_pyg/log_dir/retrieval', help='dump folder path [default: dump]')
parser.add_argument('--num_out_points', type=int, default=32, help='Number of output points [2, 4, ..., 1024] [default: 32]')
parser.add_argument('--model_name', type=str, default="GSNet", help='model name')

FLAGS = parser.parse_args()
DUMP_DIR = FLAGS.dump_dir
NUM_OUT_POINTS = FLAGS.num_out_points

FILE_NAME = 'retrieval_vectors' + '_' + str(NUM_OUT_POINTS)
FILE_EXT = '.h5'
RETRIEVAL_DATA_PATH = os.path.join(DUMP_DIR, FILE_NAME + FILE_EXT)


def print_precision_recall_to_file(retrieval_vectors, labels):
    precision_file_path = os.path.join(DUMP_DIR, 'precision_' + FILE_NAME + '.txt')
    recall_file_path = os.path.join(DUMP_DIR, 'recall_' + FILE_NAME + '.txt')

    log_fout_percision = open(precision_file_path, 'w')
    log_fout_recall = open(recall_file_path, 'w')
    print_precision_recall_curve(retrieval_vectors, labels, log_fout_percision, log_fout_recall)
    log_fout_percision.close()
    log_fout_recall.close()

    return precision_file_path, recall_file_path


def print_precision_recall_curve(retrieval_vectors, labels, log_fout_percision, log_fout_recall):
    labels_unique, label_counts = np.unique(labels, return_counts=True)

    B, _ = retrieval_vectors.shape

    percision = np.zeros(100, dtype='float32')
    recall = np.zeros(100, dtype='float32')

    for b in range(B):
        curr_per, curr_rec = precision_recall(retrieval_vectors, labels, label_counts, b)
        percision += curr_per
        recall += curr_rec

    for i in range(100):
        log_string('%f' % (percision[i]/float(B)), log_fout_percision)
        log_string('%f' % (recall[i]/float(B)), log_fout_recall)


def precision_recall(vecs, labels, labels_counts, idx):
    dists = (vecs - vecs[idx]) ** 2
    dists = np.sum(dists, axis=1)
    rets = np.argsort(dists)
    matches = (labels[rets] == labels[idx])   # 越靠前越接近越容易True 例如(True,True,True,False,True,False....)
    matches_cum = np.cumsum(matches,  dtype='float32')
    total_class = labels_counts[labels[idx]]   # 该样本这一类别的样本数量

    precision = matches_cum / range(1, labels.shape[0]+1)   # 查准率：TP/TP+FP
    recall = matches_cum / total_class    # 查全率：TP/TP+FN

    relevant_idx = np.where(matches)[0]  # 只计算同类别
    precision_relevant = precision[relevant_idx]
    recall_relevant = recall[relevant_idx]

    precision_padded = np.pad(precision_relevant, (0, 100 - np.size(precision_relevant)), 'constant', constant_values=(precision_relevant[-1])) # 在test中40个类别每个类别最大样本数为100
    recall_padded = np.pad(recall_relevant, (0, 100 - np.size(recall_relevant)), 'constant', constant_values=(recall_relevant[-1]))

    return precision_padded, recall_padded


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


def log_string(out_str, LOG_FOUT):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)


def plot_precision_recall(recall, precision):
    plt.plot(recall, precision, c='g', linewidth=3, markersize=8)

    axis_val = np.arange(0, 5 + 1, 1) / 5.
    axis_str = [str(v) for v in axis_val]

    plt.xticks(axis_val, axis_str)
    plt.yticks(axis_val, axis_str)

    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)

    plt.grid(True)
    plt.tight_layout()

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

if __name__ == '__main__':
    # load retrieval data
    retrieval_vectors, labels = load_h5(RETRIEVAL_DATA_PATH)

    # calculate mean average precision
    res_macro = calc_macro_mean_average_precision(retrieval_vectors, labels)

    # print the result to log file
    log_fout_retrieval = open(os.path.join(DUMP_DIR, 'log_retrieval_' + FILE_NAME + '.txt'), 'w')
    log_string('macro_mean_average_precision result', log_fout_retrieval)
    log_string('-----------------------------------', log_fout_retrieval)
    log_string('mAP: %f' % res_macro, log_fout_retrieval)
    log_fout_retrieval.close()

    # compute precision recall data
    precision_file_path, recall_file_path = print_precision_recall_to_file(retrieval_vectors, labels)

    # plot precision recall curve
    precision = np.genfromtxt(precision_file_path, delimiter='\n')
    recall = np.genfromtxt(recall_file_path, delimiter='\n')

    fig = plt.figure(figsize=(8, 4.5))
    plot_precision_recall(recall, precision)
    plt.legend(['%s - %d points' % (FLAGS.model_name,NUM_OUT_POINTS)], loc='lower left', fontsize=12)
    fig.savefig(os.path.join(DUMP_DIR, 'precision_recall_curve.png'))
    plt.close()
