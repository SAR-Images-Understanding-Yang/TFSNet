import torch
import os
from core.data_loader import data_loader
import random
import numpy as np
import configargparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn import metrics
from core.models import FrequencySelectModel
from collections import OrderedDict
import torch.nn as nn


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise ValueError('Boolean value expected.')


def plot_tsne(features, labels, save_path):
    ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
    '''
    # t-SNE降维处理
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    features = tsne.fit_transform(features)
    # 归一化处理
    scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
    result = scaler.fit_transform(features)
    # 颜色设置
    color = ['#E0B48C', '#BEBEBE', '#000080', '#87CEEB', '#006400',
             '#00FF00', '#4682B4', '#D02090', '#8B7765', '#B03060']
    # 可视化展示
    plt.figure(figsize=(6, 6))
    plt.title('t-SNE')
    types = []
    for i in range(10):
        index = np.argwhere(labels==i).squeeze(1)
        types.append(plt.scatter(result[index, 0], result[index, 1], c=[color[yi] for yi in labels[index]], s=10))
    plt.legend(types, ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23'])
    plt.savefig(os.path.join(save_path, 'tsne.png'), dpi=600, format='png')
    plt.close('all')

def plot_confusion_matrix(preds, labels, save_path, title=None, thresh=0.8, axis_labels=None):
    plt.figure()
    # 利用sklearn中的函数生成混淆矩阵并归一化
    labels_name = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
    cm = metrics.confusion_matrix(labels, preds, labels=None, sample_weight=None)  # 生成混淆矩阵
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    # 画图，如果希望改变颜色风格，可以改变此部分的cmap=pl.get_cmap('Blues')处
    plt.imshow(cm, interpolation='nearest', cmap=plt.get_cmap('Blues'))

    plt.colorbar()  # 绘制图例
    # 图像标题
    if title is not None:
        plt.title(title)
    # 绘制坐标
    num_local = np.array(range(len(labels_name)))
    if axis_labels is None:
        axis_labels = labels_name
    plt.xticks(num_local, axis_labels, rotation=45)  # 将标签印在x轴坐标上， 并倾斜45度
    plt.yticks(num_local, axis_labels)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    # 将百分比打印在相应的格子内，大于thresh的用白字，小于的用黑字
    for i in range(np.shape(cm)[0]):
        for j in range(np.shape(cm)[1]):
            if int(cm[i][j] * 100 + 0.5) > 0:
                plt.text(j, i, format(int(cm[i][j] * 100 + 0.5), 'd') + '%',
                        ha="center", va="center",
                        color="white" if cm[i][j] > thresh else "black")  # 如果要更改颜色风格，需要同时更改此行
    plt.savefig(os.path.join(save_path, 'ConfusionMatrix.png'), dpi=600, format='png')
    plt.close('all')


def test(model, target_test_loader, args):
    model.eval()
    test_loss = AverageMeter()
    vectors, preds, labels = [], [], []
    correct = 0
    criterion = torch.nn.CrossEntropyLoss()
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        loop = tqdm(target_test_loader)
        for data, target, asc in loop:
            data, target, asc = data.to(args.device), target.to(args.device), asc.to(args.device)
            s_output = model.predict(data, asc)
            loss = criterion(s_output, target)
            test_loss.update(loss.item())
            pred = torch.max(s_output, 1)[1]
            correct += torch.sum(pred == target)

            preds.append(torch.max(s_output, 1)[1])
            vectors.append(s_output)
            labels.append(target)

            loop.set_description(f'Model Testing -> ')
            info_dict = {'test_loss': test_loss.avg, 'acc': 100. * correct / len_target_dataset}
            loop.set_postfix(info_dict)
    acc = 100. * correct / len_target_dataset
    # 特征向量
    vectors = torch.cat(vectors, dim=0).cpu().numpy()
    # 预测
    preds = torch.cat(preds, dim=0).cpu().numpy()
    # 标签
    labels = torch.cat(labels, dim=0).cpu().numpy()
    plot_tsne(vectors, labels, args.checkpoints)
    plot_confusion_matrix(preds, labels, args.checkpoints)
    return acc, test_loss.avg


def load_data(args):
    '''
    src_domain, tgt_domain data to load
    '''
    folder_src = os.path.join(args.data_dir, args.src_domain)
    folder_train_tgt = os.path.join(args.data_dir, args.tgt_domain, "train")
    folder_test_tgt = os.path.join(args.data_dir, args.tgt_domain, "test")
    source_loader, n_class = data_loader.load_data(
        folder_src, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_train_loader, _ = data_loader.load_data(
        folder_train_tgt, args.batch_size, infinite_data_loader=not args.epoch_based_training, train=True,
        num_workers=args.num_workers)
    target_test_loader, _ = data_loader.load_data(
        folder_test_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return source_loader, target_train_loader, target_test_loader, n_class


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_optimizer(model, args):
    initial_lr = args.lr if not args.lr_scheduler else 0.01
    params = model.get_parameters(initial_lr=initial_lr)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                nesterov=False)
    return optimizer


def write_log(file, logs):
    with open(os.path.join(file, "log.txt"), 'a+') as f:
        f.writelines(logs)
        f.close()


def frequency_select(source_loader, target_train_loader, args):
    info = "Frequency Component Selecting ----> "
    print(info)
    if os.path.exists(args.checkpoints):
        pass
    else:
        os.makedirs(args.checkpoints)
    write_log(args.checkpoints, [info, ])
    pretrain_epoch = 5
    model = FrequencySelectModel(base_net=args.backbone).to(args.device)
    optimizer = get_optimizer(model, args)
    len_source_loader = len(source_loader)
    len_target_loader = len(target_train_loader)
    iter_source, iter_target = iter(source_loader), iter(target_train_loader)
    n_batch = min(len_source_loader, len_target_loader)
    if n_batch == 0:
        n_batch = args.n_iter_per_epoch
    # 预训练 pretrain_epoch-1 轮
    for e in range(1, pretrain_epoch+1):
        model.train()
        train_loss = AverageMeter()
        if max(len_target_loader, len_source_loader) != 0:
            iter_source, iter_target = iter(source_loader), iter(target_train_loader)
        loop = tqdm(range(n_batch), total=n_batch)
        for iteration in loop:
            data_source, label_source, asc_source = next(iter_source)  # .next()
            data_target, _, asc_target = next(iter_target)  # .next()
            data_source, label_source, data_target = data_source.to(args.device), label_source.to(args.device), data_target.to(args.device)
            loss = model(data_source, data_target)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(model.parameters(), clip_value=10.0)
            optimizer.step()
            train_loss.update(loss.item())
            loop.set_description(f'Epoch [{e}/{pretrain_epoch}]')
            info_dct = {'lr': optimizer.state_dict()["param_groups"][1]["lr"],
                        'loss': train_loss.avg}
            loop.set_postfix(info_dct)
            # 频率选择
    frequency_statistics = []
    # 第pretrain_epoch轮 统计最好的频率分量
    model.eval()
    loop = tqdm(range(n_batch), total=n_batch)
    for iteration in loop:
        data_source, label_source, _ = next(iter_source)  # .next()
        data_source, label_source = data_source.to(args.device), label_source.to(args.device)
        frequency_statistics.append(model.predict(data_source, label_source).cpu().view(1))
        loop.set_description(f'Frequency Component Selecting ')
    # 统计部分代码
    # 使用np.unique统计每个唯一元素及其出现次数
    frequency_statistics = torch.cat(frequency_statistics).numpy()
    unique_elements, counts = np.unique(frequency_statistics, return_counts=True)
    # 找到出现次数最多的元素
    # np.argmax返回最大值的索引，这里用于找到出现次数最多的元素的索引
    most_common_index = np.argmax(counts)
    most_common_element = unique_elements[most_common_index]
    # most_common_count = counts[most_common_index]
    indx, indy = most_common_element // 10, most_common_element % 10
    # state_dict = model.state_dict()
    state_dict = OrderedDict({})
    state_dict['indx'] = torch.tensor(indx, device=args.device, dtype=torch.int64)
    state_dict['indy'] = torch.tensor(indy, device=args.device, dtype=torch.int64)
    info = "The best frequency component is the {:2d}".format(most_common_element)
    print(info)
    write_log(args.checkpoints, [info, '\n'])
    return state_dict