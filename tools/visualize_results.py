#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :visualize_results.py
# @Time      :2024/6/11 20:16
# @Author    :Yangxinpeng
# @Introduce :
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn import metrics
import configargparse
from core.models import *
import os
import torch
from utils import str2bool
from core.data_loader import data_loader
from tqdm import tqdm


# model related
def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Transfer learning config parser",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    # general configuration
    parser.add("--config", is_config_file=True, help="config file path")
    parser.add_argument('--num_workers', type=int, default=0)

    # network related
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--use_bottleneck', type=str2bool, default=True)

    # data loading related
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--tgt_domain', type=str, required=True)

    # training related
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_epoch', type=int, default=100)
    parser.add_argument("--n_iter_per_epoch", type=int, default=20, help="Used in Iteration-based training")

    # optimizer related
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)

    # learning rate scheduler related
    parser.add_argument('--lr_gamma', type=float, default=0.0003)
    parser.add_argument('--lr_decay', type=float, default=0.75)
    parser.add_argument('--lr_scheduler', type=str2bool, default=True)

    # transfer related
    parser.add_argument('--transfer_loss_weight', type=float, default=10)
    parser.add_argument('--transfer_loss', type=str, default='mmd')

    parser.add_argument('--checkpoints', type=str, help='path for model weight')
    return parser


def make_test_data_loader(args):
    '''
        make dataloader
    '''
    folder_tgt = os.path.join(args.data_dir, args.tgt_domain)
    if os.path.exists(os.path.join(folder_tgt, "train")) and os.path.exists(os.path.join(folder_tgt, "test")):
        folder_test_tgt = os.path.join(folder_tgt, "test")
    else:
        folder_test_tgt = folder_tgt
    target_test_loader, n_classes = data_loader.load_data(
        folder_test_tgt, args.batch_size, infinite_data_loader=False, train=False, num_workers=args.num_workers)
    return target_test_loader, n_classes


def test(model, target_test_loader, args):
    model.eval()
    vectors, preds, labels = [], [], []
    correct = 0
    len_target_dataset = len(target_test_loader.dataset)
    with torch.no_grad():
        loop = tqdm(target_test_loader)
        for data, target, asc in loop:
            data = data.to(args.device)
            asc = asc.to(args.device)
            s_output = model.predict(data, asc)
            correct += torch.sum(torch.max(s_output, 1)[1].cpu() == target)
            preds.append(torch.max(s_output, 1)[1])
            vectors.append(s_output)
            labels.append(target)
            loop.set_description(f'Model Testing -> ')
            info_dict = {'acc': 100. * correct / len_target_dataset}
            loop.set_postfix(info_dict)
    # 特征向量
    vectors = torch.cat(vectors, dim=0).cpu().numpy()
    # 预测
    preds = torch.cat(preds, dim=0).cpu().numpy()
    # 标签
    labels = torch.cat(labels, dim=0).numpy()
    return vectors, preds, labels


# 可视化部分
class Visualize(object):
    '''
    Visualize features by TSNE
    '''
    def __init__(self, features, preds, labels, save_path):
        '''
        features: (m,n)
        labels: (m,)
        '''
        self.features = features
        self.preds = preds
        self.labels = labels
        self.save_path = save_path
        if os.path.isfile(os.path.join(save_path, 'log.txt')):
            self.log = os.path.join(save_path, 'log.txt')
            # load loss and accuracy
            self.loss = []
            self.acc = []
            with open(self.log, 'r') as f:
                lines = f.readlines()
                for line in lines:
                    if 'Epoch' in line:
                        for element in line.split(','):
                            name, data = element.split(':')
                            if name.replace(' ', '') == 'total_loss':
                                self.loss.append(float(data))
                            elif name.replace(' ', '') == 'test_acc':
                                self.acc.append(float(data))
        else:
            self.log = None

    def plot_tsne(self, save_eps=False):
        ''' Plot TSNE figure. Set save_eps=True if you want to save a .eps file.
        '''
        # t-SNE降维处理
        tsne = TSNE(n_components=2, init='pca', random_state=0)
        features = tsne.fit_transform(self.features)
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
            index = np.argwhere(self.labels==i).squeeze(1)
            types.append(plt.scatter(result[index, 0], result[index, 1], c=[color[yi] for yi in self.labels[index]], s=10))
        plt.legend(types, ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23'], framealpha=1)

        if save_eps:
            plt.savefig(os.path.join(self.save_path, 'tsne.png'), dpi=600, format='png')
            plt.savefig(os.path.join(self.save_path, 'tsne.eps'), dpi=600, format='eps')
        plt.show()

    def plot_confusion_matrix(self, title=None, thresh=0.8, axis_labels=None, save_eps=False):
        # 利用sklearn中的函数生成混淆矩阵并归一化
        labels_name = ['2s1', 'bmp2', 'btr70', 'm1', 'm2', 'm35', 'm60', 'm548', 't72', 'zsu23']
        cm = metrics.confusion_matrix(self.labels, self.preds, labels=None, sample_weight=None)  # 生成混淆矩阵
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
        if save_eps:
            plt.savefig(os.path.join(self.save_path, 'ConfusionMatrix.png'), dpi=600, format='png')
            plt.savefig(os.path.join(self.save_path, 'ConfusionMatrix.eps'), dpi=600, format='eps')

        plt.show()

    def plot_loss_curve(self, save_eps=False):
        # 绘制图表
        if self.log is None:
            return
        else:
            # plt.figure(figsize=(10, 6))  # 设置图表大小
            # 绘制端到端(X)的折线图
            plt.plot(range(len(self.loss)), self.loss, label='Loss', marker='o')
            # 设置X轴和Y轴的标签
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # 设置图表的标题
            plt.title('Loss Curve')
            # 添加图例
            plt.legend(framealpha=1)

            # 显示图表
            plt.grid(True)  # 添加网格线
            if save_eps:
                plt.savefig(os.path.join(self.save_path, 'LossCurve.png'), dpi=600, format='png')
                plt.savefig(os.path.join(self.save_path, 'LossCurve.eps'), dpi=600, format='eps')

            plt.show()

    def plot_accuracy_curve(self, save_eps=False):
        # 绘制图表
        if self.log is None:
            return
        else:
            # plt.figure(figsize=(10, 6))  # 设置图表大小
            # 绘制端到端(X)的折线图
            plt.plot(range(len(self.acc)), self.acc, label='Accuracy', marker='s')
            # 设置X轴和Y轴的标签
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            # 设置图表的标题
            plt.title('Accuracy Curve')
            # 添加图例
            plt.legend(framealpha=1)
            # 显示图表
            plt.grid(True)  # 添加网格线
            if save_eps:
                plt.savefig(os.path.join(self.save_path, 'AccuracyCurve.png'), dpi=600, format='png')
                plt.savefig(os.path.join(self.save_path, 'AccuracyCurve.eps'), dpi=600, format='eps')

            plt.show()


def main():
    parser = get_parser()
    args = parser.parse_args()
    setattr(args, "device", torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(args)
    target_test_loader, n_class = make_test_data_loader(args)
    setattr(args, "n_class", n_class)
    model = TFSNet(args.n_class, base_net=args.backbone, use_bottleneck=args.use_bottleneck).to(args.device)
    checkpoints = os.path.join(args.checkpoints, 'best_model.pth')
    model.load_state_dict(torch.load(checkpoints))
    features, preds, labels = test(model, target_test_loader, args)
    vis = Visualize(features, preds, labels, args.checkpoints)
    vis.plot_tsne(save_eps=True)
    vis.plot_confusion_matrix(save_eps=True)
    vis.plot_loss_curve(save_eps=True)
    vis.plot_accuracy_curve(save_eps=True)


if __name__ == "__main__":
    main()
