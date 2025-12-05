#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :TFSNet.py
# @Time      :2024/7/23 16:48
# @Author    :Yangxinpeng
# @Introduce :

'''
这个文件包含了TFSNet,以及消融实验使用到的模型
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct
from core.loss_funcs.transfer_losses import TransferLoss
from core.models.backbones import get_backbone


class TFSNet(nn.Module):
    '''
    该类别计算了单个频率分量，利用一个bottle neck和分类器进行分类
    还有时域的分类器，理论上时域的这个GAP分类器和00频率分量的性能是差不多的。
    '''
    def __init__(self,
                 num_class,
                 base_net='resnet50',
                 transfer_loss="mmd",
                 use_bottleneck=True,
                 bottleneck_width=256,
                 max_iter=1000,
                 margin=1.0,
                 **kwargs):
        super(TFSNet, self).__init__()
        # 这里设置必须的属性，包括时域、频域、散射域的伪标签过滤阈值
        for k, v in kwargs.items():
            self.__setattr__(k, v)
        for attr in ['threshold_t', 'threshold_f', 'threshold_s']:
            if hasattr(self, attr):
                self.threshold_t = 0.95

        self.num_class = num_class
        self.base_network = get_backbone(base_net)
        self.use_bottleneck = use_bottleneck
        self.transfer_loss = transfer_loss
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()

        self.classifier_layer = nn.Linear(feature_dim, num_class)
        transfer_loss_args = {
            "loss_type": self.transfer_loss,
            "max_iter": max_iter,
            "num_class": num_class
        }
        self.adapt_loss = TransferLoss(**transfer_loss_args)
        self.criterion = torch.nn.CrossEntropyLoss()

        self.scattering_domain_base_network = get_backbone('asc')
        if self.use_bottleneck:
            bottleneck_list = [
                nn.Linear(self.scattering_domain_base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.scattering_domain_bottleneck_layer = nn.Sequential(*bottleneck_list)
            bottleneck_list = [
                nn.Linear(self.base_network.output_num(), bottleneck_width),
                nn.ReLU()
            ]
            self.time_domain_bottleneck_layer = nn.Sequential(*bottleneck_list)
            feature_dim = bottleneck_width
        else:
            feature_dim = self.base_network.output_num()
        self.scattering_domain_classifier_layer = nn.Linear(feature_dim, num_class)
        self.time_domain_classifier_layer = nn.Linear(feature_dim, num_class)
        self.margin = margin
        self.recoder = 1
        self.register_buffer('indx', torch.tensor(-1, device='cuda', dtype=torch.int64))
        self.register_buffer('indy', torch.tensor(-1, device='cuda', dtype=torch.int64))

    def forward(self, source, target, source_label, source_asc, target_asc):
        source_features, source_time = self.base_network(source)
        target_features, target_time = self.base_network(target)
        _, source_scattering = self.scattering_domain_base_network(source_asc)
        _, target_scattering = self.scattering_domain_base_network(target_asc)
        # # 频域
        source = dct.dct_2d(source_features[-1])[:, :, self.indx, self.indy]
        target = dct.dct_2d(target_features[-1])[:, :, self.indx, self.indy]

        if self.use_bottleneck:
            source = self.bottleneck_layer(source)
            target = self.bottleneck_layer(target)
            source_time = self.time_domain_bottleneck_layer(source_time)
            target_time = self.time_domain_bottleneck_layer(target_time)
            source_scattering = self.scattering_domain_bottleneck_layer(source_scattering)
            target_scattering = self.scattering_domain_bottleneck_layer(target_scattering)

        source_clf = self.classifier_layer(source)
        source_time_clf = self.time_domain_classifier_layer(source_time)
        source_scattering_clf = self.scattering_domain_classifier_layer(source_scattering)
        clf_loss = self.criterion(source_clf, source_label)
        clf_loss += self.criterion(source_time_clf, source_label)
        clf_loss += self.criterion(source_scattering_clf, source_label)

        # transfer
        kwargs = {}
        kwargs['source_label'] = source_label
        target_clf = self.classifier_layer(target)
        kwargs['target_logits'] = torch.nn.functional.softmax(target_clf, dim=1)
        transfer_loss = self.adapt_loss(source, target, **kwargs)

        target_time_clf = self.time_domain_classifier_layer(target_time)
        kwargs['target_logits'] = torch.nn.functional.softmax(target_time_clf, dim=1)
        transfer_loss += self.adapt_loss(source_time, target_time, **kwargs)

        target_scattering_clf = self.scattering_domain_classifier_layer(target_scattering)
        kwargs['target_logits'] = torch.nn.functional.softmax(target_scattering_clf, dim=1)
        transfer_loss += self.adapt_loss(source_scattering, target_scattering, **kwargs)

        contrastive_loss = self.contrastive_learning(source, source_label)
        contrastive_loss += self.contrastive_learning(source_time, source_label)
        contrastive_loss += self.contrastive_learning(source_scattering, source_label)

        pseudo_label, mask = self.pseudo_label_processing(target_clf, self.threshold_f)
        pseudo_label_time, mask_time = self.pseudo_label_processing(target_time_clf, self.threshold_t)
        pseudo_label_scattering, mask_scattering = self.pseudo_label_processing(target_scattering_clf, self.threshold_s)

        index = (pseudo_label == pseudo_label_time) * (pseudo_label == pseudo_label_scattering)
        mask = mask * mask_time * mask_scattering * index
        print("Pseudo Label Count: {:.4f}".format(torch.sum(mask, dim=0) / source.size(0)))
        clf_loss += (F.nll_loss(F.log_softmax(target_clf, dim=1), pseudo_label.cuda(),
                                reduction='none') * mask).mean()
        clf_loss += (F.nll_loss(F.log_softmax(target_time_clf, dim=1), pseudo_label.cuda(),
                                reduction='none') * mask).mean()
        clf_loss += (F.nll_loss(F.log_softmax(target_scattering_clf, dim=1), pseudo_label.cuda(),
                                reduction='none') * mask).mean()
        contrastive_loss += self.contrastive_learning(target, pseudo_label, mask)
        contrastive_loss += self.contrastive_learning(target_time, pseudo_label, mask)
        contrastive_loss += self.contrastive_learning(target_scattering, pseudo_label, mask)
        self.recoder += 1
        return clf_loss, transfer_loss, contrastive_loss

    def predict(self, x, asc):
        features, output_time = self.base_network(x)
        _, output_scattering = self.scattering_domain_base_network(asc)
        # 频域
        output = dct.dct_2d(features[-1])[:, :, self.indx, self.indy]
        if self.use_bottleneck:
            output = self.bottleneck_layer(output)
            output_time = self.time_domain_bottleneck_layer(output_time)
            output_scattering = self.scattering_domain_bottleneck_layer(output_scattering)
        clf = self.classifier_layer(output)
        clf_time = self.time_domain_classifier_layer(output_time)
        clf_scattering = self.scattering_domain_classifier_layer(output_scattering)
        clf = clf + clf_time + clf_scattering
        # [b, w, h, class]
        return clf

    def get_parameters(self, initial_lr=1.0):
        # 频域 主干 + neck+classifier
        params = [
            {'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},
            {'params': self.classifier_layer.parameters(), 'lr': 1.0 * initial_lr},
        ]
        if self.use_bottleneck:
            params.append(
                {'params': self.bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr}
            )
        # Loss-dependent
        if self.transfer_loss == "adv":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
        elif self.transfer_loss == "daan":
            params.append(
                {'params': self.adapt_loss.loss_func.domain_classifier.parameters(), 'lr': 1.0 * initial_lr}
            )
            params.append(
                {'params': self.adapt_loss.loss_func.local_classifiers.parameters(), 'lr': 1.0 * initial_lr}
            )
        # 散射域 主干
        params.append({'params': self.scattering_domain_base_network.parameters(), 'lr': 1.0 * initial_lr})
        # 时域 散射域 neck
        params.append({'params': self.time_domain_bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr})
        params.append({'params': self.scattering_domain_bottleneck_layer.parameters(), 'lr': 1.0 * initial_lr})
        # 时域 散射域 classifier
        params.append({'params': self.time_domain_classifier_layer.parameters(), 'lr': 1.0 * initial_lr})
        params.append({'params': self.scattering_domain_classifier_layer.parameters(), 'lr': 1.0 * initial_lr})
        return params

    def pseudo_label_processing(self, pseudo_pred, threshold=0.95):
        prob = torch.nn.functional.softmax(pseudo_pred, dim=1)
        max_probs, index = torch.max(prob, -1)
        mask = max_probs.ge(threshold).float()
        pseudo = pseudo_pred.data.max(1)[1]
        return pseudo, mask

    def contrastive_learning(self, representation, labels, mask=None):
        # 打乱顺序
        representation_shuffle = torch.cat([representation[1:], representation[0].unsqueeze(0)], dim=0)
        labels_shuffle = torch.cat([labels[1:], labels[0].unsqueeze(0)], dim=0)
        label = (labels != labels_shuffle).float()
        distance = F.pairwise_distance(representation, representation_shuffle)
        if mask is None:
            loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
        else:
            loss_contrastive = torch.mean((1 - label) * torch.pow(distance, 2) * mask +
                                          label * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2) * mask)
        return loss_contrastive

    def epoch_based_processing(self, *args, **kwargs):
        if self.transfer_loss == "daan":
            self.adapt_loss.loss_func.update_dynamic_factor(*args, **kwargs)
        else:
            pass

