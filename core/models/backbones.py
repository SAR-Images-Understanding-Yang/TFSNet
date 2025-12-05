import torch
import torch.nn as nn
from torchvision import models
from scipy.spatial.distance import cdist
import torch.nn.functional as F
import torch_dct as dct


resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}

def get_backbone(name):
    if "resnet" in name.lower():
        return ResNetBackbone(name)
    elif "alexnet" == name.lower():
        return AlexNetBackbone()
    elif "dann" == name.lower():
        return DaNNBackbone()
    elif "asc" in name.lower():
        return ASCBackBone()


class DaNNBackbone(nn.Module):
    def __init__(self, n_input=224*224*3, n_hidden=256):
        super(DaNNBackbone, self).__init__()
        self.layer_input = nn.Linear(n_input, n_hidden)
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self._feature_dim = n_hidden

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        return x

    def output_num(self):
        return self._feature_dim
    
# convnet without the last layer
class AlexNetBackbone(nn.Module):
    def __init__(self):
        super(AlexNetBackbone, self).__init__()
        model_alexnet = models.alexnet(pretrained=True)
        self.features = model_alexnet.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_alexnet.classifier[i])
        self._feature_dim = model_alexnet.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256*6*6)
        x = self.classifier(x)
        return x

    def output_num(self):
        return self._feature_dim

class ResNetBackbone(nn.Module):
    def __init__(self, network_type):
        super(ResNetBackbone, self).__init__()
        resnet = resnet_dict[network_type](pretrained=True)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        self._feature_dim = resnet.fc.in_features
        del resnet

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x1 = self.maxpool(x)
        x2 = self.layer1(x1)
        x3 = self.layer2(x2)
        x4 = self.layer3(x3)
        x5 = self.layer4(x4)
        output = self.avgpool(x5)
        output = output.view(output.size(0), -1)
        return (x1, x2, x3, x4, x5), output
    
    def output_num(self):
        return self._feature_dim


class ResLinearModule(nn.Module):
    def __init__(self,
                 in_c,
                 out_c,
                 bn=True,
                 relu=True):
        super(ResLinearModule, self).__init__()
        self.linear = nn.Linear(in_c, out_c)
        self.bn = bn
        self.relu = relu
        if self.bn:
            self.bn_layer = nn.BatchNorm1d(out_c)
        if self.relu:
            self.relu_layer = nn.ReLU()

    def forward(self, x):
        feature = self.linear(x)
        feature = self.bn_layer(feature.permute(0, 2, 1)).permute(0, 2, 1)
        feature = self.relu_layer(feature) + x
        return feature


class ASCBackBone(nn.Module):
    def __init__(self):
        super(ASCBackBone, self).__init__()
        self.linear = nn.Linear(7, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.layer1 = nn.Sequential(ResLinearModule(64, 64),
                                    nn.Linear(64, 128))
        self.layer2 = nn.Sequential(ResLinearModule(128, 128),
                                    nn.Linear(128, 256))
        self.layer3 = nn.Sequential(ResLinearModule(256, 256),
                                    nn.Linear(256, 512))
        self.layer4 = nn.Sequential(ResLinearModule(512, 512),
                                    nn.Linear(512, 1024))
        self.layer5 = nn.Sequential(ResLinearModule(1024, 1024),
                                    nn.Linear(1024, 2048))

    def forward(self, x):
        feature = self.linear(x)
        feature = self.bn(feature.permute(0, 2, 1)).permute(0, 2, 1)
        feature = self.relu(feature)
        feature1 = self.layer1(feature)
        feature2 = self.layer2(feature1)
        feature3 = self.layer3(feature2)
        feature4 = self.layer4(feature3)
        feature5 = self.layer5(feature4)
        output = torch.mean(feature5, 1)
        return (feature1, feature2, feature3, feature4, feature5), output

    def output_num(self):
        return 2048


class FrequencySelectModel(nn.Module):
    '''
    该类别计算了单个频率分量，利用一个bottle neck和分类器进行分类
    还有时域的分类器，理论上时域的这个GAP分类器和00频率分量的性能是差不多的。
    '''
    def __init__(self,
                 base_net='resnet50',
                 **kwargs):
        super(FrequencySelectModel, self).__init__()
        self.base_network = get_backbone(base_net)
        self.conv1 = nn.Sequential(nn.Conv2d(2048, 1024, kernel_size=1),
                                   nn.BatchNorm2d(1024),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1),
                                   nn.BatchNorm2d(512),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1),
                                   nn.BatchNorm2d(256),
                                   nn.ReLU())
        self.recons3 = nn.Conv2d(256, 1, kernel_size=1)
        self.recons_loss = nn.L1Loss()

    def forward(self, source, target):
        source_features, source_ = self.base_network(source)
        feature1 = self.conv1(source_features[-1])
        feature2 = self.conv2(source_features[-2] + F.interpolate(feature1, scale_factor=2, mode='bilinear', align_corners=False))
        feature3 = self.conv3(source_features[-3] + F.interpolate(feature2, scale_factor=2, mode='bilinear', align_corners=False))
        recons3 = F.interpolate(self.recons3(feature3), scale_factor=8, mode='bilinear', align_corners=False)
        reconstruction_loss = self.recons_loss(recons3.repeat(1, 3, 1, 1), source)

        target_features, target_ = self.base_network(target)
        feature1 = self.conv1(target_features[-1])
        feature2 = self.conv2(
            target_features[-2] + F.interpolate(feature1, scale_factor=2, mode='bilinear', align_corners=False))
        feature3 = self.conv3(
            target_features[-3] + F.interpolate(feature2, scale_factor=2, mode='bilinear', align_corners=False))
        recons3 = F.interpolate(self.recons3(feature3), scale_factor=8, mode='bilinear', align_corners=False)
        reconstruction_loss += self.recons_loss(recons3.repeat(1, 3, 1, 1), target)
        return reconstruction_loss

    def get_parameters(self, initial_lr=1.0):
        params = [{'params': self.base_network.parameters(), 'lr': 0.1 * initial_lr},]
        params.append({'params': self.conv1.parameters(), 'lr': 1.0 * initial_lr})
        params.append({'params': self.conv2.parameters(), 'lr': 1.0 * initial_lr})
        params.append({'params': self.conv3.parameters(), 'lr': 1.0 * initial_lr})
        params.append({'params': self.recons3.parameters(), 'lr': 1.0 * initial_lr})
        return params

    def frequency_select(self, target_dct_feature, label):
        b, c, w, h = target_dct_feature.shape
        with torch.no_grad():
            target = target_dct_feature.permute(0, 2, 3, 1)
            classes_distance = torch.zeros((w, h))
            for i in range(w):
                j = i
                pred = label
                target_vector = (target[:, i, j, :]).cpu().detach().numpy()
                dist_matrix = cdist(target_vector, target_vector, 'cosine')
                dist_matrix = torch.from_numpy(dist_matrix).to('cuda')
                class_dists = torch.zeros((b, b), dtype=torch.int64, device='cuda')
                indices = torch.triu_indices(b, b, offset=1)
                class_dists[indices[0], indices[1]] = torch.where(pred[indices[0]] == pred[indices[1]],
                                                                  torch.tensor(-1, device='cuda'),
                                                                  torch.tensor(1, device='cuda'))
                classes_distance[i, j] = torch.mean(dist_matrix * class_dists)
            value, ind = torch.max(classes_distance.reshape((-1,)), 0)
            # classes_distance = classes_distance.cpu().detach().numpy()
            indx, indy = torch.div(ind, w, rounding_mode='trunc'), ind % h
            return indx * 10 + indy

    def predict(self, x, label):
        source_features, _ = self.base_network(x)
        # 频率选择
        source_dct = dct.dct_2d(source_features[-1])
        results = self.frequency_select(source_dct, label)
        # print(results)
        return results