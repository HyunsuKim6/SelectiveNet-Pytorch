"""
Created on Thu Aug 8, 2019
@author: HyunsuKim6(Github), hyunsukim@kaist.ac.kr
"""

import torch.nn as nn
import torch.nn.functional as F

class SelectiveNet(nn.Module):
    def __init__(self, features, num_classes=10, init_weights=True):
        super(SelectiveNet, self).__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

        self.aux_classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, num_classes)
        )

        self.selector = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        # classification head (f)

        prediction_output = self.classifier(x)

        # selection head (g)

        selection_output = self.selector(x)

        # auxiliary head (h)

        auxiliary_output = self.aux_classifier(x)

        return prediction_output, selection_output, auxiliary_output

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


def _vgg(batch_norm, **kwargs):
    model = SelectiveNet(make_layers(cfg, batch_norm=batch_norm), **kwargs)
    return model


def SelectiveNet_vgg16(**kwargs):
    return _vgg(False, **kwargs)


def SelectiveNet_vgg16_bn(**kwargs):
    return _vgg(True, **kwargs)


class OverAllLoss(nn.Module):
    def __init__(self, alpha=0.5, lambda=32, coverage=0.7):
        super(OverAllLoss, self).__init__()
        self.alpha = alpha
        self.lambda = lambda
        self.coverage = coverage

    def forward(self, prediction_input, selection_input, aux_input, labels):
        sel_log_prob = -1.0 * F.log_softmax(prediction_input, 1) * selection_input
        sel_risk = sel_log_prob.gather(1, labels.unsqueeze(1))
        sel_risk = sel_risk.mean()

        aux_log_prob = -1.0 * F.log_softmax(aux_input, 1)
        aux_loss = aux_log_prob.gather(1, labels.unsqueeze(1))
        aux_loss = aux_loss.mean()

        emp_coverage = selection_input.mean()

        return self.alpha * (sel_risk / emp_coverage + self.lambda * max(self.coverage - emp_coverage, 0) ** 2) + (
                    1 - self.alpha) * aux_loss \
            , sel_risk, emp_coverage


