import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def flatten(x):
    return x.view(x.size(0), -1)

def loss(y_pred, y_true, S=7, B=2, C=1):
    ''' Calculate the loss of YOLO model.
    args:
        y_pred: (Batch, 7 * 7 * 30)
        y_true: dict object that contains:
            class_probs,
            confs,
            coord,
            proid,
            areas,
            upleft,
            bottomright
    '''

    SS = S * S
    scale_class_prob = 1
    scale_object_conf = 1
    scale_noobject_conf = 0.5
    scale_coordinate = 5
    batch_size = y_pred.size(0)

    # ground truth
    _coord = y_true['coord']
    _coord = _coord.view(-1, SS, B, 4)
    _upleft = y_true['upleft']
    _bottomright = y_true['bottomright']
    _areas = y_true['areas']
    _confs = y_true['confs']
    _proid = y_true['proid']
    _probs = y_true['class_probs']

    # Extract the coordinate prediction from y_pred
    coords = y_pred[:, SS * (C + B):].contiguous().view(-1, SS, B, 4)
    wh = torch.pow(coords[:, :, :, 2:4], 2)
    area_pred = wh[:, :, :, 0] * wh[:, :, :, 1]
    centers = coords[:, :, :, 0:2].contiguous()
    floor = centers - (wh * 0.5)
    ceil = centers + (wh * 0.5)

    # Calculate the intersection areas
    intersect_upleft = torch.max(floor, _upleft)
    intersect_bottomright = torch.min(ceil, _bottomright)
    intersect_wh = intersect_bottomright - intersect_upleft
    zeros = Variable(torch.zeros(batch_size, 49, B, 2))
    intersect_wh = torch.max(intersect_wh, zeros)
    intersect = intersect_wh[:, :, :, 0] * intersect_wh[:, :, :, 1]

    # Calculate the best IOU, set 0.0 confidence for worse boxes
    iou = intersect / (_areas + area_pred - intersect)
    best_box = torch.eq(iou, torch.max(iou, 2)[0].unsqueeze(2))
    confs = best_box.float() * _confs

    # Take care of the weight terms
    conid = scale_noobject_conf * (1. - confs) + scale_object_conf * confs
    weight_coo = torch.cat(4 * [confs.unsqueeze(-1)], 3)
    cooid = scale_coordinate * weight_coo
    proid = scale_class_prob * _proid

    # Flatten 'em all
    probs = flatten(_probs)
    proid = flatten(proid)
    confs = flatten(confs)
    conid = flatten(conid)
    coord = flatten(_coord)
    cooid = flatten(cooid)

    true = torch.cat([probs, confs, coord], 1)
    wght = torch.cat([proid, conid, cooid], 1)

    loss = torch.pow(y_pred - true, 2)
    loss = loss * wght
    loss = torch.sum(loss, 1)
    return .5 * torch.mean(loss)

class VanillaYoloV1(nn.Module):
    def __init__(self):
        super(VanillaYoloV1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, (7,7), stride=2)
        self.conv2 = nn.Conv2d(64, 192, (3,3))
        self.conv3_1 = nn.Conv2d(192, 128, (1,1))
        self.conv3_2 = nn.Conv2d(128, 256, (3,3))
        self.conv3_3 = nn.Conv2d(256, 256, (1,1))
        self.conv3_4 = nn.Conv2d(256, 512, (3,3))
        self.conv4_1 = nn.Conv2d(512, 256, (1,1))
        self.conv4_2 = nn.Conv2d(256, 512, (3,3))
        self.conv4_3 = nn.Conv2d(512, 256, (1,1))
        self.conv4_4 = nn.Conv2d(256, 512, (3,3))
        self.conv4_5 = nn.Conv2d(512, 256, (1,1))
        self.conv4_6 = nn.Conv2d(256, 512, (3,3))
        self.conv4_7 = nn.Conv2d(512, 256, (1,1))
        self.conv4_8 = nn.Conv2d(256, 512, (3,3))
        self.conv4_9 = nn.Conv2d(512, 512, (1,1))
        self.conv4_10 = nn.Conv2d(512, 1024, (3,3))
        self.conv5_1 = nn.Conv2d(1024, 512, (1,1))
        self.conv5_2 = nn.Conv2d(512, 1024, (3,3))
        self.conv5_3 = nn.Conv2d(1024, 512, (1,1))
        self.conv5_4 = nn.Conv2d(512, 1024, (3,3))
        self.conv5_5 = nn.Conv2d(1024, 1024, (3,3))
        self.conv5_6 = nn.Conv2d(1024, 1024, (3,3), stride=2)
        self.conv6_5 = nn.Conv2d(1024, 1024, (3,3))
        self.conv6_5 = nn.Conv2d(1024, 1024, (3,3))
        self.dense_1 = nn.Linear(7*7*1024, 4096)
        self.dense_2 = nn.Linear(4096, 7*7*30)

    def forward(self, x):
        x = F.pad(x, (2,3,2,3), 'replicate')
        x = self.conv1(x)
        x = F.max_pool2d(x, (2,2), stride=2)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv2(x)
        x = F.max_pool2d(x, (2,2), stride=2)
        x = self.conv3_1(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv3_4(x)
        x = F.max_pool2d(x, (2,2), stride=2)
        x = self.conv4_1(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv4_6(x)
        x = self.conv4_7(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv4_8(x)
        x = self.conv4_9(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv4_10(x)
        x = F.max_pool2d(x, (2,2), stride=2)
        x = self.conv5_1(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv5_2(x)
        x = self.conv5_3(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv5_4(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv5_5(x)
        x = F.pad(x, (1,1,1,1), 'replicate')
        x = self.conv5_6(x)
        x = torch.reshape(x, (-1,7*7*1024))
        x = self.dense_1(x)
        x = self.dense_2(x)
        return x
