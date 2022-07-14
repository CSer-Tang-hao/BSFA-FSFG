import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import utils as vutils
from models.resnet12 import resnet12
from models.conv4 import ConvNet4
from .xcos import Xcos
from .BAS import crop_featuremaps, drop_featuremaps




class Model(nn.Module):
    def __init__(self, num_classes=64, backbone='C'):
        super(Model, self).__init__()
        self.backbone = backbone

        
        if self.backbone == 'R':
            print('Using ResNet12')
            self.base = resnet12()
            # self.width = 6
            self.in_channel = 512
            self.temp = 64
        else:
            print('Using Conv64')
            self.base = ConvNet4()
            # self.width = 5
            self.in_channel = 64
            self.temp = 8
        self.nFeat = self.base.nFeat


        self.clasifier1 = nn.Linear(self.nFeat, num_classes) 
        self.clasifier2 = nn.Linear(self.nFeat,num_classes)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, xtrain, xtest, ytrain, ytest):


        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)
        ytrain = ytrain.transpose(1, 2)

        xtrain = xtrain.view(-1, xtrain.size(2), xtrain.size(3), xtrain.size(4))
        xtest = xtest.view(-1, xtest.size(2), xtest.size(3), xtest.size(4))

        x_all = torch.cat((xtrain, xtest), 0)
        f = self.base(x_all) 
        h = f.size(-2)
        w = f.size(-1)

        drop_f = drop_featuremaps(f)
        crop_imgs = crop_featuremaps(x_all, f)
        crop_f = self.base(crop_imgs)

        if self.training:
            flatten_f = self.avgpool(drop_f)
            flatten_f = flatten_f.view(flatten_f.size(0),-1)
            flatten_crop_f = self.avgpool(crop_f)
            flatten_crop_f = flatten_crop_f.view(flatten_crop_f.size(0),-1)
            glo1 = self.clasifier1(flatten_f)

            glo2 = self.clasifier2(flatten_crop_f)
        

        ftrain = f[:batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, -1) 
        

        # Getting Prototype
        ftrain = torch.bmm(ytrain, ftrain)  
        ftrain = ftrain.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain))
        ftrain = ftrain.view(batch_size, -1, *f.size()[1:]) 

        ftest = f[batch_size * num_train:]
        ftest = ftest.view(batch_size, num_test, *f.size()[1:])  


        f1 = ftrain.unsqueeze(1).repeat(1, num_test, 1, 1, 1, 1)
        f2 = ftest.unsqueeze(2).repeat(1, 1, K, 1, 1, 1)


        ftrain_crop = crop_f[:batch_size * num_train]
        ftrain_crop = ftrain_crop.view(batch_size, num_train, -1) 
        

        # Getting Prototype
        ftrain_crop = torch.bmm(ytrain, ftrain_crop)  
        ftrain_crop = ftrain_crop.div(ytrain.sum(dim=2, keepdim=True).expand_as(ftrain_crop))
        ftrain_crop = ftrain_crop.view(batch_size, -1, *crop_f.size()[1:])  

        ftest_crop = crop_f[batch_size * num_train:]
        ftest_crop = ftest_crop.view(batch_size, num_test, *crop_f.size()[1:])  

        f1_crop = ftrain_crop.unsqueeze(1).repeat(1, num_test, 1, 1, 1, 1)
        f2_crop = ftest_crop.unsqueeze(2).repeat(1, 1, K, 1, 1, 1)

        similar2 = Xcos(f1,f2)  

        similar1 = Xcos(f1_crop,f2_crop)  


        s1 = similar1.view(-1,K,h*w)
        s2 = similar2.view(-1,K,h*w)

        if not self.training:
            return s1.sum(-1)*0.5 + s2.sum(-1)*0.5

   
        return s1, s2, glo1, glo2  





