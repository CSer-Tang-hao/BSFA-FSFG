import torch
from skimage import measure
import torch.nn.functional as F
import math
import torch.nn as nn

def AOLM(feature_maps):
    width = feature_maps.size(-1)
    height = feature_maps.size(-2)
    A = torch.sum(feature_maps, dim=1, keepdim=True)
    a = torch.mean(A, dim=[2, 3], keepdim=True)
    M = (A > a).float()


    coordinates = []
    for i, m in enumerate(M):
        mask_np = m.cpu().numpy().reshape(height, width)
        component_labels = measure.label(mask_np)

        properties = measure.regionprops(component_labels)
        areas = []
        for prop in properties:
            areas.append(prop.area)
        if len(areas)==0:
            bbox = [0,0,height, width]
        else:

            max_idx = areas.index(max(areas))

            bbox = properties[max_idx].bbox

        temp = 84/width
        temp = math.floor(temp)
        x_lefttop = bbox[0] * temp - 1
        y_lefttop = bbox[1] * temp - 1
        x_rightlow = bbox[2] * temp- 1
        y_rightlow = bbox[3] * temp - 1
        if x_lefttop < 0:
            x_lefttop = 0
        if y_lefttop < 0:
            y_lefttop = 0

        coordinate = [x_lefttop, y_lefttop, x_rightlow, y_rightlow]
        coordinates.append(coordinate)
    return coordinates

def crop_featuremaps(raw_imgs, feature_maps):
    batch_size = feature_maps.size(0)
    coordinates = AOLM(feature_maps)
    crop_imgs = torch.zeros([batch_size,3,84,84]).cuda()
    for i in range(batch_size):
        [x0, y0, x1, y1] = coordinates[i]
        crop_imgs[i:i+1] = F.interpolate(raw_imgs[i:i + 1, :, x0:(x1+1), y0:(y1+1)], size=(84, 84),
                                                mode='bilinear', align_corners=True)
    
    return crop_imgs

def drop_featuremaps(feature_maps):
    width = feature_maps.size(-1)
    height = feature_maps.size(-2)
    A = torch.sum(feature_maps, dim=1, keepdim=True)
    a = torch.max(A,dim=3,keepdim=True)[0]
    a = torch.max(a,dim=2,keepdim=True)[0]
    threshold = 0.85
    M = (A<=threshold*a).float()
    fm_temp = feature_maps*M
    return fm_temp
