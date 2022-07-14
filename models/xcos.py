import torch
import torch.nn as nn
import torch.nn.functional as F



cos = nn.CosineSimilarity(dim=1, eps=1e-6)


def Xcos(ftrain, ftest):
    B, n2, n1, C, H, W = ftrain.size()

    ftrain = Long_alignment(ftrain, ftest)

    ftrain = ftrain.view(-1, C, H, W).permute(0, 2, 3, 1)
    ftest = ftest.view(-1, C, H, W).permute(0, 2, 3, 1)

    ftrain = ftrain.contiguous().view(-1, ftrain.size(3))
    ftest = ftest.contiguous().view(-1, ftest.size(3))

    cos_map = 10*cos(ftrain,ftest).view(B*n2, n1, -1)

    return cos_map



def Long_alignment(support_x, query_x):
    b, q, s, c, h, w = support_x.shape
    support_x = F.normalize(support_x, p=2, dim=-3, eps=1e-12)
    query_x = F.normalize(query_x, p=2, dim=-3, eps=1e-12)
    support_x = support_x.view(b, q, s, c, h * w)
    query_x = query_x.view(b, q, s, c, h * w).transpose(3, 4) 

    Mt = torch.matmul(query_x, support_x)

    Mt = F.softmax(Mt,dim=4)

    support_x = support_x.transpose(3,4)

    align_support = torch.matmul(Mt, support_x)
    align_support = align_support.transpose(3,4)

    return align_support
