import torch
import math
import numpy as np
import torch.nn as nn
from torchvision.transforms import ToPILImage
import os


def calculate_mask(residuals):
    residuals = residuals.squeeze()

    median_residual = torch.median(residuals)
    inlier_loss = torch.where(residuals <= median_residual, 1.0, 0.0)

    kernel = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).unsqueeze(0).unsqueeze(0).cuda()
    has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)
    has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, kernel, padding = "same")
    has_inlier_neighbors = torch.where(has_inlier_neighbors >= 0.5, 1.0, 0.0)

    kernel_16 = 1/(16*16) * torch.ones((1,1,16,16)).cuda()
    if has_inlier_neighbors.shape[1] % 8 != 0:
        pad_h = 8 - (has_inlier_neighbors.shape[1] % 8) + 8
    else:
        pad_h = 8

    if has_inlier_neighbors.shape[2] % 8 != 0:
        pad_w = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
    else:
        pad_w = 8

    padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))
    padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate").cuda()

    is_inlier_patch = torch.nn.functional.conv2d(padded_weights.unsqueeze(0), kernel_16, stride = 8)

    is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch, scale_factor = 8)
    is_inlier_patch = is_inlier_patch.squeeze()

    padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

    if padding_indexing[1] == 0:
        padding_indexing[1] = has_inlier_neighbors.shape[1] + padding_indexing[0]
    if padding_indexing[3] == 0:
        padding_indexing[3] = has_inlier_neighbors.shape[2] + padding_indexing[2]

    is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]

    is_inlier_patch = torch.where(is_inlier_patch >= 0.6, 1.0, 0.0)

    mask = (is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze() >= 1e-3).cuda()

    return mask


kernel_size = 16
kernel_smooth = 1/(kernel_size*kernel_size) * torch.ones((1,1,kernel_size,kernel_size)).cuda()
kernel = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]]).unsqueeze(0).unsqueeze(0).cuda()
kernel_16 = 1/(16*16) * torch.ones((1,1,16,16)).cuda()
def calculate_mask_proba(residuals):
    median_residual = torch.median(residuals)
    inlier_loss = residuals

    has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)
    has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, kernel, padding = "same")
    
    if has_inlier_neighbors.shape[1] % 8 != 0:
        pad_h = 8 - (has_inlier_neighbors.shape[1] % 8) + 8
    else:
        pad_h = 8

    if has_inlier_neighbors.shape[2] % 8 != 0:
        pad_w = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
    else:
        pad_w = 8

    padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))
    padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate")

    is_inlier_patch = torch.nn.functional.conv2d(padded_weights.unsqueeze(0), kernel_16, stride = 8)

    is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch, scale_factor = 8)
    is_inlier_patch = is_inlier_patch.squeeze()

    padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

    if padding_indexing[1] == 0:
        padding_indexing[1] = has_inlier_neighbors.shape[1] + padding_indexing[0]
    if padding_indexing[3] == 0:
        padding_indexing[3] = has_inlier_neighbors.shape[2] + padding_indexing[2]

    is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]

    mask = is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze()
    mask_median = torch.median(mask)
    mask = mask - mask_median
    mask = torch.sigmoid(mask)

    sampling_weights = torch.rand(mask.shape, device = 'cuda:0')
    mask = torch.gt(mask, sampling_weights).float()

    
    mask = mask.unsqueeze(0)
    mask = torch.nn.functional.conv2d(mask, kernel_smooth, padding = "same").squeeze()
    mask = torch.where(mask >= 0.6, 0.0, 1.0)

    return mask.cuda()


class RobustLoss(torch.nn.Module):
    def __init__(self, n_residuals = 1, hidden_size = 1, per_channel = False):
        super(RobustLoss, self).__init__()
        # Define a learnable parameter
        self.n_residuals = n_residuals
        self.linear1 = torch.nn.Linear(n_residuals, hidden_size, device = 'cuda:0')
        self.sigmoid1 = torch.nn.Sigmoid()

        self.per_channel = per_channel
        if per_channel:
            self.channel = 3
        else:
            self.channel = 1

        self.kernel_size = 16
        kernel_16 = 1/(self.kernel_size*self.kernel_size) * torch.ones((self.kernel_size, self.kernel_size))
        self.kernel_16 = kernel_16.view(1, 1, self.kernel_size, self.kernel_size).repeat(self.channel, self.channel, 1, 1).cuda()
        kernel_3 = torch.tensor([[1/9, 1/9, 1/9], [1/9, 1/9, 1/9], [1/9, 1/9, 1/9]])
        self.kernel_3 = kernel_3.view(1, 1, 3, 3).repeat(self.channel, self.channel, 1, 1).cuda()

    def forward(self, residuals):
        medians = torch.median(residuals[0].flatten(start_dim=1), dim=1)[0]

        if self.per_channel == True:
            for i in range(self.channel):
                residuals[0, i] = residuals[0, i] - medians[i]
            inlier_loss = residuals[0]
        else:
            inlier_loss = residuals[0] - torch.median(residuals[0].flatten())      
                
        has_inlier_neighbors = torch.unsqueeze(inlier_loss, 0)

        has_inlier_neighbors = torch.nn.functional.conv2d(has_inlier_neighbors, self.kernel_3, padding = "same")

        if has_inlier_neighbors.shape[2] % 8 != 0:
            pad_h = 8 - (has_inlier_neighbors.shape[2] % 8) + 8
        else:
            pad_h = 8

        if has_inlier_neighbors.shape[3] % 8 != 0:
            pad_w = 8 - (has_inlier_neighbors.shape[3] % 8) + 8
        else:
            pad_w = 8

        padding = (math.ceil(pad_w/2), math.floor(pad_w/2), math.ceil(pad_h/2), math.floor(pad_h/2))

        padded_weights = torch.nn.functional.pad(has_inlier_neighbors, padding, mode = "replicate").cuda()

        is_inlier_patch = torch.nn.functional.conv2d(padded_weights.squeeze(0), self.kernel_16, stride = 8)

        is_inlier_patch = torch.nn.functional.interpolate(is_inlier_patch.unsqueeze(0), scale_factor = (8,8))

        is_inlier_patch = is_inlier_patch.squeeze()

        padding_indexing = [padding[2]-4,-(padding[3]-4), padding[0]-4,-(padding[1]-4)]

        if padding_indexing[1] == 0:
            padding_indexing[1] = has_inlier_neighbors.shape[2] + padding_indexing[0]
        if padding_indexing[3] == 0:
            padding_indexing[3] = has_inlier_neighbors.shape[3] + padding_indexing[2]

        if self.per_channel == True:
            is_inlier_patch = is_inlier_patch[:, padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]
        else:
            is_inlier_patch = is_inlier_patch[ padding_indexing[0]:padding_indexing[1], padding_indexing[2]:padding_indexing[3] ]
            is_inlier_patch = is_inlier_patch.unsqueeze(0)

        mask = (is_inlier_patch.squeeze() + has_inlier_neighbors.squeeze() + inlier_loss.squeeze()).cuda()
        mask_before_log = mask

        shape = mask.shape
        res = torch.flatten(residuals, 1)
        res = res[1:]
        res = torch.transpose(res, 0, 1)

        mask = mask.flatten().unsqueeze(1)
        if self.n_residuals > 1:
            mask = torch.cat((res, mask), 1)
        mask = self.linear1(mask)
        mask = self.sigmoid1(mask)
        mask = mask.reshape(shape)
        if self.per_channel == True:
            mask = torch.median(mask, dim=0, keepdim=True)[0]
        else:
            mask = mask.unsqueeze(0)

        return mask, mask_before_log
    
    def threshold(self, linear, sigmoid, x, residuals):
        shape = x.shape
        res = torch.flatten(residuals, 1)
        res = torch.transpose(res, 0, 1)

        x = x.flatten().unsqueeze(1)

        res = torch.cat((res, x), 1)
        x = linear(res)
        x = sigmoid(x)
        x = x.reshape(shape)
        return x
    
    def zero_center(self, x):
        return x - torch.mean(x)

