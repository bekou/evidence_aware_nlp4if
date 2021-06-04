import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class AngularLoss(nn.Module):
    """
    Implementation of https://arxiv.org/abs/1708.01682
    Args:
        alpha: The angle (as described in the paper), specified in degrees.
    """

    def __init__(self, alpha):
        super(AngularLoss, self).__init__()
        self.alpha = torch.tensor(np.radians(alpha))

    def forward(self, anchor, positive, negative, size_average=True):
        distance_anchor_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        cluster_center=(anchor+positive)/2
        distance_negative_cluster = (negative - cluster_center).pow(2).sum(1)  # .pow(.5)
        
        sq_tan_alpha = torch.tan(self.alpha) ** 2
        losses = F.relu(distance_anchor_positive - 4 * sq_tan_alpha * distance_negative_cluster)
        #print (distance_anchor_positive - 4 * sq_tan_alpha * distance_negative_cluster)
        return losses.mean() if size_average else losses.sum()