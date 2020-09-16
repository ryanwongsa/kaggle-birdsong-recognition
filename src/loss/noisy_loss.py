import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class LSoftLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, beta):
        with torch.no_grad():
            y_true_updated = beta*y_true+(1-beta)*y_pred
        return F.binary_cross_entropy(y_pred, y_true_updated, reduction='none')

class NoisyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.noisy_loss_fn = LSoftLoss()

        self.sigmoid = nn.Sigmoid()

        self.loss_keys = ["zeros_loss", "ones_loss", "primary_loss", "overall_target_loss"]

    def forward(self, output, target, beta, primary_beta):
        target_all, target_primary = target
        
        bs, s, o = target_all.shape

        output = self.sigmoid(output)
        output = torch.clamp(output, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        output = output.reshape(bs*s,o)

        target_all = target_all.reshape(bs*s,o)
        noisy_loss = self.noisy_loss_fn(output, target_all, beta)
        ones_loss = (((noisy_loss*target_all).sum()/(target_all.sum()+EPSILON_FP16)))
        zeros_loss = (((noisy_loss*(1-target_all)).sum()/((1-target_all).sum()+EPSILON_FP16)))
        
        loss = (zeros_loss + ones_loss)/2
        return loss, {"zeros_loss": zeros_loss, "ones_loss":ones_loss}