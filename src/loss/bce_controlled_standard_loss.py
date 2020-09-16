import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class LSoftLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true, beta):
        with torch.no_grad():
            y_true_updated = (beta*y_true+(1-beta)*y_pred) * y_true
        return F.binary_cross_entropy(y_pred, y_true_updated, reduction='none')

class BCEControlledStandardLoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()

        self.lsoft_fn = LSoftLoss()
        self.beta = beta
        self.loss_keys = []

    def forward(self, y_pred, target):
        y_true = target["all_labels"]
        
        bs, s, o = y_true.shape

        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)

        y_true = y_true.reshape(bs*s,o)

        bce_loss = self.lsoft_fn(y_pred, y_true, self.beta).mean(1).mean()

        return bce_loss, {}

        