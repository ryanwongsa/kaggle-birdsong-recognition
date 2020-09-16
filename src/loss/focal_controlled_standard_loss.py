import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class LSoftLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce_logit_loss = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, y_pred, y_true, beta):
        with torch.no_grad():
            y_true_updated = (beta*y_true+(1-beta)*torch.sigmoid(y_pred)) * y_true
        return self.bce_logit_loss(y_pred, y_true_updated)

class FocalControlledStandardLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha=1.0):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

        self.lsoft_fn = LSoftLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_keys = ["bce_loss", "F_loss"]

    def forward(self, y_pred, target, beta, primary_beta):
        y_true, target_primary = target
        
        bs, s, o = y_true.shape

        # y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)

        y_true = y_true.reshape(bs*s,o)

        bce_loss = self.lsoft_fn(y_pred, y_true, beta)
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        F_loss = F_loss.mean()

        return F_loss, {"bce_loss": bce_loss.mean(), "F_loss": F_loss }

        