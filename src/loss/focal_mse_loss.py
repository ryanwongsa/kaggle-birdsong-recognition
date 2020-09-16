import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class LSoftLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = torch.nn.MSELoss(reduction='none')

    def forward(self, y_pred, y_true, beta):
        with torch.no_grad():
            y_true_updated = (beta*y_true+(1-beta)*y_pred) * y_true
        return self.mse_loss(y_pred, y_true_updated)

class FocalMSELoss(nn.Module):
    def __init__(self, gamma=0.0, alpha=1.0):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

        self.lsoft_fn = LSoftLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_keys = ["mse_loss", "F_loss"]

    def forward(self, y_pred, target, beta, primary_beta):
        y_true, target_primary = target
        
        bs, s, o = y_true.shape

        y_pred = torch.sigmoid(y_pred)
        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)

        y_true = y_true.reshape(bs*s,o)

        mse_loss = self.lsoft_fn(y_pred, y_true, beta)
        pt = torch.exp(-mse_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * mse_loss
        F_loss = F_loss.mean()

        return F_loss, {"mse_loss": mse_loss.mean(), "F_loss": F_loss }