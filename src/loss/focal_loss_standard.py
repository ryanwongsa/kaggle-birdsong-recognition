import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class FocalLossStandard(nn.Module):
    def __init__(self, gamma=0.0, alpha=1.0):
        super().__init__()

        self.loss_fn = nn.BCELoss(reduction='none')
        self.gamma = gamma
        self.alpha = alpha
        self.loss_keys = ["bce_loss", "F_loss"]

    def forward(self, y_pred, target):
        y_true = target["all_labels"]
        bs, s, o = y_true.shape

        # Sigmoid has already been applied in the model

        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)
        y_true = y_true.reshape(bs*s,o)

        bce_loss = self.loss_fn(y_pred, y_true)
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        
        F_loss = F_loss.mean()

        return F_loss, {"bce_loss": bce_loss.mean(), "F_loss": F_loss }

        