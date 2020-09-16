import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class SedRemovedFocalLoss(nn.Module):
    def __init__(self, gamma=0.0, alpha=1.0, secondary_factor=0.1):
        super().__init__()

        self.loss_fn = nn.BCELoss(reduction='none')
        self.secondary_factor = secondary_factor
        self.gamma = gamma
        self.alpha = alpha
        self.loss_keys = ["bce_loss", "F_loss", "FScaled_loss"]

    def forward(self, y_pred, y_target):
        y_true = y_target["all_labels"]
        y_sec_true = y_target["secondary_labels"]
        bs, s, o = y_true.shape

        # Sigmoid has already been applied in the model

        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)
        y_true = y_true.reshape(bs*s,o)
        y_sec_true = y_sec_true.reshape(bs*s,o)
        
        with torch.no_grad():
            y_ones_mask = torch.ones_like(y_sec_true)
            y_zeros_mask = torch.ones_like(y_sec_true) *self.secondary_factor
            y_mask = torch.where(y_sec_true > 0.0, y_zeros_mask, y_ones_mask)

        bce_loss = self.loss_fn(y_pred, y_true)
        pt = torch.exp(-bce_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        FScaled_loss = y_mask*F_loss
        
        FScaled_loss = FScaled_loss.mean()

        return FScaled_loss, {"bce_loss": bce_loss.mean(), "F_loss": F_loss.mean(), "FScaled_loss": FScaled_loss }

        