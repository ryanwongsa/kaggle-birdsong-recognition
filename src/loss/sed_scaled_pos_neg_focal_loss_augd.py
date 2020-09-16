import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class SedScaledPosNegFocalLossAugd(nn.Module):
    def __init__(self, gamma=0.0, alpha_1=1.0, alpha_0=1.0, secondary_factor=0.1, primary_drop_out_factor=0.3):
        super().__init__()

        self.loss_fn = nn.BCELoss(reduction='none')
        self.secondary_factor = secondary_factor
        self.gamma = gamma
        self.alpha_1 = alpha_1
        self.alpha_0 = alpha_0
        self.primary_drop_out_factor = primary_drop_out_factor
        self.loss_keys = ["bce_loss", "F_loss", "FScaled_loss", "F_loss_0", "F_loss_1"]

    def forward(self, y_pred, y_target):
        y_true = y_target["all_labels"]
        y_sec_true = y_target["secondary_labels"]
        bs, s, o = y_true.shape
        
        # Sigmoid has already been applied in the model
        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)
        y_true = y_true.reshape(bs*s,o)
        y_sec_true = y_sec_true.reshape(bs*s,o)
        y_primary_true = y_true - y_sec_true
        
        with torch.no_grad():
            y_all_ones_mask = torch.ones_like(y_true, requires_grad=False)
            y_all_zeros_mask = torch.zeros_like(y_true, requires_grad=False)
            y_all_mask = torch.where(y_true > 0.0, y_all_ones_mask, y_all_zeros_mask)
            y_ones_mask = torch.ones_like(y_sec_true, requires_grad=False)
            y_zeros_mask = torch.ones_like(y_sec_true, requires_grad=False) *self.secondary_factor
            y_secondary_mask = torch.where(y_sec_true > 0.0, y_zeros_mask, y_ones_mask)
            
            y_primary_mask = torch.where(y_primary_true > 0.0, y_all_ones_mask, y_all_zeros_mask)
            y_primary_mask = torch.where(torch.rand(y_primary_mask.shape, device=y_primary_mask.device)<=self.primary_drop_out_factor, y_primary_mask, y_all_zeros_mask)
            
        bce_loss = self.loss_fn(y_pred, y_true)
        pt = torch.exp(-bce_loss)
        F_loss_0 = (self.alpha_0*(1-y_all_mask)) * (1-pt)**self.gamma * bce_loss
        F_loss_1 = (self.alpha_1*y_all_mask) * (1-pt)**self.gamma * bce_loss

        F_loss = F_loss_0 + F_loss_1
        
        
        FScaled_loss = y_secondary_mask*F_loss
        
        FScaled_loss2 = (1-y_primary_mask)*FScaled_loss
        FScaled_loss2 = FScaled_loss2.mean()

        return FScaled_loss2, {"bce_loss": bce_loss.mean(), "F_loss_1": F_loss_1.mean(), "F_loss_0": F_loss_0.mean(), "F_loss": F_loss.mean(), "FScaled_loss": FScaled_loss.mean() }
