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

class BCEControlledLoss(nn.Module):
    def __init__(self, primary_loss_factor = 1.0):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

        self.lsoft_fn = LSoftLoss()
        self.loss_keys = ["primary_loss" , "bce_loss"]
        self.primary_loss_factor = primary_loss_factor

    def forward(self, y_pred, target, beta, primary_beta):
        y_true, target_primary = target
        
        bs, s, o = y_true.shape

        y_pred = self.sigmoid(y_pred)
        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)

        y_true = y_true.reshape(bs*s,o)

        bce_loss = self.lsoft_fn(y_pred, y_true, beta).mean(1)
        if target_primary is not None:
            primary_preds = y_pred.view(-1)[[i*o+ix[0] for i, ix in enumerate(target_primary)]]
            primary_loss = self.lsoft_fn(primary_preds, torch.ones_like(primary_preds), primary_beta) * self.primary_loss_factor

            loss = (bce_loss + primary_loss).mean()
        
            return loss, {"primary_loss": primary_loss.mean(), "bce_loss": bce_loss.mean()}
        else:
            loss = bce_loss.mean()
        
            return loss, {"primary_loss": 0, "bce_loss": bce_loss.mean()}

        