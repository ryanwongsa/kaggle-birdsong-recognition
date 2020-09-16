import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class F1ControlledLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigmoid = nn.Sigmoid()

        self.loss_keys = ["precision", "recall", "f1"]

    def forward(self, y_pred, target, beta, primary_beta):
        y_true, target_primary = target
        
        bs, s, o = y_true.shape

        y_pred = self.sigmoid(y_pred)
        y_pred = torch.clamp(y_pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        y_pred = y_pred.reshape(bs*s,o)

        y_true = y_true.reshape(bs*s,o)
        with torch.no_grad():
            y_true_updated = (beta*y_true+(1-beta)*y_pred)*y_true

        tp = (y_true*y_pred).sum(0)
        fp = ((1-y_true)*y_pred).sum(0)
        fn = y_true_updated*(1-y_pred).sum(0)
        
        p = tp / (tp + fp +EPSILON_FP16)
        r = tp / (tp + fn +EPSILON_FP16)

        f1 = (2*p*r / (p+r+EPSILON_FP16)).mean()
        
        loss = 1 - f1
        return loss, {"precision": p.mean(), "recall": r.mean(), "f1": f1}