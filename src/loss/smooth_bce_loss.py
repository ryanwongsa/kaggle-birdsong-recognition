import torch
import torch.nn as nn
EPSILON_FP16 = 1e-5
class SmoothBCELoss(nn.Module):
    def __init__(self, smooth=0.1):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.sigmoid = nn.Sigmoid()
        self.smooth = smooth

    def forward(self, pred, actual):
        bs, s, o = pred.shape
        pred = pred.reshape(bs*s, o)
        pred = self.sigmoid(pred)
        pred = torch.clamp(pred, min=EPSILON_FP16, max=1.0-EPSILON_FP16)
        actual = actual.reshape(bs*s, o)
        loss = self.bce(pred, actual)

        indices_target_0 = ((actual == 0.0).nonzero())[:,1]
        indices_pred_thresh = ((pred <= self.smooth).nonzero())[:,1]
        combined = torch.cat((indices_target_0, indices_pred_thresh))
        uniques, counts = combined.unique(return_counts=True)
        intersection = uniques[counts > 1]
        loss[0, intersection] = 0.0

        return loss.mean(), {}