  
import torch
from torch import nn
import torch.nn.functional as F

EPSILON_FP16 = 1e-5

class LqLoss(nn.Module):
    def __init__(self, q=0.5):
        super().__init__()
        self.q = q

    def forward(self, y_pred, y_true):
        loss = y_pred * y_true
        loss = (1 - (loss + EPSILON_FP16) ** self.q) / self.q
        return loss.mean()

class LSoftLoss(nn.Module):
    def __init__(self, beta=0.5):
        super().__init__()
        self.beta = beta

    def forward(self, y_pred, y_true):
        with torch.no_grad():
            y_true_update = self.beta * y_true + (1 - self.beta) * y_pred
        
        return F.binary_cross_entropy(y_pred, y_true_update)

class CuratedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, output, target):
        return self.bce(output, target)


class NoisyCuratedLoss(nn.Module):
    def __init__(self, noisy_type, beta=0.7, q=0.7):
        super().__init__()
        if noisy_type=="lsoft":
            self.noisy_loss = LSoftLoss(beta=beta)
        elif noisy_type=="lq":
            self.noisy_loss = LqLoss(q=q)
        self.curated_loss = CuratedLoss()
        self.sigmoid = nn.Sigmoid()

    def forward(self, output, target):
        target, clean = target
        clean = clean.reshape(-1)
        bs, s, o = target.shape
        output = self.sigmoid(output)
        output = torch.clamp(output, min=EPSILON_FP16, max=1.0-EPSILON_FP16)

        output = output.reshape(bs*s,o)
        target = target.reshape(bs*s,o)

        noisy_indexes = (clean == 0).nonzero().squeeze(1)
        curated_indexes = clean.nonzero().squeeze(1)

        noisy_len = noisy_indexes.shape[0]
        if noisy_len > 0:
            noisy_target = target[noisy_indexes]
            noisy_output = output[noisy_indexes]
            noisy_loss = self.noisy_loss(noisy_output, noisy_target)
            noisy_loss = noisy_loss * (noisy_len / bs)
        else:
            noisy_loss = 0.0

        curated_len = curated_indexes.shape[0]
        if curated_len > 0:
            curated_target = target[curated_indexes]
            curated_output = output[curated_indexes]
            curated_loss = self.curated_loss(curated_output, curated_target)
            curated_loss = curated_loss * (curated_len / bs)
        else:
            curated_loss = 0.0

        loss = noisy_loss * 0.5 + curated_loss * 0.5
        return loss, {"noisy_loss": noisy_loss, "curated_loss": curated_loss}