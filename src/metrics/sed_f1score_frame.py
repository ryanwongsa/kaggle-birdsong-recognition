from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
from sklearn.metrics import f1_score
import numpy as np
class SedF1ScoreFrame(Metric):
    def __init__(self, threshold=0.5, output_transform=lambda x: x):
        self._score = None
        self._num_items = None
        self.threshold = threshold
        super(SedF1ScoreFrame, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._score = 0.0
        self._num_items = 0
        super(SedF1ScoreFrame, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred, target = output

        ## NB: SIGMOID IS NOT APPLIED HERE

        pred_frame = pred["framewise_output"]
        pred_frame = pred_frame.max(axis=1)[0].max(axis=1)[0].detach().cpu().numpy()
        target = target[:,0,:].detach().cpu().numpy()
        
        pred_clip = pred["clipwise_output"]
        pred_clip = pred_clip.max(axis=1)[0].detach().cpu().numpy()
        
        for i, (pf, pc, t) in enumerate(zip(pred_frame, pred_clip,target)):
            if t.sum()==0.0:
                if (np.sum((pf>=self.threshold) & (pc>=0.3)))>0.0:
                    score = f1_score([1], [0], average='micro')
                else:
                    score = f1_score([1], [1], average='micro')
            else:
                score = f1_score([t], [(pf>=self.threshold) & (pc>=0.3)], average='micro')
            self._score += score
            self._num_items +=1

    @sync_all_reduce("_score", "_num_items")
    def compute(self):
        return self._score / self._num_items