from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
from sklearn.metrics import label_ranking_average_precision_score

class SedLRAPS(Metric):
    def __init__(self, output_transform=lambda x: x):
        self._score = None
        self._num_items = None
        super(SedLRAPS, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._score = 0.0
        self._num_items = 0
        super(SedLRAPS, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred, target = output
        pred = pred["clipwise_output"]
        pred = pred.max(axis=1)[0].detach().cpu().numpy()
        target = target[:,0,:].detach().cpu().numpy()
        for i in range(len(pred)):
            p = pred[i]
            t = target[i]
            self._score += label_ranking_average_precision_score([t], [p])
            self._num_items +=1

    @sync_all_reduce("_score", "_num_items")
    def compute(self):
        return self._score / self._num_items