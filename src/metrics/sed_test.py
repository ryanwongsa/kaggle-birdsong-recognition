from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
from sklearn.metrics import label_ranking_average_precision_score
import json
import numpy as np
import glob
from sklearn.metrics import f1_score
from config_params.configs import BIRD_CODE


INV_BIRD_CODE = {v: k for k, v in BIRD_CODE.items()}

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class SedTest(Metric):
    def __init__(self, save_dir, combine_files=False, output_transform=lambda x: x):
        self._score_frame = None
        self._score_clip = None
        self.sigmoid = torch.nn.Sigmoid()
        self._num_items = None
        self._save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir/"final").mkdir(parents=True, exist_ok=True)
        self._combine_files = combine_files
        super(SedTest, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._score_frame = 0.0
        self._score_clip = 0.0
        self._num_items = 0
        super(SedTest, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred, y_true = output
        target = y_true["all_labels"]
        filenames = y_true["filename"]
        pred_clip = pred["clipwise_output"]
        pred_clip = pred_clip.max(axis=1)[0].detach().cpu().numpy()
        target = target[:,0,:].detach().cpu().numpy()

        pred_frames = pred["framewise_output"]
        pred_frames = pred_frames.max(axis=1)[0].max(axis=1)[0].detach().cpu().numpy()
        for i in range(len(pred_clip)):
            p_c = pred_clip[i]
            t = target[i]
            p_f = pred_frames[i]
            filename = filenames[i]
            score_clip = label_ranking_average_precision_score([t], [p_c])
            score_frames = label_ranking_average_precision_score([t], [p_f])
            solution = {
                "filename": filename
                "predicted_frames": p_f,
                "predicted_clip": p_c,
                "targets": t,
                "lraps_score_clip": score_clip,
                "lraps_score_frames": score_frames
            }
            
            with open(self._save_dir/f'{filename}.json', 'w') as f:
                f.write(json.dumps(solution, cls=NumpyEncoder))

            self._score_frame += score_frames
            self._score_clip += score_clip
            self._num_items +=1

    @sync_all_reduce("_score", "_num_items")
    def compute(self):
        
        if self._combine_files:
            read_files = self._save_dir.glob('*.json')
            with open(self._save_dir/f"final/merged_results.json", "w") as outfile:
                outfile.write('[{}]'.format(
                    ','.join([open(f, "r").read() for f in read_files])))
                
        return self._score_frame / self._num_items