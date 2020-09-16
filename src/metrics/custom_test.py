from ignite.metrics import Metric
from ignite.exceptions import NotComputableError
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
from sklearn.metrics import label_ranking_average_precision_score
import json
import numpy as np
import glob
from sklearn.metrics import f1_score

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

class CustomTest(Metric):
    def __init__(self, save_dir, combine_files=False, output_transform=lambda x: x):
        self._score = None
        self.sigmoid = torch.nn.Sigmoid()
        self._num_items = None
        self._save_dir = save_dir
        save_dir.mkdir(parents=True, exist_ok=True)
        (save_dir/"final").mkdir(parents=True, exist_ok=True)
        self._combine_files = combine_files
        super(CustomTest, self).__init__(output_transform=output_transform)

    @reinit__is_reduced
    def reset(self):
        self._score = 0.0
        self._num_items = 0
        super(CustomTest, self).reset()

    @reinit__is_reduced
    def update(self, output):
        pred, input_x = output
        target = input_x["coded_labels"]
        filenames = input_x["filenames"]
        pred = self.sigmoid(pred)
        pred = pred.max(axis=1)[0].detach().cpu().numpy()
        target = target[:,0,:].detach().cpu().numpy()
        for i in range(len(pred)):
            p = pred[i]
            t = target[i]
            filename = filenames[i].replace(".mp3","")
            score = label_ranking_average_precision_score([t], [p])
            solution = {
                "predicted": p,
                "targets": t,
                "filename": filename,
                "lraps_score": score
            }

            for threshold in [0.5,0.6,0.7,0.8,0.9]:
                if t.sum()==0.0:
                    solution["background"] = True
                    if ((p>=threshold).sum())>0.0:
                        f1_score_value = f1_score([1], [0], average='micro')
                    else:
                        f1_score_value = f1_score([1], [1], average='micro')
                else:
                    solution["background"] = False
                    f1_score_value = f1_score([t], [p>=threshold], average='micro')
                solution[f"f1_score_{int(threshold*100)}"] = f1_score_value
            
            with open(self._save_dir/f'{filename}.json', 'w') as f:
                f.write(json.dumps(solution, cls=NumpyEncoder))

            self._score += score
            self._num_items +=1

    @sync_all_reduce("_score", "_num_items")
    def compute(self):
        
        if self._combine_files:
            read_files = self._save_dir.glob('*.json')
            with open(self._save_dir/f"final/merged_results.json", "w") as outfile:
                outfile.write('[{}]'.format(
                    ','.join([open(f, "r").read() for f in read_files])))
        return self._score / self._num_items