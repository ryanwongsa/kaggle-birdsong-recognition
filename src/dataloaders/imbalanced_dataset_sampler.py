# Modified from: https://github.com/ufoym/imbalanced-dataset-sampler
import torch
class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, dataset, get_label):
        self.indices = list(range(len(dataset)))
        self.num_samples = len(self.indices)
            
        label_to_count = {}
        for idx in self.indices:
            label = dataset.get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        weights = [1.0 / label_to_count[dataset.get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)
                
    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples