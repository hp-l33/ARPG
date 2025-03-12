import torch
import numpy as np
import os
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class INatLatentDataset(Dataset):
    def __init__(self, root_dir, transform=transforms.ToTensor()):
        categories_set = set()
        self.categories = sorted([int(i) for i in list(os.listdir(root_dir))])
        self.samples = []

        for tgt_class in self.categories:
            tgt_dir = os.path.join(root_dir, str(tgt_class))
            for root, _, fnames in sorted(os.walk(tgt_dir, followlinks=True)):
                for fname in fnames:
                    path = os.path.join(root, fname)
                    item = (path, tgt_class)
                    self.samples.append(item)
        self.num_examples = len(self.samples)
        self.indices = np.arange(self.num_examples)
        self.num = self.__len__()
        print("Loaded the dataset from {}. It contains {} samples.".format(root_dir, self.num))
        self.transform = transform
  
    def __len__(self):
        return self.num_examples
  
    def __getitem__(self, index):
        index = self.indices[index]
        sample = self.samples[index]
        latents = np.load(sample[0])
        latents = self.transform(latents) # 1 * aug_num * block_size

        # select one of the augmented crops
        aug_idx = torch.randint(0, latents.shape[1], (1,)).item()
        latents = latents[:, aug_idx, :]
        label = sample[1]
        
        return latents, label


def build_imagenet_code(args):
    return INatLatentDataset(args.code_path)


import json
import linecache

class PretoeknizedDataSetJSONL(Dataset):
    def __init__(self, data_path='/public/liguoqi/ssl/wds/datasets/maskgitvq.json'):
        super().__init__()
        self.jsonl_file = data_path
        self.num_lines = sum(1 for _ in open(self.jsonl_file))
        # Ensure the file is cached
        linecache.checkcache(self.jsonl_file)
        print("Number of data:", self.num_lines)

    def __len__(self):
        return self.num_lines

    def __getitem__(self, idx):
        line = linecache.getline(self.jsonl_file, idx + 1).strip()
        data = json.loads(line)
        return torch.tensor(data["tokens"]), torch.tensor(data["class_id"])