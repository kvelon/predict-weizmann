import numpy as np
import torch
import pytorch_lightning as pl
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

data_path = Path('/home/kelvinfung/Documents/hand-wave/data/')

class WeizmannDataset(Dataset):
    def __init__(self, num_context_frames, num_target_frames, 
                 action_types="flipped_jumpingjack"):
        super().__init__()
        if action_types is not None:
            if isinstance(action_types, list):
            # A list of action types
                arr_lst = []
                for name in action_types:
                    arr_lst.append(np.load(data_path / (name + ".npy")))
                arr = np.concatenate(arr_lst, axis=0)
            else:
            # A single action type
                arr = np.load(data_path / (action_types + ".npy"))
        else:
        # If argument not given, use all action types
            # action_types = data_path.glob("*.npy")
            action_types = data_path.glob("short_int*.npy")
            arr_lst = []
            for name in action_types:
                arr_lst.append(np.load(data_path / (name.stem + ".npy")))
            arr = np.concatenate(arr_lst, axis=0)

        self.arr = arr
        self.context_frames = arr[:, :num_context_frames, :, :, :]
        self.target_frames = arr[:, num_context_frames:num_context_frames+num_target_frames, :, :, :]

        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.transform = transforms.ToTensor()

    def __len__(self):
        return self.context_frames.shape[0]

    def __getitem__(self, idx):
        f, h, w, c = self.context_frames[0].shape
        context_frames = torch.zeros(3, self.num_context_frames, h, w)
        target_frames = torch.zeros(3, self.num_target_frames, h, w)

        for i in range(self.num_context_frames):
            frame = self.context_frames[idx, i, :, :, :]  # H x W x C
            ts = self.transform(frame)  # C x H x W
            context_frames[:, i, :, :] = ts

        for j in range(self.num_target_frames):
            frame = self.target_frames[idx, j, :, :, :]  # H x W x C
            ts = self.transform(frame)  # C x H x W
            target_frames[:, j, :, :] = ts
        
        # C x F x H x W
        return context_frames, target_frames

class WeizmannDataModule(pl.LightningDataModule):
    def __init__(self, batch_size,
                 num_context_frames, num_target_frames,
                 action_types=None,
                 split_ratio=[0.8, 0.2, 0.0]):

        super().__init__()
        self.batch_size = batch_size
        self.num_context_frames = num_context_frames
        self.num_target_frames = num_target_frames
        self.action_types = action_types
        self.split_ratio = split_ratio    

    def setup(self, stage=None):
        full_dataset = WeizmannDataset(self.num_context_frames,
                                       self.num_target_frames,
                                       self.action_types)
        split = [int(len(full_dataset) * r) for r in self.split_ratio]
        split[2] = len(full_dataset) - sum(split[:2])
        train, val, test = random_split(full_dataset, split,
                                        generator=torch.Generator().manual_seed(42))
        self.split = split
        self.train = train
        self.val = val
        self.test = test
    
    def train_dataloader(self):
        return DataLoader(self.train,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)    

    def test_dataloader(self):
        return DataLoader(self.test,
                          batch_size=self.batch_size,
                          shuffle=False,
                          num_workers=4)   
