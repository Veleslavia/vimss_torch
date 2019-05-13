import torch
import numpy as np


class MockUp(torch.utils.data.Dataset):

    def __len__(self):
        return 2048

    def __getitem__(self, item):
        return np.random.normal(0, 1, (1, 16384)), \
               np.array([np.random.normal(0, 1, (1, 16384)), np.random.normal(0, 1, (1, 16384))])
