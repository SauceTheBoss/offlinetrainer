import torch
import math
import numpy as np
from torch.utils.data import Dataset, Subset

class TrainingSubset(Dataset):
    def __init__(self, valuesDataSet, goodMask, badMask, min_repeat=5):
        self.goodIndices = torch.nonzero(goodMask)
        self.badIndices = torch.nonzero(badMask)
        self.subset_size = max(len(self.goodIndices), len(self.badIndices)) 
        self.size = self.subset_size * min_repeat * (2 if len(self.goodIndices) > 0 and len(self.badIndices) > 0 else 1)

        self.src = valuesDataSet
        self.one = torch.ones(1,device=self.src.device)
        self.zero = torch.zeros(1,device=self.src.device)

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        if len(self.goodIndices) == 0:
            ind_index = index % len(self.badIndices)
            return self.src[ind_index], self.zero
        elif len(self.badIndices) == 0:
            ind_index = index % len(self.goodIndices)
            return self.src[ind_index], self.one
        else:    
            ind_index = int(index / 2)
            if index % 2 == 0:
                ind_index = ind_index % len(self.goodIndices)
                return self.src[ind_index], self.one
            else:       
                ind_index = ind_index % len(self.badIndices)
                return self.src[ind_index], self.zero
    
class ValuesSubset(Subset):
    def __init__(self, dataset, indices):
        self.device = dataset.device
        return super().__init__(dataset, indices)
        
def CreateSubSet(src, mask):
    indices = torch.nonzero(mask)
    return ValuesSubset(src, indices)


class ValuesDataset(Dataset):
    def __init__(self, file_name, device):
        import pickle
        file = open(file_name, 'rb')
        self.memory = pickle.load(file)
        file.close()

        self.device = device
        self.cached_data = []
        self.use_cache = False

    def __len__(self):
        return len(self.memory)

    def __getitem__(self, index):
        if not self.use_cache:
            x = self.makeItem(index) 
            self.cached_data.append(x)
        else:
            x = self.cached_data[index]
        return x

    def makeItem(self, index):
        d1 = self.memory[index]
        d2 = self.memory[(0 if index == 0 else index-1)]

        s1 = torch.tensor(d1['state1'], device=self.device, dtype=torch.float)
        s2 = torch.tensor(d1['state2']-d1['state1'], device=self.device, dtype=torch.float)
        action = torch.tensor(d1['action'], device=self.device, dtype=torch.float)
        prev_action = torch.tensor(d2['action'], device=self.device, dtype=torch.float)

        r1 = torch.rand(s1.size(), device=self.device, dtype=torch.float)
        r2 = torch.rand(s2.size(), device=self.device, dtype=torch.float)

        switch = index % 4
        if switch == 0 and not np.array_equal(d1['state1'],d1['state2']): 
            r1 = s2
            r2 = s1
        elif switch == 1:
            r2 = s2
        elif switch == 2:
            r1 = s1
        return s1, s2, prev_action, action, r1, r2

    # def _get_metadata(self, s1, s2):
    #     x1 = s1[0]
    #     y1 = s1[4]
    #     z1 = s1[8]

    #     x2 = s2[0]
    #     y2 = s2[4]
    #     z2 = s2[8]

    #     v_d = math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    #     va_nom = x1*x2+y1*y2+z1*z2
    #     va_denom = math.sqrt((x1**2+y1**2+z1**2)*(x2**2+y2**2+z2**2))
    #     #v_a = math.acos(va_nom/(0.0001 if va_denom == 0 else va_denom))
    #     v_a = 0

    #     return torch.tensor(np.array([v_d, v_a]), device=self.device, dtype=torch.float)
        
