import numpy as np
from torch.utils.data import Dataset
import torch

class GenerateDataMap(Dataset):
    def __init__(self,seq,target):
        super().__init__()
        self.x=torch.tensor(seq,dtype=torch.long)
        self.y=torch.tensor(target,dtype=torch.long)

def create_seq(notes,seq_len=32): #didnt add stride0
    x=[]
    y=[]
    for i in range(len(notes)-seq_len):
        x.append(notes[i:i+seq_len])
        y.append(notes[i+seq_len])
    return np.array(x),np.array(y)






