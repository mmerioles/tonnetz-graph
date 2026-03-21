import numpy as np
from torch.utils.data import Dataset
import torch
import pandas as pd

class GenerateDataMap(Dataset):
    def __init__(self,seq,target):
        super().__init__()
        self.x=torch.tensor(seq,dtype=torch.long)
        self.y=torch.tensor(target,dtype=torch.long)

    def __len__(self):             
        return len(self.x)

    def __getitem__(self, idx):    
        return self.x[idx], self.y[idx]
    

def create_seq(path_in,path_out,seq_len=31):

    with open(path_in,'r') as f ,open(path_out,'w') as f_out:
        next(f) 
        f_out.write('x,y\n')
        for notes in f :
            values=[int(v) for v in notes.strip().split(',')]
            for i in range(len(values)-seq_len):
                x=values[i:i+seq_len]

                y=values[i+seq_len]
                f_out.write(f'"{x}",{y}\n')

path_in='/data/sequences.csv'
path_out='/data/lstm_data.csv'

create_seq(path_in,path_out)








