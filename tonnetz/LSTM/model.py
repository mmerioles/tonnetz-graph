import torch.nn as nn
import torch
import numpy as np
import torch.functional as F

notes_class=49
class LSTM(nn.Module):
    def __init__(self,latent_dim=10,layer_count=2,embedding_dim=49,notes_class=notes_class,dropout=0.3):
        super().__init__()
        self.latent_dim=latent_dim
        self.layer_count=layer_count
        self.embedding=nn.Embedding(notes_class,embedding_dim)

        self.lstm= nn.LSTM(embedding_dim,latent_dim,layer_count,batch_first=True,dropout=dropout)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(latent_dim,notes_class)

    def forward(self,x,hidden=None):
        x=x.long()+1
        embed=self.embedding(x)
        # hidden=torch.zeros(self.layer_count,x.shape[0],self.latent_dim)
        out,hidden=self.lstm(embed,hidden)
        logits=self.fc(self.dropout(out))

        return logits,hidden
    
    # def init_hidden(self,batch_size,device):

    # def count_params





