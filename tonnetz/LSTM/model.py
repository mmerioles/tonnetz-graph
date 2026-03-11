import torch.nn as nn
import torch
import numpy as np
import torch.functional as F

notes_class=51
class LSTM(nn.Module):
    def __init__(self,latent_dim=25,layer_count=2,embedding_dim=50,notes_class=notes_class,dropout=0.5):
        super().__init__()
        self.latent_dim=latent_dim
        self.layer_count=layer_count
        self.embedding=nn.Embedding(notes_class,embedding_dim)

        self.lstm= nn.LSTM(embedding_dim,latent_dim,layer_count,batch_first=True,dropout=dropout)
        self.dropout=nn.Dropout(dropout)
        self.fc=nn.Linear(latent_dim,notes_class)

    def forward(self,x,hidden=None):
        embed=self.embedding(x)
        out,hidden=self.lstm(embed,hidden)
        logits=self.fc(self.dropout(out))

        return logits,hidden






