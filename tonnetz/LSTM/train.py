import torch
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np
from model import LSTM
import pandas as pd
import ast
import datagenerator
import torch.nn as nn



seq_len=32
latent_dim=256
layers_count=2
embedding_dim=64
dropout=0.3
epochs=50
batch_size=128
lr=0.001
train_split=0.9


data=pd.read_csv("") #add path
x=np.array([ast.literal_eval(seq) for seq in data['x']],dtype=np.int32)
y=np.array(data['y'],dtype=np.int32)

train_dataset=DataLoader(datagenerator.GenerateDataMap(x[:len(data)*train_split],y[:len(data)*train_split]),batch_size=batch_size,shuffle=True)
validation_dataset=DataLoader(datagenerator.GenerateDataMap(x[len(data)*train_split:],y[len(data)*train_split:]),batch_size=batch_size)

device = "cuda" if torch.cuda.is_available() else "cpu"

model=LSTM(latent_dim,layers_count,embedding_dim,dropout=dropout).to(device=device)

optimizer=optim.Adam(model.parameters(),lr=lr,betas=(0.5,0.999))
criterion=nn.CrossEntropyLoss()


scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.3,patience=5)
loss_epoch=[]
model.train()
for epoch in range(epochs):
    losses=[]
    for x_batch,y_batch in train_dataset:
        x_batch=x_batch.to(device=device)
        y_batch=y_batch.to(device=device)

        optimizer.zero_grad()
        y_pred_logits,_=model(x_batch)
        loss=criterion(y_pred_logits[:,-1,:],y_batch+1)
        loss.backward()
        #skipped clip grad
        optimizer.step()
        losses.append(loss.item())
    loss_this_epoch=sum(loss)/len(loss)
    loss_epoch.append(loss_this_epoch)
    print(f"epoch {epoch}/{epochs} has loss {loss_this_epoch}")


torch.save(model.state_dict(),'LSTM_checkpt.pth')


















