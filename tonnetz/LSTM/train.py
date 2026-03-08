import torch
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np
from model import LSTM
import pandas as pd
import ast
import datagenerator
import torch.nn as nn
import matplotlib.pyplot as plt



seq_len=30
latent_dim=10
layers_count=2
embedding_dim=49
dropout=0.3
epochs=100
batch_size=128
lr=0.001
train_split=0.9


data=pd.read_csv("/tonnetz-graph/data/lstm_data.csv") #add path
x=np.array([ast.literal_eval(seq) for seq in data['x']],dtype=np.int32)
# print(x)
# x=np.array([np.array(ast.literal_eval(seq), dtype=np.int32) for seq in data['x']])
y=np.array(data['y'],dtype=np.int32)
till=int(len(data)*train_split)
train_dataset=DataLoader(datagenerator.GenerateDataMap(x[:till],y[:till]),batch_size=batch_size,shuffle=True)
validation_dataset=DataLoader(datagenerator.GenerateDataMap(x[till:],y[till:]),batch_size=batch_size)

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
    loss_this_epoch=sum(losses)/len(losses)
    loss_epoch.append(loss_this_epoch)
    print(f"epoch {epoch}/{epochs} has loss {loss_this_epoch}")

plt.plot(loss_epoch)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
torch.save(model.state_dict(),'/tonnetz-graph/data/LSTM_checkpt.pth')


















