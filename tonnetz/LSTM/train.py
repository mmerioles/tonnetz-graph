import torch
import torch.optim as optim 
from torch.utils.data import DataLoader
import numpy as np

import pandas as pd
import ast
from datagenerator import GenerateDataMap
from model import LSTM
import torch.nn as nn
import matplotlib.pyplot as plt



seq_len=30
latent_dim=50
layers_count=2
embedding_dim=30
dropout=0.3
epochs=100
batch_size=128
lr=0.001
train_split=0.9


data=pd.read_csv("D:/aditi/Quarter1/ECE_227/tonnetz-graph/data/lstm_data.csv") #add path
x=np.array([ast.literal_eval(seq) for seq in data['x']],dtype=np.int32)
# print(x)
# x=np.array([np.array(ast.literal_eval(seq), dtype=np.int32) for seq in data['x']])
y=np.array(data['y'],dtype=np.int32)
# print(pd.Series(y).value_counts(normalize=True).head(10))
device = "cuda" if torch.cuda.is_available() else "cpu"
counts=pd.Series(y).value_counts().sort_index()
weights=1/counts
weights=weights/weights.sum()
weights_tensor=torch.tensor(weights.values,dtype=torch.float).to(device)



till=int(len(data)*train_split)
train_dataset=DataLoader(GenerateDataMap(x[:till],y[:till]),batch_size=batch_size,shuffle=True,num_workers=2)
validation_dataset=DataLoader(GenerateDataMap(x[till:],y[till:]),batch_size=batch_size,num_workers=2)



model=LSTM(latent_dim,layers_count,embedding_dim,dropout=dropout).to(device=device)

optimizer=optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion=nn.CrossEntropyLoss(weight=weights_tensor)


scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.3,patience=5)
loss_epoch=[]
val_loss_epoch = []
model.train()

for epoch in range(epochs):
    losses=[]

    for x_batch,y_batch in train_dataset:
        x_batch=x_batch.to(device=device)
        y_batch=y_batch.to(device=device)

        optimizer.zero_grad()
        y_pred_logits,hidden=model(x_batch)
        loss=criterion(y_pred_logits[:,-1,:],y_batch+1)
        loss.backward()
        #skipped clip grad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())
    loss_this_epoch=sum(losses)/len(losses)
    loss_epoch.append(loss_this_epoch)

    model.eval()
    val_losses = []
    with torch.no_grad():
        for x_batch, y_batch in validation_dataset:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            y_pred, _ = model(x_batch)
            loss = criterion(y_pred[:, -1, :], y_batch + 1)
            val_losses.append(loss.item())
    val_loss = sum(val_losses)/len(val_losses)
    val_loss_epoch.append(val_loss)
    model.train()
    # scheduler.step(loss_this_epoch)
    # print(f"epoch {epoch}/{epochs} has loss {loss_this_epoch}")
    print(f"epoch {epoch}/{epochs} | train {loss_this_epoch:.4f} | val {val_loss:.4f}")
    scheduler.step(val_loss)

plt.plot(loss_epoch,label='Train')
plt.plot(val_loss_epoch,label='Val')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()
torch.save(model.state_dict(),'D:/aditi/Quarter1/ECE_227/tonnetz-graph/data/LSTM_checkpt.pth')

