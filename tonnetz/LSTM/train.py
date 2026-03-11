import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np

import pandas as pd
import ast
from model import LSTM,notes_class
from datagenerator import GenerateDataMap
import torch.nn as nn
import matplotlib.pyplot as plt



seq_len=30
latent_dim=25
layers_count=2
embedding_dim=50
dropout=0.5
epochs=70
batch_size=128
lr=0.001
train_split=0.9


data=pd.read_csv("/data/lstm_data_multisong_8th.csv") #add path
x=np.array([ast.literal_eval(seq) for seq in data['x']],dtype=np.int32)
y=np.array(data['y'],dtype=np.int32)

device = "cuda" if torch.cuda.is_available() else "cpu"
counts=pd.Series(y).value_counts().sort_index()




till=int(len(data)*train_split)
train_dataset=DataLoader(GenerateDataMap(x[:till],y[:till]),batch_size=batch_size,shuffle=True,num_workers=2)
validation_dataset=DataLoader(GenerateDataMap(x[till:],y[till:]),batch_size=batch_size,num_workers=2)



model=LSTM(latent_dim,layers_count,embedding_dim,dropout=dropout).to(device=device)

optimizer=optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))
criterion=nn.CrossEntropyLoss()


scheduler=optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer,factor=0.3,patience=8)
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
        loss=criterion(y_pred_logits[:,-1,:],y_batch)
        loss.backward()
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
            loss = criterion(y_pred[:, -1, :], y_batch )
            val_losses.append(loss.item())
    val_loss = sum(val_losses)/len(val_losses)
    val_loss_epoch.append(val_loss)
    model.train()
    print(f"epoch {epoch}/{epochs} | train {loss_this_epoch:.4f} | val {val_loss:.4f}")
    scheduler.step(val_loss)

plt.plot(loss_epoch,label='Train')
plt.plot(val_loss_epoch,label='Val')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()
torch.save(model.state_dict(),'/data/LSTM_checkpt.pth')

