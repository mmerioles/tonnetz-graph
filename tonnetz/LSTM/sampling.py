import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
from torch.utils.data import DataLoader, SubsetRandomSampler
from datagenerator import GenerateDataMap
from model import LSTM,notes_class


def generate_seq(model,seed,length=128,temperature=1.0,top_k=10,device=None):
    generated_seq=[]

   
    input=seed.unsqueeze(0).to(device)
    _,hidden=model(input)
    input = seed[-1].reshape(1, 1).to(device)
    with torch.no_grad():
        for _ in range(length):
            out,hidden=model(input,hidden)
            logit=out[0,0]

            logit=logit/temperature
            top_vals,top_idx=torch.topk(logit,top_k)
            probabilty=F.softmax(top_vals,dim=-1)
            picked=torch.multinomial(probabilty,1) #pick 1 from the multinomial distribution
            token_picked=top_idx[picked].item()

            pred_note=token_picked
            generated_seq.append(pred_note)
            input=torch.tensor([[pred_note]],dtype=torch.long,device=device)

    return generated_seq,pred_note



latent_dim=25
layers_count=2
embedding_dim=50
dropout=0.3
train_split=0.8

data=pd.read_csv("/data/lstm_data_multisong_8th.csv")
x=np.array([ast.literal_eval(seq) for seq in data['x']],dtype=np.int32)
y=np.array(data['y'],dtype=np.int32)
till=int(len(data)*train_split)
val_size = len(data) - till
indices = np.random.randint(0, val_size, size=500)
sampler=SubsetRandomSampler(indices)
validation_dataset=DataLoader(GenerateDataMap(x[till:],y[till:]),sampler=sampler)

device = "cuda" if torch.cuda.is_available() else "cpu"

model=LSTM(latent_dim,layers_count,embedding_dim,notes_class,dropout)

model.load_state_dict(torch.load('/data/LSTM_checkpt.pth'))
model.to(device=device).eval()
output_path = '/dadta/lstm_generated_seq.csv'
with open(output_path,'w') as f:
    for x_b,_ in validation_dataset:

        seed=x_b[0].to(device)

        output,pred_note=generate_seq(model=model,seed=seed,length=30,temperature=0.7,top_k=6,device=device)
        f.write(f"'{output}'\n")







