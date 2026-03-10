import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import ast
from torch.utils.data import DataLoader, SubsetRandomSampler
from model import LSTM,notes_class
from datagenerator import GenerateDataMap


def generate_seq(model,seed,length=128,temperature=1.0,top_k=10,device=None):
    generated_seq=[]
    
    # input = torch.tensor(seed,dtype=torch.long,device=device)
    input=seed.unsqueeze(0).to(device)
    # print(seed.shape)
    _,hidden=model(input)
    # print(hidden[0].shape, hidden[1].shape)

    input = seed[-1].reshape(1, 1).to(device)
    # print(input.shape)
    with torch.no_grad():
        for _ in range(length):
            out,hidden=model(input,hidden)
            logit=out[0,0]

            logit=logit/temperature
            top_vals,top_idx=torch.topk(logit,top_k)
            print(logit.argmax().item())  # what's the top prediction before top_k?
            print(logit.topk(3).indices)
            probabilty=F.softmax(top_vals,dim=-1)
            picked=torch.multinomial(probabilty,1) #pick 1 from the multinomial distribution
            token_picked=top_idx[picked].item()

            pred_note=token_picked-1
            generated_seq.append(pred_note)
            input=torch.tensor([[pred_note]],dtype=torch.long,device=device)

    return generated_seq   



latent_dim=50
layers_count=2
embedding_dim=30
dropout=0.3
train_split=0.9

data=pd.read_csv("D:/aditi/Quarter1/ECE_227/tonnetz-graph/data/lstm_data.csv") 
x=np.array([ast.literal_eval(seq) for seq in data['x']],dtype=np.int32)
y=np.array(data['y'],dtype=np.int32)
till=int(len(data)*train_split)
val_size = len(data) - till
indices = np.random.randint(0, val_size, size=500)
# indices=np.random.randint(till+1,len(data),size=500)
sampler=SubsetRandomSampler(indices)
validation_dataset=DataLoader(GenerateDataMap(x[till:],y[till:]),sampler=sampler)

device = "cuda" if torch.cuda.is_available() else "cpu"

model=LSTM(latent_dim,layers_count,embedding_dim,notes_class,dropout)

model.load_state_dict(torch.load('D:/aditi/Quarter1/ECE_227/tonnetz-graph/data/LSTM_checkpt.pth'))
model.to(device=device).eval()
output_path = 'D:/aditi/Quarter1/ECE_227/tonnetz-graph/data/lstm_generated_seq.csv'
with open(output_path,'w') as f:
    for x_b,_ in validation_dataset:
        # print(x_b.shape)
        seed=x_b[0].to(device) 

        output=generate_seq(model=model,seed=seed,length=30,temperature=0.7,top_k=3,device=device)
        # print(output)
        f.write(f"'{output}'\n")









    



# seed=[1,45,2,-1,9,7,0]
# output=generate_seq(model=model,seed=seed,length=30,temperature=0.7,top_k=9)
# print(output)





