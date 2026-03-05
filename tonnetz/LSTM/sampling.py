import torch
import torch.nn.functional as F
from model import LSTM,notes_class


def generate_seq(model,seed,length=128,temperature=1.0,top_k=10,device=None):
    generated_seq=[]
    
    input = torch.tensor(seed,dtype=torch.long,device=device)
    _,hidden=model(input)

    input = torch.tensor([[seed[-1]]],dtype=torch.long,device=device)
    with torch.no_grad():
        for _ in range(length):
            out,hidden=model(input,hidden)
            logit=out[0,0]

            logit=logit/temperature
            top_vals,top_idx=torch.topk(logit,top_k)
            probabilty=F.softmax(top_vals,dim=-1)
            picked=torch.multinomial(probabilty,1) #pick 1 from the multinomial distribution
            token_picked=top_idx[picked].item()

            pred_note=token_picked-1
            generated_seq.append(pred_note)
            input=torch.tensor([[pred_note]],dtype=torch.long,device=device)

    return generated_seq   



latent_dim=256
layers_count=2
embedding_dim=64
dropout=0.3

device = "cuda" if torch.cuda.is_available() else "cpu"

model=LSTM(latent_dim,layers_count,embedding_dim,notes_class,dropout)

model.load_state_dict(torch.load('LSTM_checkpt.pth'))
model.to(device=device).eval()

seed=[1,45,2,-1,9,7,0]
output=generate_seq(model=model,seed=seed,length=50,temperature=0.7,top_k=9)
print(output)





