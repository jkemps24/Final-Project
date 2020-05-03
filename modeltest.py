import torch
import torch.nn as nn
import torch.optim as optim
from Model import SEIModel
from methods import import_csv


model=SEIModel()

print(model.state_dict())

lr =1e-4

n_epochs = 100

loss_fn=nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(),lr=lr)
Iobs=import_csv('19-04-2020')

S0=torch.FloatTensor([4345737.0,5020302.0,1360396.0,772094.0,523790.0,44598.0,965382.0,38787.0,14446515.0,154748.0,8433301.0,1168423.0,40369.0])
E0=torch.FloatTensor([1,5,1,0,0,0,0,0,10,0,3,1,0])
I0=torch.FloatTensor([0,1,0,0,0,0,0,0,3,0,0,0,0])
nt=Iobs[0].shape
nt=nt[0]
dt=1.0
Sobs=torch.zeros(13,80)
Iobs=import_csv('19-04-2020')
for i in range(13):
    for j in range(80):
        Sobs[i,j]=S0[i]-Iobs[i,j]
tensor_list=[Sobs,Iobs]
obs=torch.stack(tensor_list)

for epoch in range(n_epochs):
    model.train()

    comp=model(S0,E0,I0,dt,nt)

    loss= loss_fn(obs,comp)
    print(loss)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),0.1)
    optimizer.step()
    optimizer.zero_grad()
    print(epoch)
    print(model.state_dict())