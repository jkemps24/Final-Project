import torch
import torch.nn as nn
import torch.optim as optim
from Model import SEIModel
from methods import import_csv


model=SEIModel()

print(model.state_dict())

lr =1e-1

n_epochs = 10000

loss_fn=nn.MSELoss(reduction='mean')
optimizer = optim.SGD(model.parameters(),lr=lr, momentum=0.8)

S0=torch.FloatTensor([4345737.0,5020302.0,1360396.0,772094.0,523790.0,44598.0,965382.0,38787.0,14446515.0,154748.0,8433301.0,1168423.0,40369.0])
E0=torch.zeros(13)
I0=torch.FloatTensor([74,103,7,7,1,0,5,0,177,1,50,7,0])
nt=34
dt=1.0
Sobs=torch.zeros(13,34)
Iobstemp=import_csv('19-04-2020')
Iobs=Iobstemp[:,46:]
for i in range(13):
    for j in range(34):
        Sobs[i,j]=S0[i]-Iobs[i,j]
tensor_list=[Sobs,Iobs]
obs=torch.stack(tensor_list)

for epoch in range(n_epochs):
    model.train()

    comp=model(S0,E0,I0,dt,nt)

    loss= loss_fn(obs,comp)
    print(loss)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(),1)
    optimizer.step()
    optimizer.zero_grad()
    print(epoch)
    print(model.state_dict())
