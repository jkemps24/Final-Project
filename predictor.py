import torch
from Model import SEIModel
import matplotlib.pyplot as plt


model=SEIModel()
S0=torch.FloatTensor([4345737.0,5020302.0,1360396.0,772094.0,523790.0,44598.0,965382.0,38787.0,14446515.0,154748.0,8433301.0,1168423.0,40369.0])
E0=torch.zeros(13)
I0=torch.FloatTensor([2562,1618,243,118,257,5,649,0,10578,26,17521,313,9])
nt=400
dt=1.0

comp=model(S0,E0,I0,dt,nt)
Icomp=comp[1,:].detach().numpy()

t=torch.arange(nt)*dt
for i in range (13):
     plt.figure(i)
     plt.plot(t,Icomp[i])
plt.show()